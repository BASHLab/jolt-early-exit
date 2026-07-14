"""Text-domain distribution-shift evaluation for the SST-2 BERT-base cell.

Two shift axes, both same-label-space so the trained heads apply unchanged:

  imdb   cross-domain sentiment transfer SST-2 -> IMDB (Maas et al. 2011),
         25k test reviews. The canonical domain-shift pairing used by the
         early-exit threshold-adaptation line (CeeBERT / DAdEE / UAT), so the
         eventual head-to-head comparison shares the protocol. Long reviews
         are truncated to the cell's max_len like any SST-2 input.
         Reported as a single shift condition (no severity tiers).

  typo   character-level noise on the SST-2 eval half (Belinkov & Bisk 2018
         precedent): keyboard-adjacent substitutions, deletions, insertions,
         and transpositions at per-character corruption rates
         {0.05, 0.10, 0.20} for severities {1, 3, 5}. Deterministic per
         (sentence index, severity).

Same protocol as the other *_shift_eval scripts: clean-validation quantile
thresholds -> absolute vs quantile policy on the shifted stream. Uses the
text collection path (collect_exit_matrix_text).

Usage:
    python scripts/text_shift_eval.py [--force]
"""

from __future__ import annotations

import argparse
import json
import string
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from jolt.budgeted import (
    DEFAULT_Q_GRID, collect_exit_matrix_text, simulate_routing,
    thresholds_for_population,
)
from jolt.datasets.glue import build_tokenizer

from budgeted_eval import CELLS  # noqa: E402
from shift_eval import load_model  # noqa: E402

SEVERITIES = [1, 3, 5]
_RATE = {1: 0.05, 3: 0.10, 5: 0.20}

_KEYBOARD = {
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh",
    "u": "yij", "i": "uok", "o": "ipl", "p": "ol", "a": "qsz", "s": "awdx",
    "d": "sefc", "f": "drgv", "g": "fthb", "h": "gyjn", "j": "hukm",
    "k": "jil", "l": "kop", "z": "asx", "x": "zsdc", "c": "xdfv",
    "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk",
}


def typo_corrupt(text: str, rate: float, rng: np.random.Generator) -> str:
    chars = list(text)
    out = []
    i = 0
    while i < len(chars):
        ch = chars[i]
        if ch.isalpha() and rng.random() < rate:
            op = rng.integers(4)
            low = ch.lower()
            if op == 0 and low in _KEYBOARD:  # keyboard-adjacent substitution
                sub = _KEYBOARD[low][int(rng.integers(len(_KEYBOARD[low])))]
                out.append(sub.upper() if ch.isupper() else sub)
            elif op == 1:  # deletion
                pass
            elif op == 2:  # insertion
                out.append(ch)
                out.append(string.ascii_lowercase[int(rng.integers(26))])
            else:  # transposition with next char
                if i + 1 < len(chars):
                    out.append(chars[i + 1])
                    out.append(ch)
                    i += 1
                else:
                    out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def make_text_loader(texts, labels, tokenizer, max_len: int, batch_size: int = 64):
    from torch.utils.data import DataLoader, Dataset

    enc = tokenizer(list(texts), truncation=True, max_length=max_len,
                    padding="max_length", return_tensors="pt")
    labels_t = torch.as_tensor(list(labels), dtype=torch.long)

    class _DS(Dataset):
        def __len__(self):
            return labels_t.size(0)

        def __getitem__(self, i):
            item = {k: v[i] for k, v in enc.items()}
            item["labels"] = labels_t[i]
            return item

    return DataLoader(_DS(), batch_size=batch_size, shuffle=False, num_workers=0)


def sst2_eval_half(seed: int = 42):
    """The eval half of the GLUE SST-2 validation split (matches glue_dataloaders)."""
    from datasets import load_dataset
    raw = load_dataset("glue", "sst2")["validation"].shuffle(seed=seed)
    half = len(raw) // 2
    part = raw.select(range(half, len(raw)))
    return list(part["sentence"]), list(part["label"])


def imdb_test():
    from datasets import load_dataset
    ds = load_dataset("imdb")["test"]
    return list(ds["text"]), list(ds["label"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cell", default="SST-2", choices=["SST-2", "SST-2-E3612"])
    args = ap.parse_args()

    cell = CELLS[args.cell]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Text shift eval (SST-2 cell) on {device}")

    run_dirs = []
    for root_rel in cell["roots"]:
        root = REPO / root_rel
        if not root.exists():
            continue
        for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
            for seed_dir in sorted(mdir.glob("seed*")):
                if (seed_dir / "checkpoint.pt").exists() and (seed_dir / "metrics.json").exists():
                    run_dirs.append(seed_dir)

    sst_texts, sst_labels = sst2_eval_half()
    imdb_texts, imdb_labels = imdb_test()
    print(f"SST-2 eval half: {len(sst_texts)}; IMDB test: {len(imdb_texts)}")

    tok_cache = {}
    for run_dir in run_dirs:
        out_json = run_dir / "text_shift_eval.json"
        if out_json.exists() and not args.force:
            print(f"  [skip] {run_dir.relative_to(REPO)}")
            continue
        try:
            loaded = load_model(run_dir, cell, device)
        except Exception as exc:
            print(f"  [error] {run_dir.relative_to(REPO)}: {exc}")
            continue
        if loaded is None:
            continue
        model, method, cutoff, pem, cfg = loaded
        max_len = int(getattr(cfg.data, "max_len", 128))
        pn = cfg.model.pretrained_name
        if pn not in tok_cache:
            tok_cache[pn] = build_tokenizer(pn)
        tokenizer = tok_cache[pn]

        # Clean validation matrices: text path has no cached npz convention yet,
        # so collect from the calibration half via the standard loaders.
        from jolt.train import _build_dataloaders
        _, val_loader, _ = _build_dataloaders(cfg)
        mats_val = collect_exit_matrix_text(model, val_loader, device=device, cutoff_type=cutoff)
        thr_by_q = {q: thresholds_for_population(mats_val["scores"], q) for q in DEFAULT_Q_GRID}

        conditions = {}
        conditions["imdb"] = make_text_loader(imdb_texts, imdb_labels, tokenizer, max_len)
        for sev in SEVERITIES:
            rng = np.random.default_rng(1000 + sev)
            corrupted = [typo_corrupt(t, _RATE[sev], rng) for t in sst_texts]
            conditions[f"typo_{sev}"] = make_text_loader(corrupted, sst_labels, tokenizer, max_len)

        results = {}
        npz_payload = {}
        for name, loader in conditions.items():
            mats_c = collect_exit_matrix_text(model, loader, device=device, cutoff_type=cutoff)
            npz_payload[f"scores_{name}"] = mats_c["scores"].astype(np.float16)
            npz_payload[f"correct_{name}"] = mats_c["correct"].astype(np.uint8)
            rows = []
            for q, thr in thr_by_q.items():
                st = simulate_routing(mats_c["scores"], mats_c["correct"], thr, pem)
                thr_q = thresholds_for_population(mats_c["scores"], q)
                st_q = simulate_routing(mats_c["scores"], mats_c["correct"], thr_q, pem)
                rows.append({"q": q,
                             "accuracy": st["accuracy"],
                             "exit_counts": st["exit_counts"],
                             "macs_frac": st["macs_frac"],
                             "q_accuracy": st_q["accuracy"],
                             "q_exit_counts": st_q["exit_counts"],
                             "q_macs_frac": st_q["macs_frac"]})
            results[name] = {
                "n": int(mats_c["labels"].shape[0]),
                "deep_acc": float(mats_c["correct"][:, -1].mean()),
                "rows": rows,
            }
            print(f"  [{run_dir.relative_to(REPO)}] {method} {name} done")
        np.savez_compressed(run_dir / "exit_scores_tshift.npz", **npz_payload)
        out_json.write_text(json.dumps({
            "method": method, "cutoff_type": cutoff, "per_exit_macs": pem,
            "results": results,
        }, indent=None))
        print(f"  [done] {run_dir.relative_to(REPO)} method={method}")
    print("all done")


if __name__ == "__main__":
    main()
