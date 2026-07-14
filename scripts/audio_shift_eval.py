"""Audio-domain distribution-shift evaluation for GSC v2 and ESC-50.

Additive-noise shift at controlled SNR, mixed into the waveform BEFORE the
mel-spectrogram front-end (i.e., a genuine acoustic-channel shift, not a
feature-space perturbation):

  background  real background recordings shipped with GSC v2
              (_background_noise_: doing_the_dishes, exercise_bike, pink noise,
              running_tap, ...), random crop per clip
  white       Gaussian white noise

SNR tiers map severities {1, 3, 5} -> {20, 10, 0} dB. Mixing is deterministic
per (clip index, kind, severity).

Same evaluation protocol as shift_eval.py / sensor_shift_eval.py: clean-val
quantile thresholds -> absolute policy vs quantile policy on the shifted test
stream; per-run npz + json.

Usage:
    python scripts/audio_shift_eval.py --cell GSC-v2 [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from jolt.budgeted import (
    DEFAULT_Q_GRID, collect_exit_matrix, simulate_routing, thresholds_for_population,
)
from jolt.datasets.esc50 import _ESC50Dataset, TEST_FOLD
from jolt.datasets.esc50 import SAMPLE_RATE as ESC_SR
from jolt.datasets.gsc_v2 import _GSCv2Dataset

from budgeted_eval import CELLS  # noqa: E402
from shift_eval import clean_val_matrices, load_model  # noqa: E402

SEVERITIES = [1, 3, 5]
SNR_DB = {1: 20.0, 3: 10.0, 5: 0.0}
KINDS = ["background", "white"]

GSC_NOISE_DIR = REPO / "data/SpeechCommands/speech_commands_v0.02/_background_noise_"


def _load_noise_bank(target_sr: int) -> list:
    bank = []
    for wav_path in sorted(GSC_NOISE_DIR.glob("*.wav")):
        wav, sr = torchaudio.load(str(wav_path))
        wav = wav.mean(dim=0)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        bank.append(wav)
    if not bank:
        raise FileNotFoundError(f"no noise WAVs under {GSC_NOISE_DIR}")
    return bank


def _mix_at_snr(wav: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    p_sig = wav.pow(2).mean().clamp_min(1e-12)
    p_noise = noise.pow(2).mean().clamp_min(1e-12)
    scale = torch.sqrt(p_sig / (p_noise * (10.0 ** (snr_db / 10.0))))
    return wav + scale * noise


def _noise_for(bank: list, n_samples: int, kind: str, sev: int, index: int) -> torch.Tensor:
    rng = np.random.default_rng(hash((kind, sev, index)) % 2**31)
    if kind == "white":
        return torch.from_numpy(rng.normal(0.0, 1.0, size=n_samples).astype(np.float32))
    src = bank[int(rng.integers(len(bank)))]
    if src.numel() <= n_samples:
        reps = int(np.ceil(n_samples / src.numel()))
        src = src.repeat(reps)
    start = int(rng.integers(0, src.numel() - n_samples + 1))
    return src[start:start + n_samples]


class GSCShifted(_GSCv2Dataset):
    def __init__(self, root: str, kind: str, sev: int, bank: list):
        super().__init__(root, "testing")
        self._kind, self._sev, self._bank = kind, sev, bank

    def __getitem__(self, index: int):
        waveform, sr, label, _sid, _un = self.base[index]
        wav = waveform[0]
        n = 16000
        if wav.numel() < n:
            wav = torch.nn.functional.pad(wav, (0, n - wav.numel()))
        elif wav.numel() > n:
            wav = wav[:n]
        noise = _noise_for(self._bank, n, self._kind, self._sev, index)
        wav = _mix_at_snr(wav, noise, SNR_DB[self._sev])
        mel = self.melspec(wav.unsqueeze(0))
        mel = torch.log(mel.clamp_min(1e-10))
        from jolt.datasets.gsc_v2 import _LABEL_TO_IDX
        return mel, _LABEL_TO_IDX[label]


class ESC50Shifted(_ESC50Dataset):
    def __init__(self, root: str, kind: str, sev: int, bank: list):
        super().__init__(root, folds=(TEST_FOLD,))
        self._kind, self._sev, self._bank = kind, sev, bank

    def __getitem__(self, idx: int):
        path, label = self.entries[idx]
        wav = self._load(path)
        noise = _noise_for(self._bank, wav.numel(), self._kind, self._sev, idx)
        wav = _mix_at_snr(wav, noise, SNR_DB[self._sev])
        spec = self.mel(wav.unsqueeze(0))
        return torch.log1p(spec), label


def shifted_loader(cell_name: str, cell: dict, kind: str, sev: int, bank: list) -> DataLoader:
    if cell_name == "GSC-v2":
        ds = GSCShifted(cell["data_root"], kind, sev, bank)
    else:
        # esc50 loader appends ESC-50-master under the config data root
        ds = ESC50Shifted(str(Path(cell["data_root"]) / "ESC-50-master"), kind, sev, bank)
    return DataLoader(ds, batch_size=128, shuffle=False, num_workers=4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", required=True, choices=["GSC-v2", "ESC-50"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cell = CELLS[args.cell]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_sr = 16000 if args.cell == "GSC-v2" else ESC_SR
    bank = _load_noise_bank(target_sr)
    print(f"Audio shift eval for {args.cell} on {device} ({len(bank)} noise sources)")

    run_dirs = []
    for root_rel in cell["roots"]:
        root = REPO / root_rel
        if not root.exists():
            continue
        for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
            for seed_dir in sorted(mdir.glob("seed*")):
                if (seed_dir / "checkpoint.pt").exists() and (seed_dir / "metrics.json").exists():
                    run_dirs.append(seed_dir)

    for run_dir in run_dirs:
        out_json = run_dir / "audio_shift_eval.json"
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
        try:
            mats_val = clean_val_matrices(run_dir, model, cfg, cutoff, device)
        except Exception as exc:
            print(f"  [error] {run_dir.relative_to(REPO)} (val matrices): {exc}")
            continue
        thr_by_q = {q: thresholds_for_population(mats_val["scores"], q) for q in DEFAULT_Q_GRID}

        results = {}
        npz_payload = {}
        for kind in KINDS:
            results[kind] = {}
            for sev in SEVERITIES:
                loader = shifted_loader(args.cell, cell, kind, sev, bank)
                mats_c = collect_exit_matrix(model, loader, device=device, cutoff_type=cutoff)
                npz_payload[f"scores_{kind}_{sev}"] = mats_c["scores"].astype(np.float16)
                npz_payload[f"correct_{kind}_{sev}"] = mats_c["correct"].astype(np.uint8)
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
                results[kind][str(sev)] = {
                    "deep_acc": float(mats_c["correct"][:, -1].mean()),
                    "rows": rows,
                }
            print(f"  [{run_dir.relative_to(REPO)}] {method} {kind} done")
        np.savez_compressed(run_dir / "exit_scores_ashift.npz", **npz_payload)
        out_json.write_text(json.dumps({
            "method": method, "cutoff_type": cutoff, "per_exit_macs": pem,
            "snr_db": SNR_DB, "results": results,
        }, indent=None))
        print(f"  [done] {run_dir.relative_to(REPO)} method={method}")
    print("all done")


if __name__ == "__main__":
    main()
