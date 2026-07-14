"""GLUE dataset loaders (HuggingFace ``datasets`` + a transformers tokenizer).

Network-gated: the first use downloads the GLUE task and the tokenizer. ``datasets`` and
``transformers`` are imported lazily so the rest of JOLT does not depend on them.

GLUE test labels are private, so the public validation split is the de-facto test set. We
split it (seeded) into a calibration half (for threshold estimation) and an evaluation half.
"""

from __future__ import annotations

from typing import Optional, Tuple

from ..config import make_generator, seed_worker

TASK_KEYS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
TASK_NUM_LABELS = {
    "cola": 2, "sst2": 2, "mrpc": 2, "qqp": 2, "stsb": 1,
    "mnli": 3, "qnli": 2, "rte": 2, "wnli": 2,
}


def glue_num_labels(task: str) -> int:
    return TASK_NUM_LABELS[task]


def _validation_split_name(task: str) -> str:
    return "validation_matched" if task == "mnli" else "validation"


def build_tokenizer(pretrained_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(pretrained_name)


def glue_dataloaders(
    *,
    task: str,
    tokenizer,
    max_len: int = 128,
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[object, object, object]:
    """Return (train_loader, calibration_loader, eval_loader). Calibration and eval are the
    two seeded halves of the GLUE validation split."""
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    if task not in TASK_KEYS:
        raise ValueError(f"Unknown GLUE task '{task}'.")
    field1, field2 = TASK_KEYS[task]
    raw = load_dataset("glue", task)

    def tokenize(batch):
        if field2 is None:
            return tokenizer(batch[field1], truncation=True, max_length=max_len)
        return tokenizer(batch[field1], batch[field2], truncation=True, max_length=max_len)

    columns = raw["train"].column_names
    remove = [c for c in columns if c != "label"]
    tokenized = raw.map(tokenize, batched=True, remove_columns=remove)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(
        tokenized["train"], batch_size=batch_size, shuffle=True,
        collate_fn=collator, num_workers=num_workers,
        generator=make_generator(seed), worker_init_fn=seed_worker,
    )

    val = tokenized[_validation_split_name(task)].shuffle(seed=seed)
    half = len(val) // 2
    calib = val.select(range(half))
    evaluation = val.select(range(half, len(val)))
    calib_loader = DataLoader(calib, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=num_workers)
    eval_loader = DataLoader(evaluation, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=num_workers)
    return train_loader, calib_loader, eval_loader
