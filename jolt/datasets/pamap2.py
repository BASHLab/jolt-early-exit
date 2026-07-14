"""PAMAP2 (Physical Activity Monitoring) dataloaders for the EE-HAR cell.

Uses the pre-windowed NPY layout under ``<root>/subject10[1-9].dat/{ankle,chest,hand,label}.npy``.
Each ``.npy`` is shape ``(n_windows, window_len, n_channels)`` for the IMU files (typical
defaults: 6 channels = 3-axis accelerometer + 3-axis gyroscope per location) and ``(n_windows,)``
strings for the labels. Concatenating ankle / chest / hand along channels yields the canonical
18-channel DeepConvLSTM input.

Activity label 0 (transient periods) is dropped per PAMAP2 documentation. Remaining IDs are
mapped densely to ``[0, num_classes - 1]`` via the union across subjects.

LOSO is the headline protocol: ``test_subject`` is held out, all other subjects are training.
A small held-out fraction of training windows becomes the validation split for threshold
calibration. ``num_classes`` is determined from the union, so the same loader produces the same
label space across folds.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from ..config import make_generator, seed_worker
from .uci_har import _JitterScale

SENSORS = ("ankle", "chest", "hand")
DEFAULT_SUBJECTS = tuple(f"subject10{i}" for i in range(1, 10))


def _load_subject(root: Path, subject: str) -> Tuple[np.ndarray, np.ndarray]:
    base = root / f"{subject}.dat"
    channels = []
    for sensor in SENSORS:
        arr = np.load(base / f"{sensor}.npy", allow_pickle=True).astype(np.float32)
        if arr.ndim != 3:
            raise ValueError(f"{sensor}.npy expected shape (n, T, C), got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{sensor}.npy contains non-finite values for {subject}.")
        channels.append(arr)
    n_windows = channels[0].shape[0]
    window_len = channels[0].shape[1]
    if any(c.shape[0] != n_windows or c.shape[1] != window_len for c in channels):
        raise ValueError(f"Per-sensor window count or window length disagree for {subject}.")
    # (n, T, C_total) where C_total = sum of per-sensor channels (typically 6*3 = 18)
    stacked = np.concatenate(channels, axis=-1)
    # Permute to (n, C, T) for Conv1d.
    stacked = np.transpose(stacked, (0, 2, 1))
    labels = np.load(base / "label.npy", allow_pickle=True).astype(str)
    return stacked, labels


def _per_channel_stats(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std across (n_windows, T) for a (n_windows, C, T) array."""
    flat = x.transpose(1, 0, 2).reshape(x.shape[1], -1)
    mean = flat.mean(axis=1).astype(np.float32)
    std = flat.std(axis=1).astype(np.float32)
    std = np.where(std < 1e-6, np.float32(1.0), std)
    return mean, std


def _apply_normalization(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean[None, :, None]) / std[None, :, None]


def _drop_transients(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    keep = y != "0"
    return x[keep], y[keep]


def _build_label_map(root: Path, subjects: List[str]) -> dict:
    seen = set()
    for s in subjects:
        labels = np.load(root / f"{s}.dat" / "label.npy", allow_pickle=True).astype(str)
        seen.update(labels.tolist())
    seen.discard("0")
    # Sort by integer interpretation so the map is stable / interpretable.
    ordered = sorted(seen, key=lambda v: int(v))
    return {label: idx for idx, label in enumerate(ordered)}


def _to_tensor_dataset(x: np.ndarray, y: np.ndarray, label_map: dict) -> TensorDataset:
    y_int = np.array([label_map[lbl] for lbl in y], dtype=np.int64)
    return TensorDataset(torch.from_numpy(x), torch.from_numpy(y_int))


def pamap2_dataloaders(
    *,
    root: str,
    test_subject: str = "subject106",
    val_subject: str = "subject105",
    train_subjects: Tuple[str, ...] = (),
    batch_size: int = 256,
    num_workers: int = 2,
    val_size: int = 512,
    seed: int = 42,
    jitter: float = 0.0,
    scale: float = 0.0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    root_path = Path(root)
    if not train_subjects:
        train_subjects = tuple(
            s for s in DEFAULT_SUBJECTS if s not in (test_subject, val_subject)
        )

    subjects_for_map = list(train_subjects) + [val_subject, test_subject]
    label_map = _build_label_map(root_path, subjects_for_map)

    x_train_parts, y_train_parts = [], []
    for s in train_subjects:
        x, y = _load_subject(root_path, s)
        x, y = _drop_transients(x, y)
        x_train_parts.append(x)
        y_train_parts.append(y)
    x_train = np.concatenate(x_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)

    x_val, y_val = _load_subject(root_path, val_subject)
    x_val, y_val = _drop_transients(x_val, y_val)

    x_test, y_test = _load_subject(root_path, test_subject)
    x_test, y_test = _drop_transients(x_test, y_test)

    train_mean, train_std = _per_channel_stats(x_train)
    x_train = _apply_normalization(x_train, train_mean, train_std)
    x_val = _apply_normalization(x_val, train_mean, train_std)
    x_test = _apply_normalization(x_test, train_mean, train_std)

    train_ds = _to_tensor_dataset(x_train, y_train, label_map)
    val_ds = _to_tensor_dataset(x_val, y_val, label_map)
    test_ds = _to_tensor_dataset(x_test, y_test, label_map)

    train_subset = train_ds
    val_subset = val_ds
    if jitter > 0 or scale > 0:
        train_subset = _JitterScale(train_subset, jitter=jitter, scale=scale)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        generator=make_generator(seed), worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
