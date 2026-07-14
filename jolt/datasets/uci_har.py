"""UCI Human Activity Recognition (IMU) dataloaders.

Loads the nine raw inertial-signal channels (body acc / body gyro / total acc, each xyz) of
128-timestep windows and reshapes each window to ``(n_steps, n_length, channels)`` so a 2D
ResNet can consume it. With the defaults ``(4, 32, 9)`` the 128 timesteps become 4 "channels"
of 32 timesteps each, with the 9 signals on the width axis. This reshape is the
dataset-specific hack carried over from the source; ``ResNet`` therefore takes
``in_channels=4`` (see :mod:`jolt.models.resnet_exit`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from ..config import make_generator, seed_worker


class _JitterScale(Dataset):
    """Per-window IMU augmentation: additive Gaussian jitter and a multiplicative scale.

    # RESEARCH GAP: Um et al. 2017 also use rotation (of the xyz sensor triplets) and
    # time-warping; those are a future addition. Jitter + scaling are the cheapest, most common.
    """

    def __init__(self, base: Dataset, *, jitter: float = 0.0, scale: float = 0.0):
        self.base = base
        self.jitter = jitter
        self.scale = scale

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        x, y = self.base[index]
        if self.jitter > 0:
            x = x + torch.randn_like(x) * self.jitter
        if self.scale > 0:
            x = x * (1.0 + torch.randn(1).item() * self.scale)
        return x, y

_SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]
NUM_CLASSES = 6


def _load_signals(root: Path, subset: str) -> np.ndarray:
    arrays = []
    for signal in _SIGNALS:
        path = root / subset / "Inertial Signals" / f"{signal}_{subset}.txt"
        arrays.append(np.loadtxt(path))  # (N, 128)
    return np.stack(arrays, axis=-1).astype(np.float32)  # (N, 128, 9)


def _load_labels(root: Path, subset: str) -> np.ndarray:
    labels = np.loadtxt(root / subset / f"y_{subset}.txt").astype(np.int64)
    return labels - 1  # source labels are 1..6


def uci_har_dataloaders(
    *,
    root: str,
    batch_size: int = 128,
    num_workers: int = 2,
    val_size: int = 1000,
    seed: int = 42,
    n_steps: int = 4,
    n_length: int = 32,
    jitter: float = 0.0,
    scale: float = 0.0,
    layout: str = "2d",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build UCI-HAR dataloaders.

    ``layout="2d"`` (default): reshape each window to ``(n_steps, n_length, 9)`` so a 2D
    ResNet can consume it (legacy v3 setup with ResNet-18; non-canonical for EE-HAR).
    ``layout="1d"``: return ``(9, 128)`` per window (channels-first), the canonical 1D-CNN
    input expected by the v4 brief's UCI-HAR cell. Per Jordao et al. 2020, EE-HAR papers
    use 1D backbones over the 9-channel inertial signal × 128 timesteps; the 2D reshape is
    a v3 hack we are moving off of.
    """
    if layout not in ("1d", "2d"):
        raise ValueError(f"layout must be '1d' or '2d'; got {layout!r}")

    root_path = Path(root)
    x_train = _load_signals(root_path, "train")  # (N, 128, 9)
    y_train = _load_labels(root_path, "train")
    x_test = _load_signals(root_path, "test")
    y_test = _load_labels(root_path, "test")

    channels = x_train.shape[-1]
    if layout == "2d":
        x_train = x_train.reshape(-1, n_steps, n_length, channels)
        x_test = x_test.reshape(-1, n_steps, n_length, channels)
    else:
        # 1D layout: transpose (N, 128, 9) -> (N, 9, 128) so Conv1d sees the 9 inertial
        # channels and 128-timestep sequence dimension. No reshape needed.
        x_train = np.transpose(x_train, (0, 2, 1)).copy()  # (N, 9, 128)
        x_test = np.transpose(x_test, (0, 2, 1)).copy()

    train_full = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_full), generator=generator).tolist()
    val_idx, train_idx = perm[:val_size], perm[val_size:]
    train_subset = Subset(train_full, train_idx)
    val_subset = Subset(train_full, val_idx)
    if jitter > 0 or scale > 0:
        train_subset = _JitterScale(train_subset, jitter=jitter, scale=scale)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        generator=make_generator(seed), worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
