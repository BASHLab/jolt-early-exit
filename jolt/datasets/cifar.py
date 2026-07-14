"""CIFAR-10 / CIFAR-100 dataloaders.

One loader serves both datasets (the only difference is the dataset class, normalization
stats, and class count). The validation split is carved from the training set with a seeded
permutation and uses the test-time transform (no augmentation), which is what calibration
needs.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from ..config import make_generator, seed_worker

# Per-channel mean/std from the source repos.
_STATS = {
    "cifar10": (
        (0.49139968, 0.48215827, 0.44653124),
        (0.24703233, 0.24348505, 0.26158768),
    ),
    "cifar100": (
        (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
        (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
    ),
}
_NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


def cifar_dataloaders(
    *,
    name: str = "cifar100",
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    val_size: int = 5000,
    download: bool = False,
    seed: int = 42,
    randaugment: bool = False,
    randaugment_num_ops: int = 2,
    randaugment_magnitude: int = 9,
    cutout: float = 0.0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if name not in _STATS:
        raise ValueError(f"Unknown CIFAR dataset '{name}' (expected cifar10 or cifar100).")
    mean, std = _STATS[name]
    ops = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    if randaugment:
        ops.append(T.RandAugment(num_ops=randaugment_num_ops, magnitude=randaugment_magnitude))
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    if cutout > 0.0:
        ops.append(T.RandomErasing(p=cutout))
    train_tf = T.Compose(ops)
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    ds_cls = torchvision.datasets.CIFAR10 if name == "cifar10" else torchvision.datasets.CIFAR100
    train_aug = ds_cls(root=root, train=True, download=download, transform=train_tf)
    train_plain = ds_cls(root=root, train=True, download=False, transform=test_tf)
    test = ds_cls(root=root, train=False, download=download, transform=test_tf)

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_aug), generator=generator).tolist()
    val_idx, train_idx = perm[:val_size], perm[val_size:]
    train_subset = Subset(train_aug, train_idx)
    val_subset = Subset(train_plain, val_idx)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        generator=make_generator(seed), worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def cifar_num_classes(name: str) -> int:
    return _NUM_CLASSES[name]
