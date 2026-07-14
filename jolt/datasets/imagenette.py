"""Imagenette dataloaders (10-class ImageNet subset).

Follows the source setup: images are resized to 32x32 (so the same MobileNetV2 used for CIFAR
applies) and normalized with ImageNet statistics. Imagenette ships only train/val splits, so
the held-out "val" split is the test set and a seeded calibration split is carved from train
(with the test-time transform), mirroring the CIFAR loader.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from ..config import make_generator, seed_worker

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 10


def imagenette_dataloaders(
    *,
    root: str,
    batch_size: int = 128,
    num_workers: int = 2,
    image_size: int = 32,
    val_size: int = 2000,
    download: bool = False,
    seed: int = 42,
    randaugment: bool = False,
    randaugment_num_ops: int = 2,
    randaugment_magnitude: int = 9,
    cutout: float = 0.0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    ops = [
        T.Resize((image_size, image_size)),
        T.RandomCrop(image_size, padding=4),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
    ]
    if randaugment:
        ops.append(T.RandAugment(num_ops=randaugment_num_ops, magnitude=randaugment_magnitude))
    ops += [T.ToTensor(), T.Normalize(_MEAN, _STD)]
    if cutout > 0.0:
        ops.append(T.RandomErasing(p=cutout))
    train_tf = T.Compose(ops)
    test_tf = T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), T.Normalize(_MEAN, _STD)])

    train_aug = torchvision.datasets.Imagenette(root=root, split="train", size="320px", download=download, transform=train_tf)
    train_plain = torchvision.datasets.Imagenette(root=root, split="train", size="320px", download=False, transform=test_tf)
    test = torchvision.datasets.Imagenette(root=root, split="val", size="320px", download=False, transform=test_tf)

    generator = make_generator(seed)
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
