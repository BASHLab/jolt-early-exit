"""Tiny-ImageNet (200-class, 64x64) dataloaders.

Tiny-ImageNet is the canonical "harder-than-CIFAR, cheaper-than-ImageNet" benchmark used by
recent EE work (ZTW NeurIPS 2021, CGT 2025, LayerSkip 2024) at the on-device backbone size
class. Layout produced by the official ``tiny-imagenet-200.zip`` archive:

    <root>/tiny-imagenet-200/
        train/<wnid>/images/*.JPEG   (500 per class)
        val/images/*.JPEG            (10 000 total)
        val/val_annotations.txt      (filename -> wnid mapping)
        test/images/*.JPEG           (10 000 unlabeled; ignored here)

This loader will download and extract the archive on first call when ``download=True``, and
reorganises ``val/`` into a per-wnid folder layout so ``torchvision.datasets.ImageFolder``
can read it. The public test set has no released labels, so the validation split (10k
images, labelled) is used as the held-out test set; ``val_size`` is carved from the training
set for threshold calibration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from ..config import make_generator, seed_worker

_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
_FOLDER = "tiny-imagenet-200"
# ImageNet-1k stats are the standard normalisation for Tiny-ImageNet (Le & Yang 2015).
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 200


def _ensure_val_per_class(val_dir: Path) -> None:
    """Reorganise <val>/images + val_annotations.txt into <val>/<wnid>/<filename>.JPEG once."""
    images_dir = val_dir / "images"
    ann_path = val_dir / "val_annotations.txt"
    if not images_dir.exists():
        return  # already reorganised on a prior run
    if not ann_path.exists():
        raise FileNotFoundError(f"Expected {ann_path}; archive may be incomplete.")
    mapping = {}
    with ann_path.open("r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]
    for filename, wnid in mapping.items():
        src = images_dir / filename
        if not src.exists():
            continue
        dst_dir = val_dir / wnid
        dst_dir.mkdir(exist_ok=True)
        src.rename(dst_dir / filename)
    try:
        images_dir.rmdir()
    except OSError:
        pass


def _maybe_download(root: Path, download: bool) -> Path:
    dataset_dir = root / _FOLDER
    if dataset_dir.exists():
        return dataset_dir
    if not download:
        raise FileNotFoundError(
            f"Tiny-ImageNet not found at {dataset_dir} and download=False. "
            f"Pass download=True or unpack {_URL} into {root}."
        )
    root.mkdir(parents=True, exist_ok=True)
    torchvision.datasets.utils.download_and_extract_archive(_URL, str(root))
    return dataset_dir


def tiny_imagenet_dataloaders(
    *,
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
    root_path = Path(root)
    dataset_dir = _maybe_download(root_path, download)
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    _ensure_val_per_class(val_dir)

    train_ops = [T.RandomCrop(64, padding=8), T.RandomHorizontalFlip()]
    if randaugment:
        train_ops.append(T.RandAugment(num_ops=randaugment_num_ops, magnitude=randaugment_magnitude))
    train_ops += [T.ToTensor(), T.Normalize(_MEAN, _STD)]
    if cutout > 0.0:
        train_ops.append(T.RandomErasing(p=cutout))
    train_tf = T.Compose(train_ops)
    test_tf = T.Compose([T.ToTensor(), T.Normalize(_MEAN, _STD)])

    train_aug = torchvision.datasets.ImageFolder(str(train_dir), transform=train_tf)
    train_plain = torchvision.datasets.ImageFolder(str(train_dir), transform=test_tf)
    test = torchvision.datasets.ImageFolder(str(val_dir), transform=test_tf)

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
