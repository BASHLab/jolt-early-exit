"""ESC-50: 50-class environmental sound classification.

Source: Piczak, "ESC: Dataset for Environmental Sound Classification" (ACMMM 2015).
2000 clips of 5 seconds each, 50 single-label classes, 5-fold cross-validation. The
dataset ships as a single Github archive (``ESC-50-master.zip``) containing the audio
files plus ``meta/esc50.csv`` with the per-clip metadata.

We use fold 5 as the held-out test split, fold 4 as the validation split, and folds 1-3
as the training split, which is the canonical "one-fold test, one-fold val, rest train"
protocol used in the early-exit audio literature when a single non-CV report is needed.

Per-clip preprocessing: load the WAV at the native 44.1 kHz, resample to 22.05 kHz, pad
or center-crop to exactly 5 seconds (110250 samples), compute a 128-band log-mel
spectrogram with n_fft=2048 and hop=512, yielding a feature tensor of shape
``(1, 128, 216)``. SpecAugment time- and frequency-masking is applied on the training
split only.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import FrequencyMasking, MelSpectrogram, Resample, TimeMasking

from ..config import make_generator, seed_worker

SAMPLE_RATE = 22050
N_SAMPLES = 5 * SAMPLE_RATE  # 5 seconds at 22.05 kHz = 110250 samples

# Canonical 5-fold split: fold 5 = test, fold 4 = val, folds 1-3 = train.
TEST_FOLD = 5
VAL_FOLD = 4
TRAIN_FOLDS = (1, 2, 3)


class _ESC50Dataset(Dataset):
    def __init__(
        self,
        root: str,
        folds: Tuple[int, ...],
        *,
        augment: bool = False,
        n_mels: int = 128,
        n_fft: int = 2048,
        win_length: int = 2048,
        hop_length: int = 512,
        specaug_freq_mask: int = 0,
        specaug_time_mask: int = 0,
    ):
        root = Path(root)
        meta = pd.read_csv(root / "meta" / "esc50.csv")
        meta = meta[meta["fold"].isin(folds)].reset_index(drop=True)
        self.entries: List[Tuple[Path, int]] = [
            (root / "audio" / r["filename"], int(r["target"])) for _, r in meta.iterrows()
        ]
        self.augment = augment
        self.mel = MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.freq_mask = FrequencyMasking(specaug_freq_mask) if specaug_freq_mask > 0 else None
        self.time_mask = TimeMasking(specaug_time_mask) if specaug_time_mask > 0 else None
        # ESC-50 native sample rate is 44.1 kHz; resample once at load time.
        self.resample = Resample(orig_freq=44100, new_freq=SAMPLE_RATE)

    def __len__(self) -> int:
        return len(self.entries)

    def _load(self, path: Path) -> torch.Tensor:
        wav, sr = torchaudio.load(str(path))
        if sr != 44100:
            wav = torchaudio.functional.resample(wav, sr, 44100)
        wav = self.resample(wav)
        wav = wav.mean(dim=0)  # mono
        # Pad / center-crop to N_SAMPLES.
        if wav.shape[0] < N_SAMPLES:
            wav = torch.nn.functional.pad(wav, (0, N_SAMPLES - wav.shape[0]))
        else:
            start = (wav.shape[0] - N_SAMPLES) // 2
            wav = wav[start : start + N_SAMPLES]
        return wav

    def __getitem__(self, idx: int):
        path, label = self.entries[idx]
        wav = self._load(path)
        spec = self.mel(wav.unsqueeze(0))  # (1, n_mels, time)
        spec = torch.log1p(spec)
        if self.augment:
            # Apply each SpecAugment mask twice (two-mask convention from
            # Park et al. 2019); a single application is too mild for ESC-50
            # at the single-fold protocol and 1200 train samples.
            if self.freq_mask is not None:
                spec = self.freq_mask(spec)
                spec = self.freq_mask(spec)
            if self.time_mask is not None:
                spec = self.time_mask(spec)
                spec = self.time_mask(spec)
        return spec, label


def esc50_dataloaders(
    *,
    root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    download: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # ``download`` flag is accepted for API compatibility but ESC-50 must be downloaded
    # by the caller (one-time zip from the GitHub release) since the file has no stable
    # public URL on Zenodo.
    base = Path(root) / "ESC-50-master"
    if not (base / "meta" / "esc50.csv").exists():
        raise FileNotFoundError(
            f"ESC-50 not found at {base}. Expected layout {base}/meta/esc50.csv and "
            f"{base}/audio/*.wav. Download from https://github.com/karoldvl/ESC-50 once."
        )
    train_ds = _ESC50Dataset(
        base, folds=TRAIN_FOLDS,
        augment=True, specaug_freq_mask=24, specaug_time_mask=48,
    )
    val_ds = _ESC50Dataset(base, folds=(VAL_FOLD,))
    test_ds = _ESC50Dataset(base, folds=(TEST_FOLD,))
    g = make_generator(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
    return train_loader, val_loader, test_loader
