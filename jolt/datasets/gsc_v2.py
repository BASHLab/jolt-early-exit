"""Google Speech Commands V2 (35 keywords) dataloaders for EE-audio.

Source: Warden, "Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition" (2018);
v2 = ``speech_commands_v0.02``. Splits use the official ``validation_list.txt`` and
``testing_list.txt`` (speaker-hash split): 84843 train / 9981 val / 11005 test. 35 classes
plus ``_background_noise_`` (excluded from the classifier task).

Each .wav is 1 second @ 16 kHz (16000 samples). The loader pads (or center-crops) every
clip to exactly 16000 samples, applies on-the-fly mel-spectrogram (1 ch × 40 mel × 101
time-frames at 10 ms hop), then SpecAugment (time / frequency masking) and random time
shift on the training set only.

This loader uses ``torchaudio.datasets.SPEECHCOMMANDS`` for raw .wav access (auto-downloads
on first call) and ``torchaudio.transforms`` for the mel-spec + SpecAugment pipeline so
the env stays self-contained (no librosa dependency).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import FrequencyMasking, MelSpectrogram, TimeMasking

from ..config import make_generator, seed_worker


# All 35 keyword classes in GSC v2 (sorted alphabetically; same order
# as the directory listing for stable label indices).
GSC_V2_LABELS: List[str] = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five",
    "follow", "forward", "four", "go", "happy", "house", "learn", "left",
    "marvin", "nine", "no", "off", "on", "one", "right", "seven",
    "sheila", "six", "stop", "three", "tree", "two", "up", "visual",
    "wow", "yes", "zero",
]
_LABEL_TO_IDX = {label: i for i, label in enumerate(GSC_V2_LABELS)}
SAMPLE_RATE = 16000
_NUM_SAMPLES = SAMPLE_RATE  # exactly 1 second


class _GSCv2Dataset(Dataset):
    """Wraps torchaudio SPEECHCOMMANDS with per-clip mel-spec + optional augmentations."""

    def __init__(
        self,
        root: str,
        subset: str,
        *,
        n_mels: int = 40,
        n_fft: int = 480,
        win_length: int = 480,
        hop_length: int = 160,
        time_shift_ms: float = 0.0,
        specaug_freq_mask: int = 0,
        specaug_time_mask: int = 0,
        specaug_n_freq: int = 0,
        specaug_n_time: int = 0,
    ):
        super().__init__()
        self.base = SPEECHCOMMANDS(root=root, subset=subset, download=False)
        self.time_shift_samples = int(time_shift_ms * SAMPLE_RATE / 1000.0)
        self.melspec = MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.freq_mask = (
            FrequencyMasking(freq_mask_param=specaug_freq_mask) if specaug_freq_mask > 0 else None
        )
        self.time_mask = (
            TimeMasking(time_mask_param=specaug_time_mask) if specaug_time_mask > 0 else None
        )
        self.specaug_n_freq = specaug_n_freq
        self.specaug_n_time = specaug_n_time

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        waveform, sr, label, _speaker_id, _utterance_number = self.base[index]
        assert sr == SAMPLE_RATE, f"unexpected sample rate {sr}"
        # waveform shape: (1, T). Pad or center-crop to exactly 16000 samples.
        wav = waveform[0]
        if wav.numel() < _NUM_SAMPLES:
            pad = _NUM_SAMPLES - wav.numel()
            wav = torch.nn.functional.pad(wav, (0, pad))
        elif wav.numel() > _NUM_SAMPLES:
            wav = wav[: _NUM_SAMPLES]
        # Train-only random time shift.
        if self.time_shift_samples > 0:
            shift = int(torch.randint(-self.time_shift_samples, self.time_shift_samples + 1, (1,)).item())
            wav = torch.roll(wav, shifts=shift)
            if shift > 0:
                wav[:shift] = 0.0
            elif shift < 0:
                wav[shift:] = 0.0
        # (1, T) -> mel-spec (1, n_mels, frames). Log-magnitude in dB-like scale.
        mel = self.melspec(wav.unsqueeze(0))
        mel = torch.log(mel.clamp_min(1e-10))
        # SpecAugment: random freq and time masks.
        if self.freq_mask is not None:
            for _ in range(self.specaug_n_freq):
                mel = self.freq_mask(mel)
        if self.time_mask is not None:
            for _ in range(self.specaug_n_time):
                mel = self.time_mask(mel)
        target = _LABEL_TO_IDX[label]
        return mel, target


def gsc_v2_dataloaders(
    *,
    root: str,
    batch_size: int = 100,
    num_workers: int = 4,
    val_size: Optional[int] = None,
    seed: int = 42,
    download: bool = True,
    n_mels: int = 40,
    n_fft: int = 480,
    win_length: int = 480,
    hop_length: int = 160,
    time_shift_ms: float = 100.0,
    specaug_freq_mask: int = 7,
    specaug_time_mask: int = 25,
    specaug_n_freq: int = 2,
    specaug_n_time: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build GSC v2 train / val / test dataloaders with the official speaker-hash split.

    ``val_size`` is ignored: the v2 official splits already define train / val / test, and
    they're consistent across the literature (the parameter is present for API parity with
    the CIFAR / Tiny-ImageNet loaders).
    """
    _ = val_size  # unused; v2 has an official validation split
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    if download:
        # Trigger the torchaudio download if needed (idempotent if data is present).
        SPEECHCOMMANDS(root=str(root_path), download=True)

    train = _GSCv2Dataset(
        root=str(root_path), subset="training",
        n_mels=n_mels, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
        time_shift_ms=time_shift_ms,
        specaug_freq_mask=specaug_freq_mask, specaug_time_mask=specaug_time_mask,
        specaug_n_freq=specaug_n_freq, specaug_n_time=specaug_n_time,
    )
    val = _GSCv2Dataset(
        root=str(root_path), subset="validation",
        n_mels=n_mels, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
        time_shift_ms=0.0,
        specaug_freq_mask=0, specaug_time_mask=0, specaug_n_freq=0, specaug_n_time=0,
    )
    test = _GSCv2Dataset(
        root=str(root_path), subset="testing",
        n_mels=n_mels, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
        time_shift_ms=0.0,
        specaug_freq_mask=0, specaug_time_mask=0, specaug_n_freq=0, specaug_n_time=0,
    )

    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        generator=make_generator(seed), worker_init_fn=seed_worker, drop_last=False,
    )
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
