"""Build canonical-protocol PAMAP2 NPYs from raw .dat files.

Recipe (Hammerla, Halloran & Plotz IJCAI 2016 — the protocol most subsequent
DeepConvLSTM/PAMAP2 results report against):
  - 27 channels per sample = 3 IMUs (hand, chest, ankle) x 9 channels each
    (3-axis accelerometer +/-16g, 3-axis gyroscope, 3-axis magnetometer).
  - Downsample 100 Hz IMU -> 33.3 Hz by taking every 3rd row.
  - 5.12 s windows = 171 samples at 33.3 Hz, step = 1 s = 33 samples (78%
    overlap).
  - Drop activity-0 (transient) windows; assign each kept window the activity
    of its last sample (matches O&R / Hammerla last-sample-label convention).
  - Linear-interpolate NaNs per channel before windowing (PAMAP2 has wireless
    dropouts and the HR column is mostly NaN by design; we don't keep HR).
  - 12 protocol activities (IDs 1,2,3,4,5,6,7,12,13,16,17,24).

Output layout matches the existing jolt/datasets/pamap2.py loader, with three
per-sensor NPYs of shape (n_windows, 171, 9) plus label.npy of dtype str.

PAMAP2 raw .dat column layout (1-indexed, per the dataset readme):
  1: timestamp (s)
  2: activity ID
  3: heart rate (bpm)
  4-20:  hand IMU  (17 cols: temp, acc16 xyz, acc6 xyz, gyro xyz, mag xyz, orient xyzw)
  21-37: chest IMU (17 cols)
  38-54: ankle IMU (17 cols)
Within each IMU: cols offsets +0 temp, +1..+3 acc16, +4..+6 acc6, +7..+9 gyro,
+10..+12 mag, +13..+16 orient.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data/PAMAP2/raw/PAMAP2_Dataset/Protocol"
OUT_DIR = Path("data/PAMAP2/canonical27")

# 1-indexed raw column to 0-indexed numpy column when we load with `header=None`.
# Hand IMU starts at col 4 -> 0-indexed col 3.
HAND_BASE = 3
CHEST_BASE = HAND_BASE + 17  # 20
ANKLE_BASE = CHEST_BASE + 17  # 37

# Within an IMU block: 9 canonical channels = acc16 (1..3) + gyro (7..9) + mag (10..12)
IMU_OFFSETS = [1, 2, 3, 7, 8, 9, 10, 11, 12]

PROTOCOL_ACTIVITIES = {"1", "2", "3", "4", "5", "6", "7", "12", "13", "16", "17", "24"}

FS_RAW = 100  # Hz
DECIMATE = 3
FS = FS_RAW // DECIMATE  # 33 Hz (target 33.3)
WINDOW = 171  # samples = 5.12 s at 33.3 Hz
STEP = 33    # samples = 1 s = 78% overlap


def _sensor_columns(base: int) -> list[int]:
    return [base + o for o in IMU_OFFSETS]


def _load_subject(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, sep=r"\s+", header=None, na_values=["NaN"], engine="python")
    arr = df.to_numpy()
    activity = arr[:, 1].astype(int)
    sensors = {
        "hand": arr[:, _sensor_columns(HAND_BASE)].astype(np.float32),
        "chest": arr[:, _sensor_columns(CHEST_BASE)].astype(np.float32),
        "ankle": arr[:, _sensor_columns(ANKLE_BASE)].astype(np.float32),
    }
    return activity, sensors


def _interpolate_nans(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    n, c = out.shape
    for j in range(c):
        col = out[:, j]
        mask = ~np.isfinite(col)
        if not mask.any():
            continue
        idx = np.arange(n)
        good = ~mask
        if not good.any():
            out[:, j] = 0.0
            continue
        out[mask, j] = np.interp(idx[mask], idx[good], col[good])
    return out


def _decimate(activity: np.ndarray, sensors: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    return activity[::DECIMATE], {k: v[::DECIMATE] for k, v in sensors.items()}


def _window(activity: np.ndarray, sensors: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    n = activity.shape[0]
    if n < WINDOW:
        return {k: np.empty((0, WINDOW, 9), dtype=np.float32) for k in sensors}, np.empty((0,), dtype=object)

    starts = np.arange(0, n - WINDOW + 1, STEP)
    per_sensor = {k: np.empty((len(starts), WINDOW, 9), dtype=np.float32) for k in sensors}
    labels = np.empty((len(starts),), dtype=object)
    keep = []
    for i, s in enumerate(starts):
        last = activity[s + WINDOW - 1]
        # Take the last-sample label; if the last sample is outside protocol set
        # (id 0 or any optional-activity id), drop this window.
        lbl = str(int(last))
        if lbl not in PROTOCOL_ACTIVITIES:
            continue
        labels[i] = lbl
        for k, v in sensors.items():
            per_sensor[k][i] = v[s:s + WINDOW]
        keep.append(i)
    keep_idx = np.array(keep, dtype=int)
    if keep_idx.size == 0:
        return {k: np.empty((0, WINDOW, 9), dtype=np.float32) for k in sensors}, np.empty((0,), dtype=object)
    return {k: v[keep_idx] for k, v in per_sensor.items()}, labels[keep_idx]


def build_subject(subject_id: int) -> None:
    raw = RAW_DIR / f"subject10{subject_id}.dat"
    if not raw.exists():
        print(f"  [skip] {raw} does not exist")
        return
    out = OUT_DIR / f"subject10{subject_id}.dat"
    out.mkdir(parents=True, exist_ok=True)
    activity, sensors = _load_subject(raw)
    sensors = {k: _interpolate_nans(v) for k, v in sensors.items()}
    activity, sensors = _decimate(activity, sensors)
    per_sensor, labels = _window(activity, sensors)
    for k, v in per_sensor.items():
        np.save(out / f"{k}.npy", v)
    np.save(out / "label.npy", labels.astype(str))
    classes, counts = np.unique(labels, return_counts=True)
    print(f"  subject10{subject_id}: {labels.shape[0]} windows -> {dict(zip(classes.tolist(), counts.tolist()))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUT_DIR}")
    for s in args.subjects:
        build_subject(s)


if __name__ == "__main__":
    main()
