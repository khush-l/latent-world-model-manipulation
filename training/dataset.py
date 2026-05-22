"""Dataset for v3-schema HDF5 trajectory files (see DATA_FORMAT.md §4).

Samples `num_steps` consecutive rows within a single episode. Image
preprocessing matches the LeWM reference (ImageNet z-score + resize),
and column normalizers (action / proprio / state) are computed once
from the dataset's finite rows and applied per sample.
"""

import json
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _column_stats(data: np.ndarray) -> tuple:
    """Per-column z-score from rows with no NaNs."""
    if data.ndim == 1:
        data = data[:, None]
    finite_mask = np.isfinite(data).all(axis=tuple(range(1, data.ndim)))
    valid = data[finite_mask]
    if valid.size == 0:
        return (np.zeros(data.shape[1:], dtype=np.float32),
                np.ones(data.shape[1:], dtype=np.float32))
    mean = valid.mean(axis=0).astype(np.float32)
    std = valid.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def _episode_windows(episode_idx: np.ndarray, num_steps: int) -> List[tuple]:
    """List of (start_row, length) windows of `num_steps` rows that stay
    within a single episode."""
    windows = []
    n = len(episode_idx)
    i = 0
    while i < n:
        j = i
        while j < n and episode_idx[j] == episode_idx[i]:
            j += 1
        ep_len = j - i
        if ep_len >= num_steps:
            for start in range(i, j - num_steps + 1):
                windows.append((start, num_steps))
        i = j
    return windows


class HDF5SequenceDataset(Dataset):
    """Samples `num_steps` consecutive rows from a v3 HDF5 file.

    Returns a dict per sample with:
        pixels   (T, C, H, W) float32 — ImageNet-normalized, resized to img_size
        action   (T, A)       float32 — z-score normalized; NaN replaced with 0
        proprio  (T, P)       float32 — z-score normalized
        state    (T, S)       float32 — z-score normalized
        episode_idx (T,)      int64
        step_idx    (T,)      int64
    """

    REQUIRED_KEYS = ("pixels", "action", "proprio", "state", "episode_idx", "step_idx")

    def __init__(self, h5_path, num_steps: int = 4, img_size: int = 224,
                 frameskip: int = 1, normalize_pixels: bool = True,
                 cache_stats: Optional[dict] = None):
        super().__init__()
        self.h5_path = str(Path(h5_path).resolve())
        self.num_steps = num_steps
        self.img_size = img_size
        self.frameskip = frameskip
        self.normalize_pixels = normalize_pixels
        # Per-worker handles, opened lazily in __getitem__.
        self._h5 = None

        with h5py.File(self.h5_path, "r") as f:
            for k in self.REQUIRED_KEYS:
                if k not in f:
                    raise KeyError(f"{self.h5_path} missing required key /{k}")
            self.n_rows = int(f["pixels"].shape[0])
            self.h, self.w = int(f["pixels"].shape[1]), int(f["pixels"].shape[2])
            self.action_dim = int(f["action"].shape[1])
            self.proprio_dim = int(f["proprio"].shape[1])
            self.state_dim = int(f["state"].shape[1])
            ep_idx = f["episode_idx"][:]

            if cache_stats is not None:
                self.stats = cache_stats
            else:
                self.stats = {
                    "action": _column_stats(f["action"][:]),
                    "proprio": _column_stats(f["proprio"][:]),
                    "state": _column_stats(f["state"][:]),
                }

        self.windows = _episode_windows(ep_idx, num_steps * frameskip)
        if not self.windows:
            raise ValueError(
                f"no episodes long enough for num_steps*frameskip={num_steps * frameskip} "
                f"in {self.h5_path}"
            )

    def __len__(self):
        return len(self.windows)

    def _h(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, idx):
        start, length = self.windows[idx]
        rows = list(range(start, start + length, self.frameskip))
        rows = rows[: self.num_steps]
        f = self._h()

        pixels = f["pixels"][rows]  # (T, H, W, 3) uint8
        pixels = pixels.astype(np.float32) / 255.0
        pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD if self.normalize_pixels else pixels
        pixels = np.transpose(pixels, (0, 3, 1, 2))  # (T, C, H, W)
        pixels = torch.from_numpy(np.ascontiguousarray(pixels))
        if pixels.shape[-1] != self.img_size or pixels.shape[-2] != self.img_size:
            pixels = F.interpolate(
                pixels, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
            )

        def _norm(name, raw):
            mean, std = self.stats[name]
            arr = np.nan_to_num(np.asarray(raw, dtype=np.float32), nan=0.0)
            return (arr - mean) / std

        sample = {
            "pixels": pixels,
            "action": torch.from_numpy(_norm("action", f["action"][rows])),
            "proprio": torch.from_numpy(_norm("proprio", f["proprio"][rows])),
            "state": torch.from_numpy(_norm("state", f["state"][rows])),
            "episode_idx": torch.from_numpy(np.asarray(f["episode_idx"][rows], dtype=np.int64)),
            "step_idx": torch.from_numpy(np.asarray(f["step_idx"][rows], dtype=np.int64)),
        }
        return sample
