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


# ---------------------------------------------------------------------------
# GPU-resident dataset — production-style: full table on device, no per-step
# IO or CPU↔GPU copy.
# ---------------------------------------------------------------------------

class GPUDataset:
    """Loads the entire v3 HDF5 onto the GPU once. Per-step access is a single
    indexed gather → on-device normalize/transpose. Zero CPU work, zero
    per-step host↔device transfer.

    Sized for datasets that fit in GPU RAM:
      - 5000-episode rope file (uint8): ~19 GB pixels (+ small columns)
      - H100 80 GB has plenty of room for that + 18M-param model + activations
    """

    REQUIRED_KEYS = ("pixels", "action", "proprio", "state", "episode_idx", "step_idx")

    def __init__(self, h5_path, num_steps: int = 4, img_size: int = 224,
                 frameskip: int = 1, device: str = "cuda"):
        self.h5_path = str(Path(h5_path).resolve())
        self.num_steps = int(num_steps)
        self.img_size = int(img_size)
        self.frameskip = int(frameskip)
        self.device = device

        print(f"[GPUDataset] loading {self.h5_path} → {device}...", flush=True)
        with h5py.File(self.h5_path, "r") as f:
            for k in self.REQUIRED_KEYS:
                if k not in f:
                    raise KeyError(f"{self.h5_path} missing /{k}")
            self.n_rows = int(f["pixels"].shape[0])
            self.h_img, self.w_img = int(f["pixels"].shape[1]), int(f["pixels"].shape[2])
            self.action_dim = int(f["action"].shape[1])
            self.proprio_dim = int(f["proprio"].shape[1])
            self.state_dim = int(f["state"].shape[1])

            # Compute stats on CPU before uploading.
            self.stats = {
                "action":  _column_stats(f["action"][:]),
                "proprio": _column_stats(f["proprio"][:]),
                "state":   _column_stats(f["state"][:]),
            }

            # Pixels stay as uint8 on GPU — float conversion + normalize happen
            # at sample time. That's 4× smaller upload + lets us afford the
            # full dataset on a single H100.
            self._pixels  = torch.from_numpy(f["pixels"][:]).to(device)              # (N, H, W, 3) uint8
            self._action  = torch.from_numpy(f["action"][:].astype(np.float32)).to(device)
            self._proprio = torch.from_numpy(f["proprio"][:].astype(np.float32)).to(device)
            self._state   = torch.from_numpy(f["state"][:].astype(np.float32)).to(device)
            self._epidx   = torch.from_numpy(f["episode_idx"][:].astype(np.int64)).to(device)
            self._stepidx = torch.from_numpy(f["step_idx"][:].astype(np.int64)).to(device)
            ep_idx_cpu    = f["episode_idx"][:]

        # Window starts as int64 tensor for fast GPU gather.
        windows = _episode_windows(ep_idx_cpu, num_steps * frameskip)
        if not windows:
            raise ValueError("no episodes long enough for num_steps*frameskip")
        starts = np.asarray([w[0] for w in windows], dtype=np.int64)
        self._window_starts = torch.from_numpy(starts).to(device)
        self.n_windows = int(starts.shape[0])

        # Per-step row offsets within a window (e.g. [0, 1, 2, 3] when frameskip=1).
        self._row_offsets = torch.arange(
            0, self.num_steps * self.frameskip, self.frameskip,
            device=device, dtype=torch.int64,
        )[: self.num_steps]

        # Pre-upload normalization tensors so we don't recompute every batch.
        self._imagenet_mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 1, 3, 1, 1)
        self._imagenet_std  = torch.tensor(IMAGENET_STD,  device=device).view(1, 1, 3, 1, 1)
        self._action_mean   = torch.from_numpy(self.stats["action"][0]).to(device)
        self._action_std    = torch.from_numpy(self.stats["action"][1]).to(device)
        self._proprio_mean  = torch.from_numpy(self.stats["proprio"][0]).to(device)
        self._proprio_std   = torch.from_numpy(self.stats["proprio"][1]).to(device)
        self._state_mean    = torch.from_numpy(self.stats["state"][0]).to(device)
        self._state_std     = torch.from_numpy(self.stats["state"][1]).to(device)

        pixel_gb = self._pixels.numel() / 1e9
        total_gb = (self._pixels.numel()
                    + self._action.numel() * 4 + self._proprio.numel() * 4
                    + self._state.numel() * 4) / 1e9
        print(f"[GPUDataset] uploaded {pixel_gb:.2f} GB pixels (+ {total_gb-pixel_gb:.2f} GB other) "
              f"= {total_gb:.2f} GB total")
        print(f"[GPUDataset] n_windows={self.n_windows}, n_rows={self.n_rows}")

    def __len__(self):
        return self.n_windows

    def get_batch(self, batch_indices: torch.Tensor) -> dict:
        """Gather a batch of windows by index.

        Args:
            batch_indices: (B,) int64 tensor of window indices, on `self.device`.
        Returns:
            dict of (B, T, ...) tensors, all on `self.device`.
        """
        # rows: (B, T) global row indices into the flat tables
        starts = self._window_starts[batch_indices]              # (B,)
        rows = starts.unsqueeze(1) + self._row_offsets.unsqueeze(0)  # (B, T)
        flat = rows.reshape(-1)                                  # (B*T,)

        pixels = self._pixels[flat]                              # (B*T, H, W, 3) uint8
        # Normalize + reshape on GPU.
        pixels = pixels.to(torch.float32) / 255.0
        pixels = pixels.permute(0, 3, 1, 2).contiguous()         # (B*T, 3, H, W)
        pixels = pixels.unflatten(0, (-1, self.num_steps))       # (B, T, 3, H, W)
        pixels = (pixels - self._imagenet_mean) / self._imagenet_std
        if pixels.shape[-1] != self.img_size or pixels.shape[-2] != self.img_size:
            # GPU bilinear resize. Fold (B, T) into batch dim for interpolate.
            B, T = pixels.shape[:2]
            pixels = pixels.reshape(B * T, 3, *pixels.shape[-2:])
            pixels = F.interpolate(pixels, size=(self.img_size, self.img_size),
                                   mode="bilinear", align_corners=False)
            pixels = pixels.reshape(B, T, 3, self.img_size, self.img_size)

        def _norm(raw, mean, std):
            raw = torch.nan_to_num(raw, nan=0.0)
            return (raw - mean) / std

        action  = _norm(self._action[flat].unflatten(0, (-1, self.num_steps)),
                        self._action_mean,  self._action_std)
        proprio = _norm(self._proprio[flat].unflatten(0, (-1, self.num_steps)),
                        self._proprio_mean, self._proprio_std)
        state   = _norm(self._state[flat].unflatten(0, (-1, self.num_steps)),
                        self._state_mean,   self._state_std)

        return {
            "pixels":      pixels,
            "action":      action,
            "proprio":     proprio,
            "state":       state,
            "episode_idx": self._epidx[flat].unflatten(0, (-1, self.num_steps)),
            "step_idx":    self._stepidx[flat].unflatten(0, (-1, self.num_steps)),
        }


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
                 cache_stats: Optional[dict] = None, cache_pixels: bool = False):
        super().__init__()
        self.h5_path = str(Path(h5_path).resolve())
        self.num_steps = num_steps
        self.img_size = img_size
        self.frameskip = frameskip
        self.normalize_pixels = normalize_pixels
        # Per-worker handles, opened lazily in __getitem__.
        self._h5 = None
        # Optional in-memory pixel cache (kills per-step gzip decompression).
        self.cache_pixels = bool(cache_pixels)
        self._pixels_cache = None
        self._action_cache = None
        self._proprio_cache = None
        self._state_cache = None
        self._epidx_cache = None
        self._stepidx_cache = None

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

            if self.cache_pixels:
                # One-shot read of every column into RAM. Each worker shares
                # the parent's numpy arrays via copy-on-write fork semantics,
                # so we pay this memory cost once, not per worker.
                print(f"  [HDF5SequenceDataset] caching dataset in RAM "
                      f"(pixels={f['pixels'].nbytes/1e9:.2f} GB)...", flush=True)
                self._pixels_cache  = f["pixels"][:]
                self._action_cache  = f["action"][:].astype(np.float32)
                self._proprio_cache = f["proprio"][:].astype(np.float32)
                self._state_cache   = f["state"][:].astype(np.float32)
                self._epidx_cache   = f["episode_idx"][:].astype(np.int64)
                self._stepidx_cache = f["step_idx"][:].astype(np.int64)

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
        rows_arr = np.asarray(rows)

        # Read from in-memory cache when available, else lazy h5py read.
        if self._pixels_cache is not None:
            raw_pixels  = self._pixels_cache[rows_arr]
            raw_action  = self._action_cache[rows_arr]
            raw_proprio = self._proprio_cache[rows_arr]
            raw_state   = self._state_cache[rows_arr]
            raw_epidx   = self._epidx_cache[rows_arr]
            raw_stepidx = self._stepidx_cache[rows_arr]
        else:
            f = self._h()
            raw_pixels  = f["pixels"][rows]
            raw_action  = f["action"][rows]
            raw_proprio = f["proprio"][rows]
            raw_state   = f["state"][rows]
            raw_epidx   = f["episode_idx"][rows]
            raw_stepidx = f["step_idx"][rows]

        pixels = raw_pixels.astype(np.float32) / 255.0
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
            "action":  torch.from_numpy(_norm("action",  raw_action)),
            "proprio": torch.from_numpy(_norm("proprio", raw_proprio)),
            "state":   torch.from_numpy(_norm("state",   raw_state)),
            "episode_idx": torch.from_numpy(np.asarray(raw_epidx, dtype=np.int64)),
            "step_idx":    torch.from_numpy(np.asarray(raw_stepidx, dtype=np.int64)),
        }
        return sample
