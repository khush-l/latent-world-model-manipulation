# Training

LeWM-style action-conditioned JEPA on v3 HDF5 SoftGym trajectories.

- **Architecture** — see [`ARCHITECTURE.md`](./ARCHITECTURE.md)
- **Data format** — see [`../DATA_FORMAT.md`](../DATA_FORMAT.md)
- **Reference** — [`../references/le-wm-main/`](../references/le-wm-main/)

## Files

| File | Purpose |
|---|---|
| `model.py` | JEPA + ViT-Tiny encoder + ARPredictor + Embedder + SIGReg |
| `dataset.py` | `HDF5SequenceDataset` — sequence sampler with episode-aware windows + z-score normalization |
| `train.py` | Plain-PyTorch training loop, includes `--overfit-batches` smoke mode and per-step JSONL logging |
| `plot_metrics.py` | Render `metrics.jsonl` → `charts.png` (matplotlib) |
| `requirements.txt` | Pinned dependencies for the training-side venv |
| `runs/` | Per-run output dir (`metrics.jsonl`, `config.json`, `charts.png`). Gitignored. |

## Quickstart

Overfit verification on the 3-episode smoke set (drops pred_loss → ~5e-5 in ~10s on an L4):

```bash
python training/train.py \
  --data simulation/data/smoke_v3.h5 \
  --img-size 128 --patch-size 16 \
  --batch-size 4 --overfit-batches 4 \
  --steps 200 --log-every 10 \
  --no-sigreg --precision fp32
```

Full-config single-GPU run (matches `references/le-wm-main/config/train/lewm.yaml`):

```bash
python training/train.py \
  --data training/data/clothflatten_mixed_v3.h5 \
  --batch-size 128 --img-size 224 --patch-size 14 \
  --steps 100000 --lr 5e-5 \
  --run-name clothflatten_full
```

## Charts

`train.py` writes per-step metrics to `training/runs/<run_name>/metrics.jsonl`.
Render them with:

```bash
python training/plot_metrics.py training/runs/<run_name>/metrics.jsonl
# → training/runs/<run_name>/charts.png   (5 panels: loss, pred, sigreg, emb_std, emb_mean)
```

Useful flags:
- `--smooth N` — moving-average smoothing window
- `--log-y` — log-scale the loss panels
- `--output path.png` — custom output location

## Environment

Project-local venv at `.venv/` (gitignored). Setup from scratch:

```bash
python3 -m venv .venv
# Bootstrap pip if Debian's python3-venv shipped without ensurepip:
curl -sSf https://bootstrap.pypa.io/get-pip.py | .venv/bin/python -
.venv/bin/pip install -r training/requirements.txt
```

Use it directly without activating: `.venv/bin/python training/train.py ...`

**CUDA version**: `requirements.txt` pins `torch==2.5.1+cu121` from the official
CUDA-12.1 wheel index. For a different CUDA version (or CPU-only), edit the
`--index-url` line in `training/requirements.txt` — see
https://pytorch.org/get-started/locally/.
