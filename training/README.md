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

Reported SoftGym single-GPU run (native 128 x 128 observations, 16 x 16 patches):

```bash
python training/train.py \
  --data training/data/clothflatten_mixed_v3.h5 \
  --batch-size 128 --img-size 128 --patch-size 16 \
  --steps 100000 --lr 5e-5 \
  --run-name clothflatten_full
```

## Ablation sweep

`run_ablations.sh` runs 16 **hyperparameter** configurations sequentially on one
GPU, ALL on the full mixed dataset, **~8 epochs each** (per-run step counts are
auto-computed from batch size and window length). Each logs to W&B under the
shared tag `ablation_sweep_v1`. Runs are priority-ordered (most informative
first) so an interrupted sweep still yields the important ablations.

| # | Run | Hyperparameter tested |
|---|---|---|
| 1 | baseline | reference config |
| 2 | no_sigreg | collapse without SIGReg |
| 3 | sigreg_low (λ=0.01) | weak regularization |
| 4 | sigreg_high (λ=0.5) | over-regularization (paper Fig 16) |
| 5 | dropout_off (p=0.0) | no predictor dropout (paper Tab 9) |
| 6 | predictor_tiny (dim_head=12) | ViT-T predictor (paper Tab 6: −15 SR) |
| 7 | predictor_shallow (depth=3) | half-depth predictor |
| 8 | lr_high (2e-4) | 4× learning rate |
| 9 | lr_low (1e-5) | 0.2× learning rate |
| 10 | lr_cosine | cosine schedule + 2k warmup |
| 11 | weight_decay_off (0.0) | no weight decay |
| 12 | weight_decay_high (1e-2) | 10× weight decay |
| 13 | history_5 | longer prediction context |
| 14 | num_preds_2 | predict 2 steps ahead |
| 15 | batch_256 | larger batch |
| 16 | batch_64 | smaller batch (2× steps — runs last) |

```bash
export WANDB_API_KEY=<key>        # or `wandb login`
bash training/run_ablations.sh    # ~8 epochs/run; ~8 h total on H100 @ ~34 steps/s

# Override knobs via env vars:
EPOCHS=6 bash training/run_ablations.sh        # fewer epochs (faster, ~6.5 h)
START_AT=9 bash training/run_ablations.sh      # resume from run #9
```

Robust to single-run failure (no `set -e`); per-run logs in
`training/runs/abl_*/train.log`, summary in
`training/runs/ablation_sweep_v1.summary`. All curves overlay in W&B by
filtering the `ablation_sweep_v1` tag.

## Charts

`train.py` writes per-step metrics to `training/runs/<run_name>/metrics.jsonl`.
Render them with:

```bash
python training/plot_metrics.py training/runs/<run_name>/metrics.jsonl
# → training/runs/<run_name>/charts.png   (5 panels: loss, pred, sigreg, emb_std, emb_mean)
```

Useful flags:
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
