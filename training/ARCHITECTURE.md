# Training Architecture

LeWM-style action-conditioned JEPA on SoftGym deformable-object trajectories.
Architecture inspired by LeWM with two simplifications:

1. Custom **ViT-Tiny** encoder (no `transformers` / `timm` dependency).
2. Plain PyTorch training loop in `train.py` (no Hydra / Lightning / `stable_pretraining`).

The model **inputs**, **forward path**, **loss**, and **defaults** match the
reference. See `DATA_FORMAT.md` for the dataset side.

---

## 1. Forward path

```
batch["pixels"]  : (B, T=4, 3, H, W)   ImageNet-normalized
batch["action"]  : (B, T=4, A=8)       z-scored; NaN→0 at episode terminal
batch["proprio"] : (B, T=4, P=8)       z-scored  (not consumed by forward)
batch["state"]   : (B, T=4, S=15)      z-scored  (not consumed by forward)

   ┌──────────────────────────────────────────────────────────────┐
   │   ViT-Tiny encoder (shared, applied per-frame)              │
   │   (B*T, 3, H, W) → (B*T, D=192)  CLS token                  │
   └──────────────────────────────────────────────────────────────┘
                                │
                            projector (MLP 192→2048→192)
                                │
                    emb : (B, T, D=192)
                                │
                                ▼
   ┌─────────────────────────────────────────┐
   │  Action encoder (Embedder)              │
   │  Conv1d(1×1) + 2-layer MLP              │
   │  (B, T, 8) → (B, T, D=192)              │
   └─────────────────────────────────────────┘
                                │
                    act_emb : (B, T, D=192)
                                │
                                ▼
   ┌─────────────────────────────────────────────────────────────┐
   │  ARPredictor: 6× ConditionalBlock                            │
   │   - causal self-attn over T tokens                           │
   │   - AdaLN-zero conditioning on act_emb (shift/scale/gate)    │
   │  takes ctx_emb=emb[:, :history_size=3], ctx_act=act_emb[:, :3]│
   │  outputs pred_emb : (B, history_size, D)                     │
   └─────────────────────────────────────────────────────────────┘
                                │
                            pred_proj (MLP 192→2048→192)
                                │
                            pred_emb
```

Target: `tgt_emb = emb[:, num_preds=1 :]` — the embedding of the **next** frame
for each context position. With `history_size=3`, `num_preds=1`, `T=4`, the
predictor is trained to predict steps `1..3` from steps `0..2`.

---

## 2. Loss

```
pred_loss   = MSE(pred_emb, tgt_emb.detach()-free)       # all positions
sigreg_loss = SIGReg(emb.transpose(0, 1))                # epps-pulley statistic
loss        = pred_loss + λ * sigreg_loss                # λ = 0.09 (lewm.yaml)
```

**SIGReg** (Sketch Isotropic Gaussian Regularizer): pushes the embedding
distribution toward isotropic Gaussian by computing an Epps-Pulley statistic
on random 1D projections (`knots=17`, `num_proj=1024`). Prevents the trivial
collapse where the encoder maps everything to a constant.

NaN handling: `batch["action"]` may contain NaN at episode-terminal rows (see
`DATA_FORMAT.md` §1). `lewm_forward` zeros them out before encoding, matching
`references/le-wm-main/train.py:25`.

---

## 3. Components

| Module | Spec | Approx params |
|---|---|---|
| `ViTTiny` | img=224, patch=14, depth=12, heads=3, D=192, MLP=768 | ~5.7M |
| `Embedder` (action) | Conv1d(8→8, k=1) + Linear(8→768) + SiLU + Linear(768→192) | ~0.15M |
| `MLP` projector | Linear(192→2048) + LN + GELU + Linear(2048→192) | ~0.79M |
| `MLP` pred_proj | same shape | ~0.79M |
| `ARPredictor` | 6× ConditionalBlock, dim=192, heads=16, dim_head=12, MLP=2048, dropout=0.1 | ~3.3M |
| **Total** | | **~11M** |

`MLP` defaults to `LayerNorm` instead of the reference's `BatchNorm1d` —
keeps the projector well-behaved with `batch_size=1` overfit tests. Switch to
BN for full-scale runs with `use_bn_proj=True` if you want exact parity.

---

## 4. Configuration knobs

Match `references/le-wm-main/config/train/lewm.yaml` defaults.

| Flag | Default | Source |
|---|---|---|
| `--img-size` | 224 | `lewm.yaml:13` |
| `--patch-size` | 14 | `model/lewm.yaml:6` |
| `--embed-dim` | 192 | `lewm.yaml:38` |
| `--history-size` | 3 | `lewm.yaml:36` |
| `--num-preds` | 1 | `lewm.yaml:37` |
| `--frameskip` | 1 | `DATA_FORMAT.md §2` (env `action_repeat=8` already aggregates) |
| `--predictor-depth` | 6 | `model/lewm.yaml:17` |
| `--predictor-heads` | 16 | `model/lewm.yaml:18` |
| `--predictor-dropout` | 0.1 | `model/lewm.yaml:21` |
| `--lr` / `--weight-decay` | 5e-5 / 1e-3 | `lewm.yaml:31–32` |
| `--grad-clip` | 1.0 | `lewm.yaml:20` |
| `--precision` | bf16 | `lewm.yaml:19` |
| `--sigreg-weight` | 0.09 | `lewm.yaml:42` |

---

## 5. Smoke-test workflow

The `--overfit-batches N` flag restricts the dataset to `N * batch_size`
fixed windows. The model should drive `pred_loss → 0` (with SIGReg keeping
`emb_std` above ~0.5). If it can't, the architecture or dataloader is broken
before any data-scale concerns matter.

```bash
python training/train.py \
  --data simulation/data/smoke_v3.h5 \
  --img-size 128 --patch-size 16 \
  --batch-size 4 --overfit-batches 4 \
  --steps 200 --log-every 10 \
  --no-sigreg --precision fp32
```

Why those overrides:
- **`--img-size 128 --patch-size 16`** — train at the native collection
  resolution; avoids a resize and shrinks the patch grid from 16×16 to 8×8.
- **`--no-sigreg`** — for overfit you *want* embeddings to collapse onto each
  sample; SIGReg fights this. Drop it for smoke; re-enable for real training.
- **`--precision fp32`** — bf16 noise can mask whether overfit is "really"
  happening on a tiny dataset.

---

## 6. TODOs

- **MPC rollout / planning** — the reference's `JEPA.rollout` and
  `get_cost` methods aren't ported. Needed for evaluation on SoftGym tasks.
- **Decoder for latent visualization** — not in the reference either; the
  DESIGN doc calls for one as a probe head.
- **Checkpointing / WandB logging** — print-based logging only. Easy to add
  once a real run is justified.