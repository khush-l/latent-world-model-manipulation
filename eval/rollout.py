"""Latent rollout helpers for MPC planning over a trained LeWM.

The two primitives MPC needs:

    encode_pixels(model, pixels)               -> latents
    rollout(model, pixels_history, actions)    -> autoregressive latent trajectory

Mirrors references/le-wm-main/jepa.py:JEPA.rollout, adapted to our
training/model.py:JEPA interface (dict-batched encode + (emb, act_emb)-based
predict).
"""

import sys
from pathlib import Path

import torch
from einops import rearrange

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from model import JEPA  # noqa: E402


@torch.no_grad()
def encode_pixels(model: JEPA, pixels: torch.Tensor) -> torch.Tensor:
    """Encode a sequence of pixel frames into latents.

    Args:
        pixels: (B, T, C, H, W) float tensor, ImageNet-normalized.

    Returns:
        emb: (B, T, D) latent embeddings.
    """
    out = model.encode({"pixels": pixels})
    return out["emb"]


@torch.no_grad()
def rollout(
    model: JEPA,
    pixels_history: torch.Tensor,
    action_sequence: torch.Tensor,
    history_size: int,
) -> torch.Tensor:
    """Autoregressively roll out latents given pixel history + action plan.

    At each rollout step we feed the predictor the last `history_size` latents
    + the corresponding `history_size` action embeddings; the predictor's
    last-position output is the next-step latent prediction, which is then
    appended to the rolling latent buffer.

    Args:
        pixels_history: (B, S, H, C, H_img, W_img) — last H frames per CEM sample.
                        B = task batch, S = parallel CEM samples, H = history_size.
        action_sequence: (B, S, T, A) — full action plan to be rolled out.
                         T = H + n_future. The first H entries are the
                         history actions aligned to `pixels_history`. The
                         remaining T-H are the future actions whose effect
                         on the latents we want to predict.
        history_size: H — must equal the predictor's context length.

    Returns:
        emb: (B, S, T, D) — latents over the full T-step sequence.
             emb[:, :, :H] are the encoded history latents.
             emb[:, :, H:] are the predicted future latents.
    """
    B, S, H, C, H_img, W_img = pixels_history.shape
    _, _, T, A = action_sequence.shape
    assert H == history_size, (H, history_size)
    assert T >= H, (T, H)
    n_steps = T - H

    # Encode history pixels — flatten (B, S) into the batch dim for one big call.
    px = rearrange(pixels_history, "b s t c h w -> (b s) t c h w")
    init = model.encode({"pixels": px})
    emb = init["emb"]  # ((B*S), H, D)

    # Encode the entire action sequence in one shot; Embedder is per-position
    # so this is equivalent to encoding action windows one at a time.
    act = rearrange(action_sequence, "b s t a -> (b s) t a")
    act_emb_all = model.action_encoder(act)  # ((B*S), T, D)

    # Autoregressive rollout. At step t, the predictor's input window is the
    # last H latents + the corresponding H action embeddings; we take the
    # last position's output as the next latent.
    for t in range(n_steps):
        ctx_emb = emb[:, -H:]              # ((B*S), H, D)
        ctx_act = act_emb_all[:, t : t + H]  # ((B*S), H, D)
        pred = model.predict(ctx_emb, ctx_act)[:, -1:]  # ((B*S), 1, D)
        emb = torch.cat([emb, pred], dim=1)

    return rearrange(emb, "(b s) t d -> b s t d", b=B)


@torch.no_grad()
def goal_cost(predicted_emb: torch.Tensor, goal_emb: torch.Tensor) -> torch.Tensor:
    """Terminal latent distance cost used by CEM.

    Args:
        predicted_emb: (B, S, T, D) — rolled-out latent trajectories.
        goal_emb:      (B, D) — encoded goal observation, broadcast across S.

    Returns:
        cost: (B, S) — sum-of-squares distance between terminal predicted
              latent and goal latent.
    """
    terminal = predicted_emb[:, :, -1, :]            # (B, S, D)
    diff = terminal - goal_emb.unsqueeze(1)          # (B, S, D)
    return (diff * diff).sum(dim=-1)                 # (B, S)
