"""
LeWM-style JEPA world model.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Building blocks (ported from references/le-wm-main/module.py)
# ---------------------------------------------------------------------------

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal: bool = False):
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class Block(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, causal=False):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.causal = causal

    def forward(self, x):
        x = x + self.attn(x, causal=self.causal)
        x = x + self.mlp(x)
        return x


class ConditionalBlock(nn.Module):
    """AdaLN-zero block — conditions on per-token vector `c`."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, causal=True):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        self.causal = causal

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), causal=self.causal)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# ---------------------------------------------------------------------------
# Vision encoder — minimal ViT-Tiny
# ---------------------------------------------------------------------------

class ViTTiny(nn.Module):
    """Hand-rolled ViT-Tiny. Mirrors HF defaults (12 layers, 3 heads, D=192).
    Returns CLS-token embedding of shape (B, D).
    """

    def __init__(self, img_size=224, patch_size=14, in_channels=3, depth=12,
                 heads=3, dim_head=64, mlp_dim=768, dropout=0.0):
        super().__init__()
        assert img_size % patch_size == 0, (img_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.dim = heads * dim_head  # 192 for ViT-Tiny

        self.patch_embed = nn.Conv2d(in_channels, self.dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, self.dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            Block(self.dim, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim,
                  dropout=dropout, causal=False)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(self.dim)

    def forward(self, x):
        # x: (B, C, H, W)
        B = x.size(0)
        x = self.patch_embed(x)  # (B, D, h, w)
        x = rearrange(x, "b d h w -> b (h w) d")
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1+N, D)
        x = x + self.pos_embed[:, : x.size(1)]
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]  # CLS token, (B, D)


# ---------------------------------------------------------------------------
# Action encoder, projector, predictor (ported from reference module.py)
# ---------------------------------------------------------------------------

class Embedder(nn.Module):
    """Action embedder: per-step linear projection + 2-layer MLP."""

    def __init__(self, input_dim, emb_dim, mlp_scale=4, smoothed_dim=None):
        super().__init__()
        smoothed_dim = smoothed_dim or input_dim
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        # x: (B, T, D_in). Do NOT upcast to fp32 here — that breaks the
        # autocast bf16 chain and forces a fp32 matmul on H100 tensor cores.
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        return self.embed(x)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, use_bn=False):
        super().__init__()
        norm = nn.BatchNorm1d(hidden_dim) if use_bn else nn.LayerNorm(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm,
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor with causal attention + AdaLN action conditioning."""

    def __init__(self, num_frames, dim, depth=6, heads=16, dim_head=12,
                 mlp_dim=2048, dropout=0.1, emb_dropout=0.0):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim) * 0.02)
        self.dropout = nn.Dropout(emb_dropout)
        self.blocks = nn.ModuleList([
            ConditionalBlock(dim, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim,
                             dropout=dropout, causal=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, c):
        # x, c: (B, T, dim)
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x, c)
        return self.norm(x)


# ---------------------------------------------------------------------------
# SIGReg loss (Sketch Isotropic Gaussian Regularizer)
# ---------------------------------------------------------------------------

class SIGReg(nn.Module):
    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        # proj: (T, B, D)
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


# ---------------------------------------------------------------------------
# JEPA — top-level model
# ---------------------------------------------------------------------------

class JEPA(nn.Module):
    def __init__(self, encoder, predictor, action_encoder, projector=None, pred_proj=None,
                 proprio_encoder=None, state_head=None):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        # Optional auxiliary head: predicts the low-dim ground-truth rope/picker
        # state from `emb`, used ONLY during training to force the pixel latent to
        # retain rope geometry (decode grids showed `emb` loses coiled-shape
        # detail). Not used at plan time.
        self.state_head = state_head
        # Optional proprio encoder. When present, proprioception (picker xyz +
        # grip) is fused into the predictor's *conditioning* — NOT into `emb`
        # (the goal-matched, SIGReg-regularized pixel latent). This gives the
        # dynamics model explicit picker grounding while keeping goal-matching
        # about rope shape only. See PROPRIO_FUSION.md.
        self.proprio_encoder = proprio_encoder
        self.projector = projector if projector is not None else nn.Identity()
        self.pred_proj = pred_proj if pred_proj is not None else nn.Identity()

    def encode(self, batch):
        """batch: dict with 'pixels' (B, T, C, H, W), 'action' (B, T, A), and
        optionally 'proprio' (B, T, P)."""
        pixels = batch["pixels"].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t c h w -> (b t) c h w")
        cls = self.encoder(pixels)  # (B*T, D)
        emb = self.projector(cls)
        batch["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)
        if "action" in batch:
            cond = self.action_encoder(batch["action"])
            if self.proprio_encoder is not None and "proprio" in batch:
                # Additive fusion: both embeddings live in D-dim space.
                cond = cond + self.proprio_encoder(batch["proprio"])
            batch["act_emb"] = cond  # "act_emb" now means predictor conditioning
        return batch

    def predict(self, emb, cond):
        """cond = predictor conditioning (action, optionally + proprio)."""
        preds = self.predictor(emb, cond)
        b = emb.size(0)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        return rearrange(preds, "(b t) d -> b t d", b=b)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_lewm(
    img_size: int = 128,
    patch_size: int = 16,
    embed_dim: int = 192,
    encoder_depth: int = 12,
    encoder_heads: int = 3,
    predictor_depth: int = 6,
    predictor_heads: int = 16,
    predictor_dim_head: int = 64,  # heads=16 × dim_head=64 → 1024 inner-dim (ViT-S predictor, per LeWM Tab 6)
    predictor_mlp_dim: int = 2048,
    predictor_dropout: float = 0.1,
    history_size: int = 3,
    num_preds: int = 1,
    action_dim: int = 8,
    proprio_dim: int = 8,
    use_proprio: bool = False,
    proj_hidden: int = 2048,
    use_bn_proj: bool = False,
    
    state_dim: int = 0,        # >0 enables the auxiliary state-prediction head
):
    encoder = ViTTiny(
        img_size=img_size, patch_size=patch_size,
        depth=encoder_depth, heads=encoder_heads,
        dim_head=embed_dim // encoder_heads,
        mlp_dim=4 * embed_dim,
    )
    num_frames = history_size + num_preds
    predictor = ARPredictor(
        num_frames=num_frames, dim=embed_dim,
        depth=predictor_depth, heads=predictor_heads,
        dim_head=predictor_dim_head, mlp_dim=predictor_mlp_dim,
        dropout=predictor_dropout,
    )
    action_encoder = Embedder(input_dim=action_dim, emb_dim=embed_dim)
    projector = MLP(embed_dim, proj_hidden, embed_dim, use_bn=use_bn_proj)
    pred_proj = MLP(embed_dim, proj_hidden, embed_dim, use_bn=use_bn_proj)
    proprio_encoder = Embedder(input_dim=proprio_dim, emb_dim=embed_dim) if use_proprio else None
    # Auxiliary state-prediction head (train-time only): emb (D) -> rope/picker state.
    state_head = MLP(embed_dim, proj_hidden, state_dim) if state_dim and state_dim > 0 else None
    return JEPA(encoder, predictor, action_encoder, projector, pred_proj,
                proprio_encoder=proprio_encoder, state_head=state_head)


def lewm_forward(model: JEPA, batch: dict, history_size: int, num_preds: int,
                 sigreg: Optional[SIGReg] = None, sigreg_weight: float = 0.09,
                 state_weight: float = 0.0):
    """Mirrors le-wm-main/train.py:lejepa_forward."""
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    if "proprio" in batch:
        batch["proprio"] = torch.nan_to_num(batch["proprio"], 0.0)
    out = model.encode(batch)
    emb = out["emb"]            # (B, T, D)
    act_emb = out["act_emb"]    # (B, T, D)

    ctx_emb = emb[:, :history_size]
    ctx_act = act_emb[:, :history_size]
    tgt_emb = emb[:, num_preds:]
    pred_emb = model.predict(ctx_emb, ctx_act)

    pred_loss = (pred_emb - tgt_emb).pow(2).mean()
    if sigreg is not None:
        sig_loss = sigreg(emb.transpose(0, 1))
        loss = pred_loss + sigreg_weight * sig_loss
    else:
        sig_loss = torch.zeros((), device=pred_loss.device)
        loss = pred_loss

    # Auxiliary state-prediction loss (train-time only): force `emb` to retain
    # rope geometry by regressing the low-dim ground-truth state from every
    # frame's latent. Predicting from emb (the goal-matched latent) sharpens the
    # representation MPC plans in.
    state_loss = torch.zeros((), device=pred_loss.device)
    if state_weight > 0.0 and model.state_head is not None and "state" in batch:
        state_tgt = torch.nan_to_num(batch["state"].float(), 0.0)   # (B, T, S)
        state_pred = model.state_head(emb)                          # (B, T, S)
        state_loss = (state_pred - state_tgt).pow(2).mean()
        loss = loss + state_weight * state_loss

    # Return tensors (not Python floats) so the caller controls when to .item() —
    # avoids per-step GPU syncs on H100. Caller should .item() only when logging.
    return {
        "loss": loss,
        "pred_loss": pred_loss.detach(),
        "sigreg_loss": sig_loss.detach(),
        "state_loss": state_loss.detach(),
        "emb_std": emb.detach().std(),
        "emb_mean": emb.detach().mean(),
    }
