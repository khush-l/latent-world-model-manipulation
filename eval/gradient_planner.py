"""Gradient-based latent trajectory planner for LeWM world models.

Our core methodological contribution. Where the LeWM paper plans with the
Cross-Entropy Method (CEM) — a gradient-free sampling optimizer that ignores
the differentiability of the learned world model — we exploit the fact that
the encoder and predictor form a *fully differentiable* latent dynamics model.
We can therefore optimize the action sequence by **backpropagating the latent
goal-distance cost directly through the predictor**, and take a handful of
Adam steps to convergence.

Why this is faster:
    CEM (paper):  n_iters × n_samples × horizon predictor evaluations
                  = 30 × 300 × 5 ≈ 45,000 rollout-steps per plan.
    Gradient:     n_restarts × n_iters × horizon (fwd) + same (bwd)
                  = 32 × 15 × 5 × ~2 ≈ 4,800 rollout-steps per plan,
                  and crucially every step moves *toward* the optimum rather
                  than randomly sampling around the current mean.

    In practice the gradient planner reaches equal or lower terminal latent
    cost in a fraction of the wall-clock time (benchmarked in
    eval/benchmark_planners.py).

Design notes:
    - The world model is frozen; only the candidate action sequence carries
      gradients. We optimize an unconstrained tensor and squash it through
      tanh into the action box, so the optimizer never needs projected
      gradients or clamping mid-step.
    - The discrete grip dimension is relaxed to continuous [0, 1] during
      planning (via the same squash) and thresholded at execution time. The
      predictor was trained on grip ∈ {0, 1} but tolerates continuous values
      at plan time — a minor train/plan mismatch we quantify in the report.
    - We use a small number of random restarts (batched on the GPU) to avoid
      poor local minima; the best-terminal-cost restart is returned.
    - Receding-horizon warm start: the previous plan's tail seeds the next.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from model import JEPA  # noqa: E402


@dataclass
class GradientPlanConfig:
    plan_horizon: int = 5
    n_restarts: int = 32      # parallel gradient-descent trajectories (batched)
    n_iters: int = 15         # Adam steps per plan
    lr: float = 0.1           # Adam LR on the (unconstrained) action params
    action_reg: float = 0.0   # optional L2 penalty on action magnitude
    warm_start: bool = True   # seed next plan from previous solution


class GradientPlanner:
    """Plans an action sequence by gradient descent through the latent dynamics."""

    def __init__(
        self,
        model: JEPA,
        config: GradientPlanConfig,
        action_low: torch.Tensor,   # (A,)
        action_high: torch.Tensor,  # (A,)
        history_size: int,
        device: str = "cuda",
    ):
        self.model = model
        self.cfg = config
        self.device = device
        self.history_size = history_size
        self.action_low = action_low.to(device)
        self.action_high = action_high.to(device)
        self.action_dim = action_low.numel()
        self._half_range = 0.5 * (self.action_high - self.action_low)
        self._mid = 0.5 * (self.action_high + self.action_low)
        self.prev_raw: Optional[torch.Tensor] = None  # (P, A) unconstrained warm start

        # Freeze the world model — we only differentiate w.r.t. actions.
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def reset(self):
        self.prev_raw = None

    # ---- action parameterization ----

    def _squash(self, raw):
        """Map an unconstrained tensor → action box via tanh. (..., A) → (..., A)."""
        return self._mid + self._half_range * torch.tanh(raw)

    # ---- differentiable rollout cost ----

    def _rollout_cost(self, z_hist, hist_act_emb, future_actions, goal_emb):
        """Differentiable terminal latent cost.

        Args:
            z_hist:        (B, H, D) encoded history latents (detached, fixed)
            hist_act_emb:  (B, H, D) history action embeddings (detached, fixed)
            future_actions:(B, P, A) candidate future actions (carry grad)
            goal_emb:      (B, D) goal latent (detached)
        Returns:
            cost: (B,) terminal latent L2² distance to goal (+ optional action reg)
        """
        H = self.history_size
        fut_act_emb = self.model.action_encoder(future_actions)  # (B, P, D)

        emb = z_hist
        act_emb = hist_act_emb
        for t in range(self.cfg.plan_horizon):
            ctx_emb = emb[:, -H:]
            ctx_act = act_emb[:, -H:]
            pred = self.model.predict(ctx_emb, ctx_act)[:, -1:]   # (B, 1, D)
            emb = torch.cat([emb, pred], dim=1)
            act_emb = torch.cat([act_emb, fut_act_emb[:, t:t + 1]], dim=1)

        terminal = emb[:, -1]                                     # (B, D)
        cost = ((terminal - goal_emb) ** 2).sum(dim=-1)           # (B,)
        if self.cfg.action_reg > 0.0:
            cost = cost + self.cfg.action_reg * (future_actions ** 2).sum(dim=(-1, -2))
        return cost

    # ---- main entry ----

    def plan(self, pixels_history, goal_emb, history_actions):
        """Optimize an action sequence and return the first action.

        Args:
            pixels_history: (H, C, H_img, W_img) recent frames, ImageNet-normalized.
            goal_emb:       (D,) target latent.
            history_actions:(H, A) actions taken over the history window.
        Returns:
            first_action: (A,) the first action of the best plan (grip un-thresholded;
                          caller thresholds the grip dims at execution).
        """
        cfg = self.cfg
        H = self.history_size
        P = cfg.plan_horizon
        R = cfg.n_restarts
        A = self.action_dim
        device = self.device

        # Encode history once (no grad — fixed context).
        with torch.no_grad():
            init = self.model.encode({"pixels": pixels_history.unsqueeze(0)})  # (1, H, D)
            z_hist = init["emb"]
            hist_act_emb = self.model.action_encoder(history_actions.unsqueeze(0))  # (1, H, D)

        # Expand fixed context to R restarts.
        z_hist_b = z_hist.expand(R, -1, -1).contiguous()
        hist_act_emb_b = hist_act_emb.expand(R, -1, -1).contiguous()
        goal_b = goal_emb.unsqueeze(0).expand(R, -1).contiguous()

        # Initialize unconstrained action params. Restart 0 = warm-start (if any),
        # restarts 1..R-1 = warm-start + noise (or pure noise if no warm start).
        if cfg.warm_start and self.prev_raw is not None:
            base = torch.cat([self.prev_raw[1:], torch.zeros(1, A, device=device)], dim=0)
        else:
            base = torch.zeros(P, A, device=device)
        raw = base.unsqueeze(0).expand(R, -1, -1).clone()
        raw[1:] += 0.5 * torch.randn(R - 1, P, A, device=device)  # restart diversity
        raw.requires_grad_(True)

        opt = torch.optim.Adam([raw], lr=cfg.lr)
        for _ in range(cfg.n_iters):
            opt.zero_grad(set_to_none=True)
            actions = self._squash(raw)                       # (R, P, A)
            cost = self._rollout_cost(z_hist_b, hist_act_emb_b, actions, goal_b)
            cost.sum().backward()
            opt.step()

        # Pick the best restart by terminal cost.
        with torch.no_grad():
            actions = self._squash(raw)
            final_cost = self._rollout_cost(z_hist_b, hist_act_emb_b, actions, goal_b)
            best = int(torch.argmin(final_cost).item())
            best_actions = actions[best]                      # (P, A)
            self.prev_raw = raw[best].detach().clone()

        return best_actions[0]

    def plan_full_sequence(self, pixels_history, goal_emb, history_actions):
        """Same as plan() but returns the entire optimized (P, A) sequence —
        used by open-loop replay-mode evaluation. Note: plan() needs gradients
        internally, so this must NOT be wrapped in torch.no_grad()."""
        self.plan(pixels_history, goal_emb, history_actions)
        with torch.no_grad():
            return self._squash(self.prev_raw)
