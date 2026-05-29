"""Cross-Entropy Method planner over LeWM latent dynamics.

Given:
  - a trained JEPA (encoder + predictor)
  - the current observation's encoded history latents
  - a goal latent
  - action bounds

CEM samples N candidate action sequences from a diagonal Gaussian, scores
each by terminal latent distance after a model rollout, keeps the top-K
elites, and refits the Gaussian. Repeat for I iterations.

Mirrors references/le-wm-main/config/eval/solver/cem.yaml hyperparameters:
  num_samples=300, n_iters=30 (PushT) / 10 (others), topk=30 (top 10%),
  var_scale=1.0, plan_horizon=5.

Latency target: ~1 s per plan on L4 with paper config (single replan).
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from rollout import rollout, goal_cost  # noqa: E402


@dataclass
class CEMConfig:
    """CEM hyperparameters. Defaults match references/le-wm-main solver/cem.yaml."""
    num_samples: int = 300        # population per iteration
    n_iters: int = 30             # CEM iterations per plan
    topk: int = 30                # elite count (top 10%)
    var_scale: float = 1.0        # initial sampling variance scale (× action range)
    plan_horizon: int = 5         # number of future actions per plan
    min_std: float = 1e-3         # floor on std-dev to avoid collapse


class CEMPlanner:
    """Stateful CEM planner — call .plan(...) each time you need a new sequence.

    Uses warm-start: each call reuses the previous solution's mean shifted by
    one step (receding-horizon convention) as the next plan's initial mean.
    """

    def __init__(
        self,
        model,
        config: CEMConfig,
        action_low: torch.Tensor,    # (A,)
        action_high: torch.Tensor,   # (A,)
        history_size: int,
        device: str = "cuda",
    ):
        self.model = model
        self.config = config
        self.action_low = action_low.to(device)
        self.action_high = action_high.to(device)
        self.action_dim = action_low.numel()
        self.history_size = history_size
        self.device = device
        self.prev_mean: Optional[torch.Tensor] = None  # (H_plan, A)

    def reset(self):
        """Clear warm-start state — call between episodes."""
        self.prev_mean = None

    @torch.no_grad()
    def plan(self, pixels_history: torch.Tensor, goal_emb: torch.Tensor,
             history_actions: torch.Tensor) -> torch.Tensor:
        """Optimize an action sequence and return the first action.

        Args:
            pixels_history: (H, C, H_img, W_img) — recent H pixel frames.
            goal_emb:       (D,)                 — target latent.
            history_actions:(H, A)               — the actions actually taken
                                                    over the history window
                                                    (used to align predictor
                                                    inputs with pixel history).

        Returns:
            first_action: (A,) — the first action of the optimized plan, to be
                                  executed in the env this step.
        """
        cfg = self.config
        H_hist = self.history_size
        H_plan = cfg.plan_horizon
        N = cfg.num_samples
        A = self.action_dim
        T = H_hist + H_plan
        device = self.device

        # Initial mean: warm-start if available, else zero (env-action midpoint).
        if self.prev_mean is None:
            mean = torch.zeros(H_plan, A, device=device)
        else:
            # Shift previous solution forward by 1 step, pad with zero at the end.
            shifted = torch.cat(
                [self.prev_mean[1:], torch.zeros(1, A, device=device)], dim=0
            )
            mean = shifted

        # Initial std: var_scale × half-action-range. After clipping, the
        # Gaussian effectively spans the action space.
        half_range = 0.5 * (self.action_high - self.action_low)  # (A,)
        std = cfg.var_scale * half_range.unsqueeze(0).expand(H_plan, A).clone()
        std = torch.clamp(std, min=cfg.min_std)

        # Pre-expand pixel history + history actions to (1, N, H_hist, ...).
        px_hist = pixels_history.unsqueeze(0).unsqueeze(0).expand(1, N, *pixels_history.shape).contiguous()
        hist_act = history_actions.unsqueeze(0).unsqueeze(0).expand(1, N, *history_actions.shape).contiguous()
        goal_emb_batched = goal_emb.unsqueeze(0)  # (1, D)

        for it in range(cfg.n_iters):
            # Sample N candidate plans from current Gaussian, clipped to action bounds.
            noise = torch.randn(N, H_plan, A, device=device)
            samples = mean.unsqueeze(0) + std.unsqueeze(0) * noise   # (N, H_plan, A)
            samples = torch.clamp(samples,
                                  self.action_low.view(1, 1, A),
                                  self.action_high.view(1, 1, A))

            # Build full action sequence: [history_actions, sampled_future].
            # Shape: (1, N, T, A)
            full_act = torch.cat([hist_act, samples.unsqueeze(0)], dim=2)

            # Roll out latents in parallel.
            emb = rollout(self.model, px_hist, full_act, history_size=H_hist)  # (1, N, T, D)
            costs = goal_cost(emb, goal_emb_batched).squeeze(0)  # (N,)

            # Pick elites and refit Gaussian (diagonal).
            elite_idx = torch.topk(costs, k=cfg.topk, largest=False).indices
            elites = samples[elite_idx]                # (topk, H_plan, A)
            mean = elites.mean(dim=0)                  # (H_plan, A)
            std = elites.std(dim=0) + cfg.min_std       # (H_plan, A)

        self.prev_mean = mean.detach().clone()
        # Final plan = mean of the last iteration's elite distribution.
        # Return the first action only — the executor calls plan() again next step.
        return torch.clamp(mean[0], self.action_low, self.action_high)
