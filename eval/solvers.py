"""Cross-Entropy Method solver for MPC over the LeWM cost model.

Mirrors references/stable-worldmodel-main/stable_worldmodel/solver/cem.py,
adapted to our cost contract: the solver optimizes a plan of `horizon`
NORMALIZED future actions; cost = LeWMCost.get_cost(info_dict, candidates)
returns (B, S). Gradient-free (runs under no_grad).
"""

import torch


class CEMSolver:
    """Cross-Entropy Method over normalized action sequences.

    Args:
        cost_model: a LeWMCost (provides get_cost + normalized action box).
        horizon: number of future actions to plan.
        num_samples: candidate plans per CEM iteration.
        n_steps: CEM iterations per plan.
        topk: elite count (top samples kept to refit the distribution).
        var_scale: initial std (in normalized action units).
        min_std: floor on std to avoid premature collapse.
        device, seed.
    """

    def __init__(self, cost_model, horizon, num_samples=300, n_steps=30,
                 topk=30, var_scale=1.0, min_std=0.05, device="cuda", seed=1234,
                 freeze_idx=None, freeze_val=None):
        self.cost = cost_model
        self.horizon = horizon
        self.num_samples = num_samples
        self.n_steps = n_steps
        self.topk = topk
        self.var_scale = var_scale
        self.min_std = min_std
        self.device = device
        self.action_dim = cost_model.action_dim
        self.gen = torch.Generator(device=device).manual_seed(seed)
        # normalized action box for clipping
        self.low = cost_model.norm_low.view(1, 1, 1, -1)
        self.high = cost_model.norm_high.view(1, 1, 1, -1)
        # Optional: pin certain action dims to a fixed NORMALIZED value for every
        # candidate (e.g. confine to the ground plane by freezing the vertical
        # delta to raw-0). Keeps the optimization consistent with execution.
        self.freeze_idx = (torch.as_tensor(freeze_idx, dtype=torch.long, device=device)
                           if freeze_idx else None)
        self.freeze_val = (torch.as_tensor(freeze_val, dtype=torch.float32, device=device)
                           if freeze_val is not None else None)

    def _apply_freeze(self, x):
        if self.freeze_idx is not None:
            x[..., self.freeze_idx] = self.freeze_val
        return x

    @torch.no_grad()
    def solve(self, info_dict: dict, init_mean: torch.Tensor | None = None) -> dict:
        """Optimize a plan. Returns {'actions': (B, horizon, A)} normalized,
        and 'cost_history': list of elite-mean costs per iteration."""
        B = info_dict["hist_emb"].shape[0]
        H, A, dev = self.horizon, self.action_dim, self.device

        if init_mean is None:
            mean = torch.zeros(B, H, A, device=dev)
        else:
            mean = init_mean.to(dev)
            if mean.shape[1] < H:  # pad warm-start tail with zeros
                pad = torch.zeros(B, H - mean.shape[1], A, device=dev)
                mean = torch.cat([mean, pad], dim=1)
        std = self.var_scale * torch.ones(B, H, A, device=dev)
        mean = self._apply_freeze(mean)

        cost_history = []
        for _ in range(self.n_steps):
            noise = torch.randn(B, self.num_samples, H, A, generator=self.gen, device=dev)
            cand = mean.unsqueeze(1) + std.unsqueeze(1) * noise   # (B,S,H,A)
            cand[:, 0] = mean                                     # keep current mean
            cand = torch.clamp(cand, self.low, self.high)
            cand = self._apply_freeze(cand)                       # pin frozen dims

            costs = self.cost.get_cost(info_dict, cand)           # (B,S)
            topk_vals, topk_idx = torch.topk(costs, k=self.topk, dim=1, largest=False)
            bidx = torch.arange(B, device=dev).unsqueeze(1).expand(-1, self.topk)
            elites = cand[bidx, topk_idx]                         # (B,topk,H,A)
            mean = self._apply_freeze(elites.mean(dim=1))
            std = elites.std(dim=1).clamp_min(self.min_std)
            cost_history.append(float(topk_vals.mean().item()))

        return {"actions": mean.detach(), "cost_history": cost_history}
