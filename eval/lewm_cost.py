"""
Mirrors references/stable-worldmodel-main/stable_worldmodel/wm/lewm/lewm.py
(encode / rollout / criterion / get_cost), adapted to our JEPA
(training/model.py) and our action/proprio conventions.

Action layout (A=8):  [d0_xyz(3), grip0, d1_xyz(3), grip1]
Proprio layout (P=8):  [p0_xyz(3), p1_xyz(3), hold0, hold1]
"""

import sys
from pathlib import Path
from typing import Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from model import JEPA  # noqa: E402


class LeWMCost:
    """Wraps a frozen JEPA as a Costable world model for the CEM solver.

    Context (history embeddings, real history actions, current proprio, goal
    embedding) is encoded ONCE per replan and passed through `info_dict`;
    `get_cost` broadcasts it over the CEM sample dimension internally (no pixel
    re-encoding per candidate).
    """

    def __init__(
        self,
        model: JEPA,
        action_low: torch.Tensor,    # (A,) env action box, raw scale
        action_high: torch.Tensor,   # (A,)
        history_size: int,
        action_mean: torch.Tensor,   # (A,) z-score stats from training data
        action_std: torch.Tensor,    # (A,)
        proprio_mean: Optional[torch.Tensor] = None,  # (P,)
        proprio_std: Optional[torch.Tensor] = None,   # (P,)
        device: str = "cuda",
    ):
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.H = history_size
        self.action_dim = action_low.numel()
        self.uses_proprio = getattr(model, "proprio_encoder", None) is not None

        self.action_low = action_low.to(device)
        self.action_high = action_high.to(device)
        self.action_mean = action_mean.to(device)
        self.action_std = action_std.to(device)
        self.proprio_mean = proprio_mean.to(device) if proprio_mean is not None else None
        self.proprio_std = proprio_std.to(device) if proprio_std is not None else None

        # Normalized action box (for the solver to clip candidates into).
        self.norm_low = self.norm_action(self.action_low)
        self.norm_high = self.norm_action(self.action_high)

        self._goal_emb: Optional[torch.Tensor] = None       # (D,) active goal/subgoal
        self._waypoints: Optional[torch.Tensor] = None       # (N, D) expert-demo latent path
        self.tempdist_head = None                            # optional learned cost-to-go

    # ---- normalization ----

    def norm_action(self, a):
        return (a - self.action_mean) / self.action_std

    def denorm_action(self, a):
        return a * self.action_std + self.action_mean

    def _norm_proprio(self, p):
        if self.proprio_mean is None:
            return p
        return (p - self.proprio_mean) / self.proprio_std

    # ---- encoding ----

    @torch.no_grad()
    def encode_obs(self, pixels):
        """pixels: (H, C, H_img, W_img) ImageNet-normalized -> (H, D) detached."""
        out = self.model.encode({"pixels": pixels.unsqueeze(0).to(self.device)})
        return out["emb"][0].detach()  # (H, D)

    @torch.no_grad()
    def set_goal(self, goal_pixels):
        """goal_pixels: (C, H_img, W_img) -> cache (D,) goal embedding."""
        out = self.model.encode({"pixels": goal_pixels.unsqueeze(0).unsqueeze(0).to(self.device)})
        self._goal_emb = out["emb"][0, 0].detach()  # (D,)
        return self._goal_emb

    # ---- latent subgoals from an expert demo (crumpled -> flat) ----

    @torch.no_grad()
    def set_waypoints(self, frames_chw):
        """Encode an expert demo into a latent waypoint path.

        frames_chw: (N, C, H_img, W_img) ImageNet-normalized frames, ordered from
        the episode's start state to the flat goal. Cached as (N, D) latents that
        pick_subgoal() walks along. These are real encoded frames, so every
        waypoint is guaranteed on the data manifold.
        """
        self._waypoints = self.encode_obs(frames_chw.to(self.device))  # (N, D)
        return self._waypoints

    @torch.no_grad()
    def set_subgoal_index(self, idx):
        """Set the active goal to waypoint `idx` (clamped). Used by time-indexed
        selection: the expert demo is time-aligned to the episode start, so the
        elapsed env-step directly indexes the waypoint — no latent matching, which
        avoids mis-selection in the blurry intermediate-latent regime."""
        i = min(max(int(idx), 0), self._waypoints.shape[0] - 1)
        self._goal_emb = self._waypoints[i]
        return i

    @torch.no_grad()
    def pick_subgoal(self, cur_emb, lookahead, min_idx=0):
        """Set the active goal to the waypoint `lookahead` steps past the one
        nearest the current latent (monotonic: search only idx >= min_idx).

        cur_emb: (D,) current state latent. Returns (nearest_idx, target_idx).
        """
        W = self._waypoints
        d = ((W[min_idx:] - cur_emb.to(W).unsqueeze(0)) ** 2).sum(dim=-1)  # (N-min_idx,)
        nearest = int(d.argmin().item()) + min_idx
        target = min(nearest + lookahead, W.shape[0] - 1)
        self._goal_emb = W[target]
        return nearest, target

    # ---- future proprio from planned actions (raw space) ----

    def _integrate_future_proprio(self, start_proprio_raw, planned_raw):
        """Derive proprio for the integrated future steps from planned actions.

        Picker is directly controllable: picker_xyz += action delta; grip :=
        action grip command. Returns proprio for steps that need integration.

        start_proprio_raw: (N, P) current raw proprio (state s_{H-1}).
        planned_raw:       (N, m, A) raw planned action deltas (m = n-1 used).
        Returns: (N, m, P) raw proprio for s_H .. s_{H-1+m}.
        """
        p0 = start_proprio_raw[:, 0:3].unsqueeze(1)   # (N,1,3)
        p1 = start_proprio_raw[:, 3:6].unsqueeze(1)
        d0 = planned_raw[:, :, 0:3]                   # (N,m,3)
        d1 = planned_raw[:, :, 4:7]
        pos0 = p0 + torch.cumsum(d0, dim=1)
        pos1 = p1 + torch.cumsum(d1, dim=1)
        g0 = planned_raw[:, :, 3:4]
        g1 = planned_raw[:, :, 7:8]
        return torch.cat([pos0, pos1, g0, g1], dim=-1)  # (N,m,P)

    # ---- cost ----

    @torch.no_grad()
    def _rollout(self, info_dict: dict, candidates: torch.Tensor) -> torch.Tensor:
        """Imagined latent trajectory for each candidate plan.

        info_dict (all per-env, batch B):
            hist_emb:        (B, H, D)    encoded history latents (s_0..s_{H-1})
            hist_act_norm:   (B, H-1, A)  NORMALIZED real past actions (a_0..a_{H-2})
            start_proprio:   (B, P)       raw current proprio (s_{H-1}); optional
            hist_proprio_norm:(B, H, P)   normalized real history proprio; optional
        candidates:          (B, S, n, A) NORMALIZED planned actions (a_{H-1}..)
        Returns: emb_seq (B*S, H+n, D) = [s_0..s_{H-1} history, s_H..s_{H-1+n} imagined]
        """
        H = self.H
        B, S, n, A = candidates.shape
        D = info_dict["hist_emb"].shape[-1]
        dev = self.device

        hist_emb = info_dict["hist_emb"].to(dev)              # (B,H,D)
        hist_act = info_dict["hist_act_norm"].to(dev)         # (B,H-1,A)

        # Flatten (B,S) -> N for batched rollout.
        N = B * S
        emb0 = hist_emb.unsqueeze(1).expand(B, S, H, D).reshape(N, H, D)
        hist_act_e = hist_act.unsqueeze(1).expand(B, S, H - 1, A).reshape(N, H - 1, A)
        cand = candidates.reshape(N, n, A)

        # Full normalized action sequence: [H-1 real past, n planned] -> M = H-1+n
        act_seq_norm = torch.cat([hist_act_e, cand], dim=1)   # (N, M, A)
        cond = self.model.action_encoder(act_seq_norm)        # (N, M, D)

        # Proprio conditioning (cond += proprio_emb), if the model uses it.
        if self.uses_proprio and info_dict.get("start_proprio") is not None:
            hist_p = info_dict["hist_proprio_norm"].to(dev)   # (B,H,P)
            hist_p = hist_p.unsqueeze(1).expand(B, S, H, -1).reshape(N, H, -1)
            start_p = info_dict["start_proprio"].to(dev)
            start_p = start_p.unsqueeze(1).expand(B, S, -1).reshape(N, -1)
            # integrate proprio for future steps s_H..s_{H-2+n} (n-1 of them),
            # from raw planned actions (denormalized).
            if n > 1:
                planned_raw = self.denorm_action(cand[:, : n - 1])  # (N,n-1,A)
                fut_p_raw = self._integrate_future_proprio(start_p, planned_raw)  # (N,n-1,P)
                fut_p = self._norm_proprio(fut_p_raw)
                prop_seq = torch.cat([hist_p, fut_p], dim=1)        # (N, H-1+n, P) = (N,M,P)
            else:
                prop_seq = hist_p
            cond = cond + self.model.proprio_encoder(prop_seq)

        # Sliding-window autoregressive rollout. Window length = H.
        # Predict s_H .. s_{H-1+n} (n steps); a_{H-1}=planned[0] drives s_H.
        emb_list = list(emb0.unbind(dim=1))   # H tensors (N, D)
        for j in range(n):
            win_emb = torch.stack(emb_list[j : j + H], dim=1)  # (N,H,D)
            win_cond = cond[:, j : j + H]                      # (N,H,D)
            pred = self.model.predict(win_emb, win_cond)[:, -1]  # (N, D)
            emb_list.append(pred)

        return torch.stack(emb_list, dim=1)                    # (N, H+n, D)

    @torch.no_grad()
    def get_cost(self, info_dict: dict, candidates: torch.Tensor) -> torch.Tensor:
        """Cost of the terminal latent vs the goal for each candidate plan. Returns (B, S).

        Default = terminal-latent MSE (the LeWM metric). If a learned temporal-distance
        head is attached (set_tempdist_head), cost = predicted #steps-to-goal from the
        rolled-out terminal latent — an informative cost-to-go in place of flat MSE.
        """
        B, S = candidates.shape[:2]
        goal = self._goal_emb.to(self.device)                 # (D,)
        emb_seq = self._rollout(info_dict, candidates)         # (B*S, H+n, D)
        terminal = emb_seq[:, -1]                              # (N, D) = s_{H-1+n}
        if self.tempdist_head is not None:
            cost = self.tempdist_head(terminal, goal.unsqueeze(0).expand_as(terminal))  # (N,)
        else:
            cost = ((terminal - goal.unsqueeze(0)) ** 2).sum(dim=-1)  # (N,)
        return cost.reshape(B, S)

    def set_tempdist_head(self, head):
        """Attach a learned temporal-distance head; switches get_cost to cost-to-go."""
        self.tempdist_head = head.eval().to(self.device)
        for p in self.tempdist_head.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def imagine(self, info_dict: dict, plan: torch.Tensor) -> torch.Tensor:
        """Decode-ready imagined latents for ONE plan (B=S=1).

        plan: (n, A) normalized planned actions.
        Returns: (n+1, D) = [current s_{H-1}, then n imagined s_H..s_{H-1+n}].
        """
        cand = plan.to(self.device).unsqueeze(0).unsqueeze(0)  # (1,1,n,A)
        emb_seq = self._rollout(info_dict, cand)[0]            # (H+n, D)
        return emb_seq[self.H - 1:]                            # (n+1, D)
