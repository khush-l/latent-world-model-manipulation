"""Render one in-distribution-vs-OOD contrast slide per RopeConfiguration goal
character (S, O, M, C, U).

Reuses the cached RopeFlatten "success" panel from make_contrast_slide.py's
slide_cache.npz (run that first) so we only need to run RopeConfiguration here.
For each target character we run the gradient planner closed-loop and compose a
side-by-side GIF: flatten (succeeds) | that character (fails, OOD).

Usage:
    python eval/make_contrast_slide.py --n-steps 50     # produces slide_cache.npz
    python eval/make_all_slides.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "training"))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from model import build_lewm                              # noqa: E402
from gradient_planner import GradientPlanner, GradientPlanConfig  # noqa: E402
from env_client import SubprocessSoftgym                  # noqa: E402
import make_contrast_slide as mcs                         # noqa: E402

CHARACTERS = ["S", "O", "M", "C", "U"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="training/runs/rope_full_mixed_v1/ckpt_final.pt")
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--hist", type=int, default=3)
    p.add_argument("--num-variations", type=int, default=40,
                   help="scan this many configs to find all 5 target characters")
    p.add_argument("--out-dir", default="eval/runs/ropeconfig_test")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = PROJECT_ROOT / args.out_dir
    cache = out_dir / "slide_cache.npz"
    if not cache.exists():
        raise SystemExit(f"missing {cache} — run make_contrast_slide.py first to cache the flatten panel")
    z = np.load(cache, allow_pickle=True)
    f_frames = list(z["f_frames"]); f_perfs = list(z["f_perfs"]); flat_goal = z["flat_goal"]
    print(f"reusing cached flatten panel (start={f_perfs[0]:.3f} best={max(f_perfs):.3f})")

    cfgp = Path(args.ckpt).parent / "config.json"
    use_proprio = bool(json.loads(cfgp.read_text()).get("use_proprio", False)) if cfgp.exists() else False
    model = build_lewm(history_size=args.hist, use_proprio=use_proprio).to(device)
    st = torch.load(args.ckpt, map_location=device)
    st = st.get("model_state_dict", st) if isinstance(st, dict) else st
    model.load_state_dict(st); model.eval()
    low = torch.tensor([-0.01, -0.01, -0.01, 0., -0.01, -0.01, -0.01, 0.])
    high = torch.tensor([0.01, 0.01, 0.01, 1., 0.01, 0.01, 0.01, 1.])

    def mk_planner():
        return GradientPlanner(model, GradientPlanConfig(plan_horizon=5, n_restarts=32, n_iters=15),
                               low, high, history_size=args.hist, device=device)

    env = SubprocessSoftgym(env_name="RopeConfiguration", num_variations=args.num_variations, img_size=mcs.IMG)
    found = {}
    try:
        for cid in range(args.num_variations):
            if len(found) == len(CHARACTERS):
                break
            obs = env.reset(config_id=cid)
            ch = obs["info"].get("goal_character")
            if ch in CHARACTERS and ch not in found:
                env._last_obs = obs
                goal = env.get_goal_image()
                print(f"== character {ch} (cfg {cid}) ==")
                frames, perfs = mcs.run_episode(env, model, mk_planner(), device,
                                                args.n_steps, args.hist, goal_img=goal)
                print(f"   start={perfs[0]:.3f} best={max(perfs):.3f}")
                out = out_dir / f"contrast_slide_{ch}.gif"
                mcs._compose(out, f_frames, f_perfs, frames, perfs, flat_goal, goal)
                found[ch] = cid
    finally:
        env.close()

    missing = [c for c in CHARACTERS if c not in found]
    print(f"\nrendered {len(found)} slides: {sorted(found)}"
          + (f"  (missing {missing} — raise --num-variations)" if missing else ""))


if __name__ == "__main__":
    main()
