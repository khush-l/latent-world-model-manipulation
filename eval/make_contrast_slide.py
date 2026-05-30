"""Presentation slide: in-distribution success vs out-of-distribution failure.

Runs the gradient MPC planner closed-loop on two SoftGym tasks with the SAME
RopeFlatten-trained checkpoint and renders one annotated side-by-side GIF:

  LEFT  — RopeFlatten (in-distribution): rope | flat goal. The planner flattens.
  RIGHT — RopeConfiguration (out-of-distribution): rope | 'S' goal. The planner
          can't form the character — the model never trained on shaping.

The contrast is the slide: "solves what it was trained on, fails on a task it
never saw."

Usage:
    python eval/make_contrast_slide.py --ckpt training/runs/rope_full_mixed_v1/ckpt_final.pt
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as iio
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "training"))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from model import build_lewm                              # noqa: E402
from rollout import encode_pixels                         # noqa: E402
from gradient_planner import GradientPlanner, GradientPlanConfig  # noqa: E402
from env_client import SubprocessSoftgym                  # noqa: E402
from sample_goals import _to_imagenet_float               # noqa: E402

IMG = 128


def _enc(model, frame_uint8, device):
    chw = _to_imagenet_float(frame_uint8)
    t = torch.from_numpy(chw).float().to(device).unsqueeze(0).unsqueeze(0)
    return encode_pixels(model, t).squeeze(0).squeeze(0)


def run_episode(env, model, planner, device, n_steps, hist, goal_img=None, ref_frames=None, k=5):
    """Closed-loop gradient MPC. If `ref_frames` is given, follow k-ahead
    subgoals along that reference (the effective flatten setup); otherwise chase
    a single fixed `goal_img`. Returns frames, perfs."""
    obs = env._last_obs
    fixed_goal_emb = _enc(model, goal_img, device) if goal_img is not None else None
    T_ref = len(ref_frames) if ref_frames is not None else 0
    init_px = torch.from_numpy(_to_imagenet_float(obs["pixels"])).float().to(device)
    pixel_hist = collections.deque([init_px] * hist, maxlen=hist)
    action_hist = collections.deque([torch.zeros(8, device=device)] * hist, maxlen=hist)
    planner.reset()
    frames = [obs["pixels"]]
    perfs = [float(obs["info"].get("normalized_performance", obs["info"].get("performance", 0.0)))]
    for t in range(n_steps):
        if ref_frames is not None:
            goal_emb = _enc(model, ref_frames[min(t + k, T_ref - 1)], device)
        else:
            goal_emb = fixed_goal_emb
        px = torch.stack(list(pixel_hist), dim=0)
        act = torch.stack(list(action_hist), dim=0)
        a = planner.plan(px, goal_emb, act)
        out = env.step(a.cpu().numpy().astype(np.float32))
        frames.append(out["pixels"])
        perfs.append(float(out["info"].get("normalized_performance", out["info"].get("performance", 0.0))))
        pixel_hist.append(torch.from_numpy(_to_imagenet_float(out["pixels"])).float().to(device))
        action_hist.append(a.detach())
        if out["done"]:
            break
    return frames, perfs


def _font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def panel(rope, goal, title, perf, ok, scale=2):
    """One task panel: title bar + [rope | sep | goal] + perf line, scaled up."""
    sep = np.full((IMG, 6, 3), 255, np.uint8)
    body = np.concatenate([rope, sep, goal], axis=1)            # (128, 262, 3)
    W = body.shape[1]
    title_h, foot_h = 26, 22
    canvas = np.full((title_h + IMG + foot_h, W, 3), 255, np.uint8)
    canvas[title_h:title_h + IMG] = body
    im = Image.fromarray(canvas)
    if scale != 1:
        im = im.resize((W * scale, im.height * scale), Image.NEAREST)
    d = ImageDraw.Draw(im)
    col = (20, 130, 30) if ok else (200, 40, 40)
    d.text((6 * scale, 5 * scale), title, fill=(0, 0, 0), font=_font(11 * scale))
    d.text((6 * scale, (title_h + IMG + 4) * scale),
           f"perf {perf:+.3f}", fill=col, font=_font(11 * scale))
    # "goal" label in the footer, under the right (goal) sub-image
    d.text(((IMG + 40) * scale, (title_h + IMG + 4) * scale),
           "↑ goal", fill=(90, 90, 90), font=_font(10 * scale))
    return np.asarray(im)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="training/runs/rope_full_mixed_v1/ckpt_final.pt")
    p.add_argument("--flatten-cfg", type=int, default=0)
    p.add_argument("--config-cfg", type=int, default=0)
    p.add_argument("--n-steps", type=int, default=30)
    p.add_argument("--hist", type=int, default=3)
    p.add_argument("--out", default="eval/runs/ropeconfig_test/contrast_slide.gif")
    p.add_argument("--from-cache", action="store_true",
                   help="Re-render the slide from cached frames (skip the envs).")
    args = p.parse_args()

    out = PROJECT_ROOT / args.out
    cache = out.parent / "slide_cache.npz"

    if args.from_cache and cache.exists():
        z = np.load(cache, allow_pickle=True)
        f_frames = list(z["f_frames"]); f_perfs = list(z["f_perfs"])
        c_frames = list(z["c_frames"]); c_perfs = list(z["c_perfs"])
        flat_goal = z["flat_goal"]; conf_goal = z["conf_goal"]
        print(f"loaded cached frames from {cache}")
        _compose(out, f_frames, f_perfs, c_frames, c_perfs, flat_goal, conf_goal)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    # ---- RopeFlatten (in-distribution): goal = expert's flattened final frame ----
    print("== RopeFlatten ==")
    envF = SubprocessSoftgym(env_name="RopeFlatten", num_variations=args.flatten_cfg + 1, img_size=IMG)
    try:
        envF._last_obs = envF.reset(config_id=args.flatten_cfg, seed=args.flatten_cfg)
        ref_frames, _ = envF.make_goal_trajectory(n_steps=args.n_steps, seed=1000 + args.flatten_cfg)
        flat_goal = ref_frames[-1]  # shown as the goal panel
        f_frames, f_perfs = run_episode(envF, model, mk_planner(), device, args.n_steps, args.hist,
                                        ref_frames=ref_frames, k=5)
    finally:
        envF.close()
    print(f"   flatten: start={f_perfs[0]:.3f} best={max(f_perfs):.3f}")

    # ---- RopeConfiguration (out-of-distribution): goal = target character ----
    print("== RopeConfiguration ==")
    envC = SubprocessSoftgym(env_name="RopeConfiguration", num_variations=args.config_cfg + 1, img_size=IMG)
    try:
        envC._last_obs = envC.reset(config_id=args.config_cfg)
        conf_goal = envC.get_goal_image()
        c_frames, c_perfs = run_episode(envC, model, mk_planner(), device, args.n_steps, args.hist,
                                        goal_img=conf_goal)
    finally:
        envC.close()
    print(f"   config: start={c_perfs[0]:.3f} best={max(c_perfs):.3f}")

    np.savez(cache,
             f_frames=np.array(f_frames), f_perfs=np.array(f_perfs),
             c_frames=np.array(c_frames), c_perfs=np.array(c_perfs),
             flat_goal=flat_goal, conf_goal=conf_goal)
    print(f"cached frames -> {cache}")
    _compose(out, f_frames, f_perfs, c_frames, c_perfs, flat_goal, conf_goal)


def _compose(out, f_frames, f_perfs, c_frames, c_perfs, flat_goal, conf_goal):
    """Render the annotated side-by-side GIF from collected frames."""
    T = max(len(f_frames), len(c_frames))
    def at(seq, i): return seq[min(i, len(seq) - 1)]
    out_frames = []
    for i in range(T):
        L = panel(at(f_frames, i), flat_goal, "RopeFlatten  (in-distribution)",
                  at(f_perfs, i), ok=True)
        R = panel(at(c_frames, i), conf_goal, "RopeConfiguration  (out-of-distribution)",
                  at(c_perfs, i), ok=False)
        g = np.full((L.shape[0], 24, 3), 255, np.uint8)
        out_frames.append(np.concatenate([L, g, R], axis=1))
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.mimsave(out, out_frames, duration=0.18, loop=0)
    print(f"\nwrote {out}  ({T} frames, {out_frames[0].shape[1]}x{out_frames[0].shape[0]})")


if __name__ == "__main__":
    main()
