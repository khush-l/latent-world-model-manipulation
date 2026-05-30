"""Scan several RopeFlatten configs with the gradient planner and dump each
final frame, so we can pick the visually cleanest flatten for the slide.

Saves per-config full frame sequences to flatten_scan.npz and a montage of
final frames to flatten_finals.png.
"""
import sys, json
from pathlib import Path
import numpy as np, torch, imageio.v2 as iio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "training")); sys.path.insert(0, str(PROJECT_ROOT / "eval"))
from model import build_lewm
from gradient_planner import GradientPlanner, GradientPlanConfig
from env_client import SubprocessSoftgym
import make_contrast_slide as mcs

CFGS = [0, 2, 3, 5, 6, 7]
N_STEPS, HIST = 50, 3
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = "training/runs/rope_full_mixed_v1/ckpt_final.pt"
model = build_lewm(history_size=HIST, use_proprio=False).to(device)
st = torch.load(ckpt, map_location=device); st = st.get("model_state_dict", st) if isinstance(st, dict) else st
model.load_state_dict(st); model.eval()
low = torch.tensor([-0.01,-0.01,-0.01,0.,-0.01,-0.01,-0.01,0.]); high = torch.tensor([0.01,0.01,0.01,1.,0.01,0.01,0.01,1.])
mk = lambda: GradientPlanner(model, GradientPlanConfig(plan_horizon=5, n_restarts=32, n_iters=15), low, high, history_size=HIST, device=device)

env = SubprocessSoftgym(env_name="RopeFlatten", num_variations=max(CFGS)+1, img_size=mcs.IMG)
store = {}
try:
    for c in CFGS:
        env._last_obs = env.reset(config_id=c, seed=c)
        ref, _ = env.make_goal_trajectory(n_steps=N_STEPS, seed=1000 + c)
        frames, perfs = mcs.run_episode(env, model, mk(), device, N_STEPS, HIST, ref_frames=ref, k=5)
        store[c] = (frames, perfs, ref[-1])
        print(f"cfg{c}: start={perfs[0]:.3f} final={perfs[-1]:.3f} best={max(perfs):.3f}")
finally:
    env.close()

# montage of final frames (labelled), 6 across
import numpy as np
cells = []
for c in CFGS:
    fr, pf, _ = store[c]
    img = fr[-1].copy()
    cells.append(img)
montage = np.concatenate([np.pad(c_, ((0,16),(0,4),(0,0)), constant_values=255) for c_ in cells], axis=1)
iio.imwrite(str(PROJECT_ROOT / "eval/runs/ropeconfig_test/flatten_finals.png"), montage)
np.savez(PROJECT_ROOT / "eval/runs/ropeconfig_test/flatten_scan.npz",
         **{f"cfg{c}_frames": np.array(store[c][0]) for c in CFGS},
         **{f"cfg{c}_perfs": np.array(store[c][1]) for c in CFGS},
         **{f"cfg{c}_goal": store[c][2] for c in CFGS})
print("order:", CFGS, "-> eval/runs/ropeconfig_test/flatten_finals.png")
