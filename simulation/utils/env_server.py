"""SoftGym env server — long-running py3.6 process inside the softgym Docker.

Protocol: line-delimited JSON over stdin/stdout.

Client sends one command per line, server replies with one line per command.
Numeric arrays are base64-encoded raw bytes with shape + dtype metadata so
that pickling is not required (safe + portable).

Commands:
    {"cmd": "ping"}                    -> {"ok": true, "pid": <int>}
    {"cmd": "reset",                   -> {"ok": true, "pixels": <ndarray>,
     "config_id": <int> | null,            "state": <ndarray>,
     "seed": <int> | null}                 "info": <dict>}
    {"cmd": "step",                    -> {"ok": true, "pixels": <ndarray>,
     "action": <ndarray>}                  "state": <ndarray>,
                                           "reward": <float>, "done": <bool>,
                                           "info": <dict>}
    {"cmd": "close"}                   -> {"ok": true}; server exits.

ndarray encoding:
    {"__ndarray__": "<base64>", "shape": [...], "dtype": "<dtype-str>"}

Error replies have {"ok": false, "error": "<msg>"}.
"""

from __future__ import print_function

import base64
import copy
import json
import os
import sys
import time
import traceback

import numpy as np

# --- protocol channel isolation -------------------------------------------
# SoftGym / PyFlex write diagnostics ("Pyflex init done!", "config 0: ...")
# to stdout via both Python prints and C-level printf. That collides with our
# JSON-over-stdout protocol. We therefore:
#   1. dup the real stdout fd to a private file object (_PROTOCOL) for replies,
#   2. redirect fd 1 (stdout) to fd 2 (stderr) so ALL noise — Python and C —
#      goes to stderr, where the client ignores it.
_PROTOCOL_FD = os.dup(1)
_PROTOCOL = os.fdopen(_PROTOCOL_FD, "w")
os.dup2(2, 1)          # C-level printf (pyflex) now writes to stderr
sys.stdout = sys.stderr  # Python-level prints now write to stderr


def _send(resp):
    """Write one JSON reply on the private protocol channel."""
    _PROTOCOL.write(json.dumps(resp) + "\n")
    _PROTOCOL.flush()


SIM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if SIM_ROOT not in sys.path:
    sys.path.insert(0, SIM_ROOT)

from softgym.registered_env import SOFTGYM_ENVS, env_arg_dict  # noqa: E402


# ---------------------------------------------------------------------------
# ndarray <-> JSON
# ---------------------------------------------------------------------------

def _encode_ndarray(arr):
    arr = np.ascontiguousarray(arr)
    return {
        "__ndarray__": base64.b64encode(arr.tobytes()).decode("ascii"),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


def _decode_ndarray(obj):
    raw = base64.b64decode(obj["__ndarray__"])
    arr = np.frombuffer(raw, dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
    return arr.copy()  # make it writable


def _encode_info(info):
    out = {}
    for k, v in info.items():
        if k == "flex_env_recorded_frames":
            continue
        try:
            arr = np.asarray(v)
        except Exception:
            out[k] = repr(v)
            continue
        if arr.shape == ():
            out[k] = float(arr) if arr.dtype.kind in "fi" else str(arr)
        elif arr.size <= 64:
            out[k] = arr.tolist()
        else:
            out[k] = _encode_ndarray(arr.astype(np.float32))
    return out


# ---------------------------------------------------------------------------
# env wrapper
# ---------------------------------------------------------------------------

class EnvSession(object):
    def __init__(self, env_name="RopeFlatten", num_variations=200, img_size=128,
                 num_picker=2, headless=True, render=True,
                 observation_mode=None, action_mode=None, render_mode=None,
                 use_cached_states=False):
        env_class = SOFTGYM_ENVS[env_name]
        kwargs = copy.deepcopy(env_arg_dict[env_name])
        kwargs["headless"] = headless
        kwargs["render"] = render
        kwargs["num_variations"] = num_variations
        kwargs["camera_width"] = 720
        kwargs["camera_height"] = 720
        kwargs["use_cached_states"] = use_cached_states
        kwargs["save_cached_states"] = False
        if observation_mode is not None:
            kwargs["observation_mode"] = observation_mode
        if action_mode is not None:
            kwargs["action_mode"] = action_mode
        if render_mode is not None:
            kwargs["render_mode"] = render_mode

        self.env = env_class(**kwargs)
        self.env_name = env_name
        self.img_size = int(img_size)
        self.num_picker = int(num_picker)
        self.action_dim = int(getattr(self.env.action_space, "shape", (4 * num_picker,))[0])

    # ----- helpers ------------------------------------------------------

    def _render(self):
        import pyflex
        from utils.collect_trajectories import render_rgb_depth  # type: ignore
        rgb, _ = render_rgb_depth(self.env, self.img_size, include_depth=False)
        return rgb  # uint8 (H, W, 3)

    def _state(self):
        import pyflex
        from utils.collect_trajectories import extract_compact_state, extract_proprio  # type: ignore
        return extract_compact_state(self.num_picker), extract_proprio(self.env, self.num_picker)

    # ----- public op handlers ------------------------------------------

    def reset(self, config_id=None, seed=None):
        if seed is not None:
            np.random.seed(int(seed))
        # softgym .reset takes (config, config_id, initial_state). Use config_id
        # to address a specific variation if asked, else random.
        if config_id is None:
            self.env.reset()
        else:
            cfg = self.env.cached_configs[int(config_id) % len(self.env.cached_configs)] \
                  if getattr(self.env, "cached_configs", None) else None
            init = self.env.cached_init_states[int(config_id) % len(self.env.cached_init_states)] \
                   if getattr(self.env, "cached_init_states", None) else None
            self.env.reset(config=cfg, config_id=int(config_id), initial_state=init)
        state, proprio = self._state()
        return {
            "pixels": _encode_ndarray(self._render()),
            "state": _encode_ndarray(state),
            "proprio": _encode_ndarray(proprio),
            "info": {},
        }

    def make_goal_trajectory(self, n_steps, noise_scale=0.02):
        """Generate an expert reference trajectory ON THE CURRENT ROPE, then
        restore the env to its current state.

        Snapshots env state, runs the scripted geometric expert for `n_steps`,
        recording a rendered frame + normalized_performance at each step, then
        restores the snapshot so the caller's env is untouched.

        Returns (frames, perfs):
          frames: (n_steps+1, H, W, 3) uint8 — the expert's flattening sequence
          perfs:  list of normalized_performance per step
        These frames become the reference whose k-ahead entries are MPC goals.
        """
        import pyflex  # noqa: F401
        from geometric_policy import make_policy  # type: ignore

        snapshot = self.env.get_state()
        policy = make_policy(self.env_name, self.env, self.num_picker, kind="geometric",
                             noise_scale=noise_scale)
        policy.reset()

        frames = [self._render()]
        perfs = []
        for _ in range(int(n_steps)):
            action = policy.get_action()
            _, _, done, info = self.env.step(action)
            frames.append(self._render())
            perfs.append(float(info.get("normalized_performance",
                                        info.get("performance", 0.0))))
            if done:
                break

        # Restore the env to exactly where it was before the expert rollout.
        self.env.set_state(snapshot)

        return np.asarray(frames, dtype=np.uint8), perfs

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.action_dim,):
            raise ValueError("expected action shape (%d,) got %s" % (self.action_dim, action.shape))
        _, reward, done, info = self.env.step(action)
        state, proprio = self._state()
        return {
            "pixels": _encode_ndarray(self._render()),
            "state": _encode_ndarray(state),
            "proprio": _encode_ndarray(proprio),
            "reward": float(reward),
            "done": bool(done),
            "info": _encode_info(info),
        }


# ---------------------------------------------------------------------------
# top-level loop
# ---------------------------------------------------------------------------

def _log(msg):
    sys.stderr.write("[env_server] %s\n" % msg)
    sys.stderr.flush()


def main():
    session = None
    _log("ready, awaiting commands on stdin (line-delimited JSON)")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "ping":
                resp = {"ok": True, "pid": os.getpid()}
            elif cmd == "init":
                # Optional explicit init; otherwise auto-init on first reset.
                kwargs = dict(req)
                kwargs.pop("cmd", None)
                session = EnvSession(**kwargs)
                resp = {"ok": True, "action_dim": session.action_dim,
                        "img_size": session.img_size}
            elif cmd == "reset":
                if session is None:
                    session = EnvSession(env_name=req.get("env_name", "RopeFlatten"))
                resp = {"ok": True}
                resp.update(session.reset(config_id=req.get("config_id"),
                                          seed=req.get("seed")))
            elif cmd == "step":
                action = _decode_ndarray(req["action"]) if isinstance(req["action"], dict) \
                         else np.asarray(req["action"], dtype=np.float32)
                resp = {"ok": True}
                resp.update(session.step(action))
            elif cmd == "make_goal_trajectory":
                frames, perfs = session.make_goal_trajectory(
                    n_steps=int(req.get("n_steps", 75)),
                    noise_scale=float(req.get("noise_scale", 0.02)),
                )
                resp = {"ok": True, "frames": _encode_ndarray(frames), "perfs": perfs}
            elif cmd == "close":
                resp = {"ok": True}
                _send(resp)
                break
            else:
                resp = {"ok": False, "error": "unknown cmd: %r" % cmd}
        except Exception as e:
            resp = {"ok": False, "error": str(e), "traceback": traceback.format_exc()}
            _log("error: %s" % e)
        _send(resp)


if __name__ == "__main__":
    main()
