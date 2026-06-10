import base64
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _decode_ndarray(obj):
    if not isinstance(obj, dict) or "__ndarray__" not in obj:
        return obj
    raw = base64.b64decode(obj["__ndarray__"])
    arr = np.frombuffer(raw, dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
    return arr.copy()


def _encode_ndarray(arr):
    arr = np.ascontiguousarray(arr)
    return {
        "__ndarray__": base64.b64encode(arr.tobytes()).decode("ascii"),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


class SubprocessSoftgym:
    """Closed-loop SoftGym env adapter that runs in a separate py36 Docker process.

    The Docker container is launched lazily on first reset(). Stays alive across
    multiple episodes; call .close() to terminate.
    """

    def __init__(
        self,
        env_name: str = "RopeFlatten",
        num_variations: int = 200,
        img_size: int = 128,
        num_picker: int = 2,
        headless: bool = True,
        render: bool = True,
        env_seed: int = 0,
        docker_script: str = "simulation/docker/softgym-local.sh",
    ):
        self.env_name = env_name
        self.num_variations = num_variations
        self.img_size = img_size
        self.num_picker = num_picker
        self.headless = headless
        self.render = render
        self.env_seed = env_seed
        self.docker_script = str(PROJECT_ROOT / docker_script)
        self._proc: Optional[subprocess.Popen] = None
        self._action_dim: Optional[int] = None

    # ------------------------------------------------------------- IPC

    def _ensure_started(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        # The docker wrapper takes the command as a single quoted string and
        # runs it inside the container with the softgym env vars already set.
        # SOFTGYM_STDIN=1 makes the wrapper pass `docker run -i` so the
        # container keeps stdin open for our JSON command stream.
        cmd = [self.docker_script, "run", "python -u utils/env_server.py"]
        proc_env = dict(os.environ, SOFTGYM_STDIN="1")
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
            env=proc_env,
        )
        # Initialize the env inside the server.
        resp = self._call({
            "cmd": "init",
            "env_name": self.env_name,
            "num_variations": self.num_variations,
            "img_size": self.img_size,
            "num_picker": self.num_picker,
            "headless": self.headless,
            "render": self.render,
            "env_seed": self.env_seed,
        })
        self._action_dim = int(resp["action_dim"])

    def _call(self, req: dict) -> dict:
        self._ensure_started()
        assert self._proc is not None and self._proc.stdin and self._proc.stdout
        self._proc.stdin.write(json.dumps(req) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            err = self._proc.stderr.read() if self._proc.stderr else ""
            raise RuntimeError("env_server died:\n" + err)
        resp = json.loads(line)
        if not resp.get("ok", False):
            raise RuntimeError("env_server error: %s\n%s" % (
                resp.get("error", "(no error msg)"),
                resp.get("traceback", "")
            ))
        return resp

    # -------------------------------------------------------- EnvAdapter

    def reset(self, config_id: Optional[int] = None, seed: Optional[int] = None):
        resp = self._call({"cmd": "reset", "config_id": config_id, "seed": seed})
        return {
            "pixels":  _decode_ndarray(resp["pixels"]),
            "state":   _decode_ndarray(resp["state"]),
            "proprio": _decode_ndarray(resp["proprio"]),
            "info":    resp.get("info", {}),
        }

    def make_goal_trajectory(self, n_steps: int = 75, noise_scale: float = 0.02, seed=None):
        """Generate an expert reference trajectory on the current rope (the env
        is restored to its pre-call state afterward). Pass `seed` for a
        reproducible reference (identical subgoals across runs).

        Returns (frames, perfs):
            frames: (n_steps+1, H, W, 3) uint8
            perfs:  list[float] normalized_performance per step
        """
        req = {"cmd": "make_goal_trajectory", "n_steps": int(n_steps),
               "noise_scale": float(noise_scale)}
        if seed is not None:
            req["seed"] = int(seed)
        resp = self._call(req)
        return _decode_ndarray(resp["frames"]), resp["perfs"]

    def grip_endpoints(self, max_steps: int = 40):
        """Run the scripted geometric expert through APPROACH->DESCEND->GRIP so both
        pickers grip the rope ENDPOINTS, then stop (env left gripped, NOT restored).
        Returns the gripped-state obs (same keys as reset) plus 'hold' / 'grip_steps'."""
        resp = self._call({"cmd": "grip_endpoints", "max_steps": int(max_steps)})
        return {
            "pixels":  _decode_ndarray(resp["pixels"]),
            "state":   _decode_ndarray(resp["state"]),
            "proprio": _decode_ndarray(resp["proprio"]),
            "info":    resp.get("info", {}),
            "hold":    resp.get("hold", []),
            "grip_steps": resp.get("grip_steps", 0),
        }

    def get_goal_image(self):
        """Task goal image as (img_size, img_size, 3) uint8 (e.g. RopeConfiguration
        target character). Falls back to the current frame if the env has none."""
        resp = self._call({"cmd": "get_goal_image"})
        return _decode_ndarray(resp["goal_image"])

    def step(self, action: np.ndarray):
        resp = self._call({"cmd": "step", "action": _encode_ndarray(action.astype(np.float32))})
        return {
            "pixels":  _decode_ndarray(resp["pixels"]),
            "state":   _decode_ndarray(resp["state"]),
            "proprio": _decode_ndarray(resp["proprio"]),
            "reward":  float(resp["reward"]),
            "done":    bool(resp["done"]),
            "info":    {k: (_decode_ndarray(v) if isinstance(v, dict) and "__ndarray__" in v else v)
                        for k, v in resp.get("info", {}).items()},
        }

    @property
    def action_dim(self) -> int:
        self._ensure_started()
        return self._action_dim  # type: ignore

    def close(self):
        if self._proc is None or self._proc.poll() is not None:
            return
        try:
            self._call({"cmd": "close"})
        except Exception:
            pass
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._proc = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
