"""Host-side (py3.12) client for the SoftGym env_server running inside Docker.

Spawns the softgym Docker via `simulation/docker/softgym-local.sh run "..."`
with `python utils/env_server.py`, then exchanges line-delimited JSON over
the subprocess's stdin/stdout. Provides a clean EnvAdapter interface that
matches the abstract one in `eval/run_mpc.py`.

Usage:
    env = SubprocessSoftgym(env_name="RopeFlatten", num_variations=200, img_size=128)
    obs = env.reset(config_id=0)
    obs, reward, done, info = env.step(action)
    env.close()
"""

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
        docker_script: str = "simulation/docker/softgym-local.sh",
    ):
        self.env_name = env_name
        self.num_variations = num_variations
        self.img_size = img_size
        self.num_picker = num_picker
        self.headless = headless
        self.render = render
        self.docker_script = str(PROJECT_ROOT / docker_script)
        self._proc: Optional[subprocess.Popen] = None
        self._action_dim: Optional[int] = None

    # ------------------------------------------------------------- IPC

    def _ensure_started(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        # The docker wrapper takes the command as a single quoted string and
        # runs it inside the container with the softgym env vars already set.
        cmd = [self.docker_script, "run", "python -u utils/env_server.py"]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
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
