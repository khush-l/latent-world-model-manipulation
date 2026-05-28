#!/usr/bin/env python3
"""Small HTTP policy server for SoftGym rollouts.

SoftGym runs in the old CUDA/Python Docker image while ACT/SmolVLA run in the
modern LeRobot environment. This server keeps the LeRobot policy on the host
and exposes two endpoints:

  GET  /health
  POST /reset
  POST /predict   body: npz with image uint8 HWC, state float32, optional task
"""

from __future__ import annotations

import argparse
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors


class PolicyRuntime:
    def __init__(self, checkpoint: Path, device: str, task: str):
        self.checkpoint = checkpoint
        self.task = task
        self.cfg = PreTrainedConfig.from_pretrained(checkpoint)
        self.cfg.device = device
        policy_cls = get_policy_class(self.cfg.type)
        self.policy = policy_cls.from_pretrained(checkpoint)
        self.policy.eval()
        self.device = next(self.policy.parameters()).device
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.cfg, pretrained_path=str(checkpoint)
        )
        self.n_predictions = 0

    def reset(self):
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def predict(self, image_hwc: np.ndarray, state: np.ndarray, task: str | None = None) -> np.ndarray:
        if image_hwc.ndim != 3 or image_hwc.shape[-1] != 3:
            raise ValueError(f"expected image HWC RGB, got {image_hwc.shape}")
        if state.shape != (23,):
            raise ValueError(f"expected 23D observation.state, got {state.shape}")

        image_chw = np.transpose(image_hwc.astype(np.float32) / 255.0, (2, 0, 1))
        obs = {
            "observation.images.front": torch.from_numpy(image_chw),
            "observation.state": torch.from_numpy(state.astype(np.float32)),
            "task": task or self.task,
        }
        batch = self.preprocessor(obs)
        with torch.no_grad():
            action = self.policy.select_action(batch)
            action = self.postprocessor(action)
        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy()
        else:
            action_np = np.asarray(action)
        action_np = action_np.reshape(-1).astype(np.float32)
        self.n_predictions += 1
        return action_np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--device", default="cuda")
    p.add_argument("--task", default="")
    return p.parse_args()


def make_handler(runtime: PolicyRuntime):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send_json(self, status: int, payload: dict):
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path != "/health":
                self._send_json(404, {"error": "not found"})
                return
            self._send_json(
                200,
                {
                    "ok": True,
                    "policy_type": runtime.cfg.type,
                    "checkpoint": str(runtime.checkpoint),
                    "device": str(runtime.device),
                    "n_predictions": runtime.n_predictions,
                },
            )

        def do_POST(self):  # noqa: N802
            try:
                if self.path == "/reset":
                    runtime.reset()
                    self._send_json(200, {"ok": True})
                    return
                if self.path != "/predict":
                    self._send_json(404, {"error": "not found"})
                    return

                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                payload = np.load(io.BytesIO(body), allow_pickle=False)
                image = payload["image"]
                state = payload["state"]
                task = None
                if "task" in payload.files:
                    task = str(payload["task"].item())
                action = runtime.predict(image, state, task=task)
                self._send_json(200, {"action": action.tolist()})
            except Exception as exc:  # Keep errors visible to the rollout client.
                self._send_json(500, {"error": repr(exc)})

        def log_message(self, fmt, *args):
            return

    return Handler


def main() -> None:
    args = parse_args()
    runtime = PolicyRuntime(args.checkpoint.resolve(), args.device, args.task)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runtime))
    print(
        f"policy_server ready type={runtime.cfg.type} checkpoint={runtime.checkpoint} "
        f"device={runtime.device} http://{args.host}:{args.port}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
