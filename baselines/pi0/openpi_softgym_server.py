#!/usr/bin/env python3
"""OpenPI HTTP policy server compatible with the SoftGym rollout client.

It exposes the same small NPZ-over-HTTP API as `baselines/policy_server.py`:

  GET  /health
  POST /reset
  POST /predict   body: npz with image uint8 HWC, state float32, optional task

The output is always an 8D SoftGym picker action plus timing metadata.
"""

from __future__ import annotations

import argparse
import traceback
import io
import json
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as openpi_config


SOFTGYM_ACTION_DIM = 8


def _resize_image(image_hwc: np.ndarray, size: int = 224) -> np.ndarray:
    if image_hwc.shape[:2] == (size, size):
        return image_hwc.astype(np.uint8)
    try:
        import cv2

        return cv2.resize(image_hwc.astype(np.uint8), (size, size), interpolation=cv2.INTER_AREA)
    except Exception:
        # Nearest-neighbor fallback that avoids adding a hard cv2 dependency to
        # the host OpenPI environment.
        y = np.linspace(0, image_hwc.shape[0] - 1, size).astype(np.int64)
        x = np.linspace(0, image_hwc.shape[1] - 1, size).astype(np.int64)
        return image_hwc[y][:, x].astype(np.uint8)


def _softgym_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if action.ndim == 1:
        vec = action
    else:
        vec = action.reshape(-1, action.shape[-1])[0]
    out = np.zeros((SOFTGYM_ACTION_DIM,), dtype=np.float32)
    n = min(SOFTGYM_ACTION_DIM, vec.shape[0])
    out[:n] = vec[:n]
    out[[0, 1, 2, 4, 5, 6]] = np.clip(out[[0, 1, 2, 4, 5, 6]], -0.01, 0.01)
    out[[3, 7]] = np.clip(out[[3, 7]], 0.0, 1.0)
    return out


class OpenPIRuntime:
    def __init__(self, config_name: str, checkpoint: str, task: str):
        self.config_name = config_name
        self.task = task
        self.cfg = openpi_config.get_config(config_name)
        checkpoint = checkpoint or "gs://openpi-assets/checkpoints/{}".format(config_name)
        self.checkpoint_dir = download.maybe_download(checkpoint)
        self.policy = policy_config.create_trained_policy(self.cfg, self.checkpoint_dir)
        self.queue = deque()
        self.n_predictions = 0
        self.latency_ms = []

    def reset(self):
        self.queue.clear()

    def _example(self, image_hwc: np.ndarray, state: np.ndarray, task: str | None):
        image = _resize_image(image_hwc)
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        return {
            "observation/images/front": image,
            "observation/state": state,
            "prompt": task or self.task,
        }

    def predict(self, image_hwc: np.ndarray, state: np.ndarray, task: str | None):
        if image_hwc.ndim != 3 or image_hwc.shape[-1] != 3:
            raise ValueError("expected image HWC RGB, got {}".format(image_hwc.shape))
        if state.shape != (23,):
            raise ValueError("expected 23D SoftGym state, got {}".format(state.shape))

        generated = False
        if not self.queue:
            start = time.perf_counter()
            out = self.policy.infer(self._example(image_hwc, state, task))
            latency_ms = (time.perf_counter() - start) * 1000.0
            actions = np.asarray(out["actions"], dtype=np.float32)
            for action in actions.reshape(-1, actions.shape[-1]):
                self.queue.append(_softgym_action(action))
            generated = True
        else:
            start = time.perf_counter()
            latency_ms = (time.perf_counter() - start) * 1000.0

        action = self.queue.popleft()
        self.n_predictions += 1
        self.latency_ms.append(float(latency_ms))
        return action, {
            "inference_latency_ms": float(latency_ms),
            "action_chunk_generated": generated,
            "action_queue_len_after": len(self.queue),
            "n_action_steps": len(self.queue) + 1 if generated else None,
        }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-name", default="pi05_droid")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--task", default="straighten the rope")
    return p.parse_args()


def make_handler(runtime: OpenPIRuntime):
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
            self._send_json(200, {
                "ok": True,
                "policy_type": "openpi",
                "config_name": runtime.config_name,
                "checkpoint_dir": str(runtime.checkpoint_dir),
                "n_predictions": runtime.n_predictions,
                "mean_inference_latency_ms": float(np.mean(runtime.latency_ms)) if runtime.latency_ms else None,
                "p90_inference_latency_ms": float(np.percentile(runtime.latency_ms, 90)) if runtime.latency_ms else None,
            })

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
                payload = np.load(io.BytesIO(self.rfile.read(length)), allow_pickle=False)
                task = str(payload["task"].item()) if "task" in payload.files else None
                action, meta = runtime.predict(payload["image"], payload["state"], task)
                self._send_json(200, {"action": action.tolist(), **meta})
            except Exception as exc:
                traceback.print_exc()
                self._send_json(500, {"error": repr(exc)})

        def log_message(self, fmt, *args):
            return

    return Handler


def main():
    args = parse_args()
    runtime = OpenPIRuntime(args.config_name, args.checkpoint, args.task)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runtime))
    print(
        "openpi_softgym_server ready config={} checkpoint={} http://{}:{}".format(
            args.config_name, runtime.checkpoint_dir, args.host, args.port
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
