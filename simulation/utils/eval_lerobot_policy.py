#!/usr/bin/env python
"""Evaluate a host-side LeRobot policy server in SoftGym.

This script runs inside the SoftGym Docker/Python 3.6 environment. It sends
observations to baselines/policy_server.py on the host and receives 8D actions.
"""

from __future__ import print_function

import argparse
import csv
import io
import json
import os
import sys
import time

import cv2
import gym
import numpy as np

try:
    from urllib import request as urllib_request
except ImportError:  # pragma: no cover - Python 2 fallback, not expected here.
    import urllib2 as urllib_request


SIM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if SIM_ROOT not in sys.path:
    sys.path.insert(0, SIM_ROOT)

from softgym.registered_env import SOFTGYM_ENVS, env_arg_dict  # noqa: E402
import pyflex  # noqa: E402

from utils.collect_trajectories import (  # noqa: E402
    extract_compact_state,
    extract_proprio,
    render_rgb_depth,
    seed_everything,
)


TASK_INSTRUCTIONS = {
    "RopeFlatten": "straighten the rope",
    "ClothFlatten": "flatten the cloth",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-name", choices=sorted(SOFTGYM_ENVS.keys()), required=True)
    p.add_argument("--policy-url", required=True, help="Host policy server URL, e.g. http://172.17.0.1:8765")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--horizon", type=int, default=75)
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--camera-width", type=int, default=720)
    p.add_argument("--camera-height", type=int, default=720)
    p.add_argument("--num-variations", type=int, default=50)
    p.add_argument("--headless", type=int, default=1)
    p.add_argument("--render", type=int, default=1)
    p.add_argument("--render-mode", default=None)
    p.add_argument("--action-repeat", type=int, default=None)
    p.add_argument("--use-cached-states", action="store_true")
    p.add_argument("--save-cached-states", action="store_true")
    p.add_argument("--eval-split", action="store_true")
    p.add_argument("--success-threshold", type=float, default=0.8)
    p.add_argument("--save-every-video", type=int, default=1)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--task", default=None)
    p.add_argument("--policy-name", default="lerobot_policy")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--timeout-s", type=float, default=30.0)
    p.add_argument("--fixed-config-ids", action="store_true", help="Evaluate config IDs config_start..config_start+n-1 instead of random resets.")
    p.add_argument("--config-start", type=int, default=0)
    return p.parse_args()


def build_env_kwargs(args):
    kwargs = dict(env_arg_dict[args.env_name])
    kwargs["headless"] = bool(args.headless)
    kwargs["render"] = bool(args.render)
    kwargs["num_variations"] = args.num_variations
    kwargs["use_cached_states"] = bool(args.use_cached_states)
    kwargs["save_cached_states"] = bool(args.save_cached_states)
    kwargs["camera_width"] = args.camera_width
    kwargs["camera_height"] = args.camera_height
    kwargs["horizon"] = args.horizon
    if args.action_repeat is not None:
        kwargs["action_repeat"] = args.action_repeat
    if args.render_mode is not None:
        kwargs["render_mode"] = args.render_mode
    return kwargs


def post_npz(url, arrays, timeout_s):
    bio = io.BytesIO()
    np.savez_compressed(bio, **arrays)
    data = bio.getvalue()
    req = urllib_request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/octet-stream"},
    )
    resp = urllib_request.urlopen(req, timeout=timeout_s)
    body = resp.read().decode("utf-8")
    return json.loads(body)


def post_reset(base_url, timeout_s):
    req = urllib_request.Request(base_url.rstrip("/") + "/reset", data=b"{}", headers={"Content-Type": "application/json"})
    resp = urllib_request.urlopen(req, timeout=timeout_s)
    resp.read()


def get_action(args, image, obs_state, task):
    payload = {
        "image": image.astype(np.uint8),
        "state": obs_state.astype(np.float32),
        "task": np.array(task),
    }
    start = time.time()
    result = post_npz(args.policy_url.rstrip("/") + "/predict", payload, args.timeout_s)
    roundtrip_ms = (time.time() - start) * 1000.0
    if "error" in result:
        raise RuntimeError("policy server error: {}".format(result["error"]))
    action = np.asarray(result["action"], dtype=np.float32)
    inference_ms = float(result.get("inference_latency_ms", np.nan))
    chunk_generated = result.get("action_chunk_generated", None)
    return action, inference_ms, float(roundtrip_ms), chunk_generated


def clip_action(action, action_space):
    action = action.astype(np.float32)
    if isinstance(action_space, gym.spaces.Box):
        return np.clip(action, action_space.low, action_space.high).astype(np.float32)
    return action


def save_video(path, frames, fps):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
    writer.release()


def summarize(rows):
    if not rows:
        return {}
    final_norm = np.asarray([r["final_normalized_performance"] for r in rows], dtype=np.float32)
    successes = np.asarray([r["success"] for r in rows], dtype=np.float32)
    summary = {
        "n_episodes": int(len(rows)),
        "mean_final_normalized_performance": float(np.mean(final_norm)),
        "std_final_normalized_performance": float(np.std(final_norm)),
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean([r["return"] for r in rows])),
    }
    if "final_end_point_distance" in rows[0]:
        vals = [r.get("final_end_point_distance", np.nan) for r in rows]
        summary["mean_final_end_point_distance"] = float(np.nanmean(vals))
    for prefix in ("policy_inference_latency_ms", "policy_roundtrip_latency_ms"):
        vals = [float(r.get(prefix + "_mean", np.nan)) for r in rows]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            summary[prefix + "_episode_mean"] = float(np.mean(np.asarray(vals, dtype=np.float32)))
    all_infer = []
    all_roundtrip = []
    for row in rows:
        all_infer.extend(row.get("_all_inference_latencies", []))
        all_roundtrip.extend(row.get("_all_roundtrip_latencies", []))
    all_chunk_infer = []
    all_cached_action = []
    all_chunk_roundtrip = []
    all_cached_roundtrip = []
    for row in rows:
        all_chunk_infer.extend(row.get("_all_chunk_inference_latencies", []))
        all_cached_action.extend(row.get("_all_cached_action_latencies", []))
        all_chunk_roundtrip.extend(row.get("_all_chunk_roundtrip_latencies", []))
        all_cached_roundtrip.extend(row.get("_all_cached_roundtrip_latencies", []))
    for name, vals in (("policy_inference_latency_ms", all_infer), ("policy_roundtrip_latency_ms", all_roundtrip), ("policy_chunk_inference_latency_ms", all_chunk_infer), ("policy_cached_action_latency_ms", all_cached_action), ("policy_chunk_roundtrip_latency_ms", all_chunk_roundtrip), ("policy_cached_roundtrip_latency_ms", all_cached_roundtrip)):
        if vals:
            arr = np.asarray(vals, dtype=np.float32)
            summary[name + "_mean"] = float(np.mean(arr))
            summary[name + "_p50"] = float(np.percentile(arr, 50))
            summary[name + "_p90"] = float(np.percentile(arr, 90))
            summary[name + "_p95"] = float(np.percentile(arr, 95))
            summary[name + "_max"] = float(np.max(arr))
    return summary


def main():
    args = parse_args()
    seed_everything(args.seed)
    task = args.task or TASK_INSTRUCTIONS.get(args.env_name, args.env_name)
    out_dir = os.path.abspath(args.output_dir)
    video_dir = os.path.join(out_dir, "videos")
    if not os.path.isdir(video_dir):
        os.makedirs(video_dir)

    env_kwargs = build_env_kwargs(args)
    env = SOFTGYM_ENVS[args.env_name](**env_kwargs)
    if args.eval_split:
        env.eval_flag = True

    rows = []
    try:
        for ep in range(args.num_episodes):
            post_reset(args.policy_url, args.timeout_s)
            if args.fixed_config_ids:
                cfg_id = (args.config_start + ep) % len(env.cached_configs)
                cfg = env.cached_configs[cfg_id] if getattr(env, "cached_configs", None) else None
                init = env.cached_init_states[cfg_id] if getattr(env, "cached_init_states", None) else None
                env.reset(config=cfg, config_id=cfg_id, initial_state=init)
            else:
                env.reset()
            num_picker = int(env_kwargs.get("num_picker", getattr(env.action_tool, "num_picker", 2)))
            frames = []
            rewards = []
            inference_latencies = []
            roundtrip_latencies = []
            chunk_inference_latencies = []
            cached_action_latencies = []
            chunk_roundtrip_latencies = []
            cached_roundtrip_latencies = []
            info = env._get_info()
            start_norm = float(info.get("normalized_performance", np.nan))

            # rollout loop: render, ask host policy, then step softgym
            for step in range(args.horizon):
                image, _ = render_rgb_depth(env, args.img_size, False)
                if args.save_every_video > 0 and ep % args.save_every_video == 0:
                    frames.append(image.copy())
                obs_state = np.concatenate(
                    [extract_proprio(env, num_picker), extract_compact_state(num_picker)],
                    axis=0,
                ).astype(np.float32)
                action_raw, inference_ms, roundtrip_ms, chunk_generated = get_action(args, image, obs_state, task)
                inference_latencies.append(inference_ms)
                roundtrip_latencies.append(roundtrip_ms)
                if chunk_generated is True:
                    # chunk calls are the fair latency number for action policies
                    chunk_inference_latencies.append(inference_ms)
                    chunk_roundtrip_latencies.append(roundtrip_ms)
                elif chunk_generated is False:
                    cached_action_latencies.append(inference_ms)
                    cached_roundtrip_latencies.append(roundtrip_ms)
                action = clip_action(action_raw, env.action_space)
                _, reward, done, info = env.step(action)
                rewards.append(float(reward))
                if done:
                    break

            final_image, _ = render_rgb_depth(env, args.img_size, False)
            if args.save_every_video > 0 and ep % args.save_every_video == 0:
                frames.append(final_image.copy())
                save_video(os.path.join(video_dir, "episode_{:03d}.mp4".format(ep)), frames, args.fps)

            final_norm = float(info.get("normalized_performance", np.nan))
            row = {
                "episode": ep,
                "config_id": (args.config_start + ep) % len(env.cached_configs) if args.fixed_config_ids else "",
                "env": args.env_name,
                "policy": args.policy_name,
                "checkpoint": args.checkpoint,
                "task": task,
                "steps": step + 1,
                "return": float(np.sum(rewards)),
                "start_normalized_performance": start_norm,
                "final_normalized_performance": final_norm,
                "final_performance": float(info.get("performance", np.nan)),
                "success": bool(final_norm >= args.success_threshold),
                "video_path": os.path.join(video_dir, "episode_{:03d}.mp4".format(ep))
                if args.save_every_video > 0 and ep % args.save_every_video == 0
                else "",
                "policy_inference_latency_ms_mean": float(np.nanmean(inference_latencies)) if inference_latencies else np.nan,
                "policy_inference_latency_ms_p90": float(np.nanpercentile(inference_latencies, 90)) if inference_latencies else np.nan,
                "policy_roundtrip_latency_ms_mean": float(np.nanmean(roundtrip_latencies)) if roundtrip_latencies else np.nan,
                "policy_roundtrip_latency_ms_p90": float(np.nanpercentile(roundtrip_latencies, 90)) if roundtrip_latencies else np.nan,
                "policy_chunk_inference_latency_ms_mean": float(np.nanmean(chunk_inference_latencies)) if chunk_inference_latencies else np.nan,
                "policy_chunk_inference_latency_ms_p90": float(np.nanpercentile(chunk_inference_latencies, 90)) if chunk_inference_latencies else np.nan,
                "policy_cached_action_latency_ms_mean": float(np.nanmean(cached_action_latencies)) if cached_action_latencies else np.nan,
                "policy_cached_action_latency_ms_p90": float(np.nanpercentile(cached_action_latencies, 90)) if cached_action_latencies else np.nan,
                "policy_chunk_roundtrip_latency_ms_mean": float(np.nanmean(chunk_roundtrip_latencies)) if chunk_roundtrip_latencies else np.nan,
                "policy_chunk_roundtrip_latency_ms_p90": float(np.nanpercentile(chunk_roundtrip_latencies, 90)) if chunk_roundtrip_latencies else np.nan,
                "policy_cached_roundtrip_latency_ms_mean": float(np.nanmean(cached_roundtrip_latencies)) if cached_roundtrip_latencies else np.nan,
                "policy_cached_roundtrip_latency_ms_p90": float(np.nanpercentile(cached_roundtrip_latencies, 90)) if cached_roundtrip_latencies else np.nan,
                "n_policy_chunk_generations": int(len(chunk_inference_latencies)),
                "_all_inference_latencies": inference_latencies,
                "_all_chunk_inference_latencies": chunk_inference_latencies,
                "_all_cached_action_latencies": cached_action_latencies,
                "_all_chunk_roundtrip_latencies": chunk_roundtrip_latencies,
                "_all_cached_roundtrip_latencies": cached_roundtrip_latencies,
                "_all_roundtrip_latencies": roundtrip_latencies,
            }
            if "end_point_distance" in info:
                row["final_end_point_distance"] = float(info["end_point_distance"])
            rows.append(row)
            print(
                "episode {} final_norm={:.4f} return={:.4f} success={}".format(
                    ep, row["final_normalized_performance"], row["return"], row["success"]
                )
            )

    finally:
        env.close()

    csv_rows = []
    # raw latency arrays go in summary stats, not the per-episode csv
    for row in rows:
        row = dict(row)
        row.pop("_all_inference_latencies", None)
        row.pop("_all_roundtrip_latencies", None)
        csv_rows.append(row)
    keys = sorted(set().union(*[r.keys() for r in csv_rows]))
    scores_path = os.path.join(out_dir, "scores.csv")
    with open(scores_path, "w") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    summary = summarize(rows)
    summary.update(
        {
            "env": args.env_name,
            "policy": args.policy_name,
            "checkpoint": args.checkpoint,
            "success_threshold": args.success_threshold,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
