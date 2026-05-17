#!/usr/bin/env python
"""Generate a static HTML viewer for collected SoftGym NPZ trajectories."""

from __future__ import print_function

import argparse
import glob
import json
import os
import sys

import numpy as np

try:
    import imageio.v2 as imageio
except ImportError:  # imageio<2.9
    import imageio


SIM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_DATA_DIR = os.path.join(SIM_ROOT, "data", "trajectories")
DEFAULT_VIEWER_DIR = os.path.join(SIM_ROOT, "data", "viewer")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a browser-viewable HTML viewer for a trajectory NPZ."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to a trajectory .npz. Defaults to the most recently modified trajectory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Viewer output directory. Defaults to data/viewer/<npz-name>/.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Export every Nth frame.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on exported frames after applying stride.",
    )
    parser.add_argument(
        "--image-format",
        choices=("jpg", "png"),
        default="jpg",
        help="Frame image format.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality when --image-format jpg.",
    )
    parser.add_argument(
        "--depth-percentile",
        type=float,
        default=99.0,
        help="Upper percentile for depth visualization normalization.",
    )
    return parser.parse_args()


def latest_npz():
    paths = glob.glob(os.path.join(DEFAULT_DATA_DIR, "*.npz"))
    if not paths:
        raise RuntimeError("No trajectory NPZ files found in {}".format(DEFAULT_DATA_DIR))
    return max(paths, key=os.path.getmtime)


def resolve_path(path):
    if path is None:
        return latest_npz()
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(SIM_ROOT, path))


def default_output_dir(npz_path):
    name = os.path.splitext(os.path.basename(npz_path))[0]
    return os.path.join(DEFAULT_VIEWER_DIR, name)


def scalar_to_python(value):
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        if isinstance(item, np.generic):
            return item.item()
        return item
    return None


def array_summary(data):
    summary = {}
    for key in data.files:
        arr = data[key]
        entry = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }
        scalar = scalar_to_python(arr)
        if scalar is not None and key != "metadata_json":
            entry["value"] = scalar
        summary[key] = entry
    return summary


def load_metadata(data):
    if "metadata_json" not in data.files:
        return {}
    raw = scalar_to_python(data["metadata_json"])
    if raw is None:
        raw = str(data["metadata_json"])
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return {"raw_metadata_json": str(raw)}


def select_indices(frame_count, stride, max_frames):
    if stride < 1:
        raise ValueError("--stride must be >= 1")
    indices = list(range(0, frame_count, stride))
    if frame_count > 0 and indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    if max_frames is not None:
        indices = indices[:max_frames]
    return indices


def ensure_rgb_uint8(frames):
    frames = np.asarray(frames)
    if frames.dtype == np.uint8:
        return frames
    frames = np.clip(frames, 0, 255)
    return frames.astype(np.uint8)


def save_rgb_frame(path, frame, image_format, jpeg_quality):
    if image_format == "jpg":
        imageio.imwrite(path, frame, quality=jpeg_quality)
    else:
        imageio.imwrite(path, frame)


def normalize_depth(depth_frames, percentile):
    depth = np.asarray(depth_frames, dtype=np.float32)
    finite = np.isfinite(depth)
    positive = finite & (depth > 0)
    valid = depth[positive]
    if valid.size == 0:
        valid = depth[finite]
    if valid.size == 0:
        return np.zeros_like(depth, dtype=np.uint8)

    lo = float(np.percentile(valid, 1.0))
    hi = float(np.percentile(valid, percentile))
    if hi <= lo:
        hi = lo + 1e-6
    norm = (depth - lo) / (hi - lo)
    norm[~finite] = 0
    norm = np.clip(norm, 0, 1)
    return (norm * 255).astype(np.uint8)


def export_frames(data, output_dir, indices, image_format, jpeg_quality, depth_percentile):
    rgb = ensure_rgb_uint8(data["rgb"])
    frame_dir = os.path.join(output_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)

    rgb_paths = []
    depth_paths = []
    for out_idx, frame_idx in enumerate(indices):
        rel_path = os.path.join("frames", "rgb_{:05d}.{}".format(out_idx, image_format))
        save_rgb_frame(
            os.path.join(output_dir, rel_path),
            rgb[frame_idx],
            image_format,
            jpeg_quality,
        )
        rgb_paths.append(rel_path)

    if "depth" in data.files:
        depth_vis = normalize_depth(data["depth"], depth_percentile)
        for out_idx, frame_idx in enumerate(indices):
            rel_path = os.path.join("frames", "depth_{:05d}.png".format(out_idx))
            imageio.imwrite(os.path.join(output_dir, rel_path), depth_vis[frame_idx])
            depth_paths.append(rel_path)

    return rgb_paths, depth_paths


def compact_float_array(arr, decimals=5):
    arr = np.asarray(arr)
    if arr.size == 0:
        return []
    return np.round(arr.astype(np.float64), decimals=decimals).tolist()


def sampled_step_values(data, indices, key, frame_aligned=False):
    if key not in data.files:
        return []
    values = data[key]
    selected = []
    for frame_idx in indices:
        if frame_aligned:
            source_idx = frame_idx
        else:
            source_idx = frame_idx - 1
        if source_idx < 0 or source_idx >= len(values):
            selected.append(None)
            continue
        value = values[source_idx]
        arr = np.asarray(value)
        if arr.shape == ():
            item = arr.item()
            if isinstance(item, np.generic):
                item = item.item()
            selected.append(item)
        else:
            selected.append(compact_float_array(arr))
    return selected


def viewer_payload(data, indices, rgb_paths, depth_paths):
    info_keys = sorted(
        key for key in data.files
        if key.startswith("info_") and np.asarray(data[key]).ndim <= 1
    )
    payload = {
        "frame_indices": indices,
        "rgb_paths": rgb_paths,
        "depth_paths": depth_paths,
        "action_raw": sampled_step_values(data, indices, "action_raw"),
        "action_normalized": sampled_step_values(data, indices, "action_normalized"),
        "reward": sampled_step_values(data, indices, "reward"),
        "done": sampled_step_values(data, indices, "done"),
        "info": {
            key: sampled_step_values(data, indices, key)
            for key in info_keys
        },
    }
    return payload


def write_json(path, value):
    with open(path, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_html(path, npz_path, metadata, summary, payload):
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SoftGym Trajectory Viewer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #1d232a;
      --muted: #617080;
      --line: #d9dee5;
      --accent: #0f766e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      padding: 20px 24px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 { margin: 0 0 6px; font-size: 22px; font-weight: 650; }
    .path { color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; overflow-wrap: anywhere; }
    main {
      display: grid;
      grid-template-columns: minmax(340px, 1fr) 380px;
      gap: 16px;
      padding: 16px 24px 24px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }
    .stage {
      display: grid;
      gap: 12px;
    }
    .image-wrap {
      width: 100%;
      min-height: 260px;
      display: grid;
      place-items: center;
      background: #111820;
      border-radius: 6px;
      overflow: hidden;
    }
    img {
      max-width: 100%;
      max-height: calc(100vh - 240px);
      image-rendering: auto;
      display: block;
    }
    .controls {
      display: grid;
      grid-template-columns: auto 1fr auto auto auto;
      gap: 10px;
      align-items: center;
    }
    button, select {
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
      border-radius: 6px;
      padding: 8px 10px;
      font: inherit;
    }
    button.primary {
      background: var(--accent);
      color: white;
      border-color: var(--accent);
      min-width: 72px;
    }
    input[type="range"] { width: 100%; }
    .kv {
      display: grid;
      grid-template-columns: 145px 1fr;
      gap: 7px 10px;
      font-size: 13px;
    }
    .kv dt { color: var(--muted); }
    .kv dd { margin: 0; overflow-wrap: anywhere; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    pre {
      margin: 0;
      padding: 10px;
      max-height: 280px;
      overflow: auto;
      background: #f0f3f6;
      border-radius: 6px;
      font-size: 12px;
    }
    details { margin-top: 12px; }
    summary { cursor: pointer; color: var(--muted); }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; padding: 12px; }
      header { padding: 16px 12px 10px; }
      .controls { grid-template-columns: 1fr 1fr; }
      .controls input { grid-column: 1 / -1; }
    }
  </style>
</head>
<body>
  <header>
    <h1>SoftGym Trajectory Viewer</h1>
    <div class="path" id="sourcePath"></div>
  </header>
  <main>
    <section class="panel stage">
      <div class="image-wrap">
        <img id="frameImage" alt="trajectory frame">
      </div>
      <div class="controls">
        <button id="playButton" class="primary">Play</button>
        <input id="frameSlider" type="range" min="0" step="1" value="0">
        <span id="frameLabel"></span>
        <select id="imageMode">
          <option value="rgb">RGB</option>
          <option value="depth">Depth</option>
        </select>
        <select id="fps">
          <option value="5">5 fps</option>
          <option value="10" selected>10 fps</option>
          <option value="20">20 fps</option>
        </select>
      </div>
    </section>
    <aside class="panel">
      <dl class="kv" id="stepInfo"></dl>
      <details open>
        <summary>Metadata</summary>
        <pre id="metadata"></pre>
      </details>
      <details>
        <summary>Arrays</summary>
        <pre id="summary"></pre>
      </details>
    </aside>
  </main>
  <script>
    const SOURCE_PATH = __SOURCE_PATH__;
    const METADATA = __METADATA__;
    const SUMMARY = __SUMMARY__;
    const DATA = __PAYLOAD__;

    const image = document.getElementById("frameImage");
    const slider = document.getElementById("frameSlider");
    const label = document.getElementById("frameLabel");
    const playButton = document.getElementById("playButton");
    const mode = document.getElementById("imageMode");
    const fps = document.getElementById("fps");
    const info = document.getElementById("stepInfo");

    let current = 0;
    let timer = null;

    document.getElementById("sourcePath").textContent = SOURCE_PATH;
    document.getElementById("metadata").textContent = JSON.stringify(METADATA, null, 2);
    document.getElementById("summary").textContent = JSON.stringify(SUMMARY, null, 2);
    slider.max = Math.max(DATA.rgb_paths.length - 1, 0);
    if (!DATA.depth_paths.length) {
      mode.querySelector("option[value='depth']").disabled = true;
    }

    function valueAt(values, index) {
      if (!values || index < 0 || index >= values.length) return null;
      return values[index];
    }

    function formatValue(value) {
      if (value === null || value === undefined) return "n/a";
      if (Array.isArray(value)) return "[" + value.join(", ") + "]";
      return String(value);
    }

    function addRow(name, value) {
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = name;
      dd.textContent = formatValue(value);
      info.appendChild(dt);
      info.appendChild(dd);
    }

    function render(index) {
      current = Math.max(0, Math.min(index, DATA.rgb_paths.length - 1));
      slider.value = current;
      label.textContent = `${current + 1} / ${DATA.rgb_paths.length}`;
      const paths = mode.value === "depth" && DATA.depth_paths.length ? DATA.depth_paths : DATA.rgb_paths;
      image.src = paths[current];

      info.replaceChildren();
      addRow("frame_index", valueAt(DATA.frame_indices, current));
      addRow("transition", current === 0 ? "initial" : current - 1);
      addRow("reward", valueAt(DATA.reward, current));
      addRow("done", valueAt(DATA.done, current));
      addRow("action_raw", valueAt(DATA.action_raw, current));
      addRow("action_normalized", valueAt(DATA.action_normalized, current));
      Object.keys(DATA.info).forEach((key) => addRow(key, valueAt(DATA.info[key], current)));
    }

    function stop() {
      if (timer !== null) clearInterval(timer);
      timer = null;
      playButton.textContent = "Play";
    }

    function play() {
      stop();
      playButton.textContent = "Pause";
      timer = setInterval(() => {
        if (current >= DATA.rgb_paths.length - 1) {
          stop();
          return;
        }
        render(current + 1);
      }, 1000 / Number(fps.value));
    }

    slider.addEventListener("input", () => render(Number(slider.value)));
    mode.addEventListener("change", () => render(current));
    fps.addEventListener("change", () => {
      if (timer !== null) play();
    });
    playButton.addEventListener("click", () => {
      if (timer === null) play();
      else stop();
    });
    window.addEventListener("keydown", (event) => {
      if (event.key === "ArrowRight") render(current + 1);
      if (event.key === "ArrowLeft") render(current - 1);
      if (event.key === " ") {
        event.preventDefault();
        if (timer === null) play();
        else stop();
      }
    });

    render(0);
  </script>
</body>
</html>
"""
    html = html.replace("__SOURCE_PATH__", json.dumps(npz_path))
    html = html.replace("__METADATA__", json.dumps(metadata, sort_keys=True))
    html = html.replace("__SUMMARY__", json.dumps(summary, sort_keys=True))
    html = html.replace("__PAYLOAD__", json.dumps(payload, sort_keys=True))
    with open(path, "w") as handle:
        handle.write(html)


def main():
    args = parse_args()
    npz_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir) if args.output_dir else default_output_dir(npz_path)
    os.makedirs(output_dir, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)
    if "rgb" not in data.files:
        raise RuntimeError("{} does not contain an 'rgb' array".format(npz_path))

    indices = select_indices(len(data["rgb"]), args.stride, args.max_frames)
    rgb_paths, depth_paths = export_frames(
        data,
        output_dir,
        indices,
        args.image_format,
        args.jpeg_quality,
        args.depth_percentile,
    )

    summary = array_summary(data)
    metadata = load_metadata(data)
    payload = viewer_payload(data, indices, rgb_paths, depth_paths)
    write_json(os.path.join(output_dir, "summary.json"), {
        "source": npz_path,
        "metadata": metadata,
        "arrays": summary,
        "exported_frame_indices": indices,
    })
    write_html(
        os.path.join(output_dir, "index.html"),
        npz_path,
        metadata,
        summary,
        payload,
    )

    print("source:", npz_path)
    print("frames:", len(data["rgb"]), "exported:", len(indices))
    print("viewer:", os.path.join(output_dir, "index.html"))
    print("summary:", os.path.join(output_dir, "summary.json"))


if __name__ == "__main__":
    main()
