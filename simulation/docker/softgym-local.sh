#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
IMAGE=${SOFTGYM_LOCAL_IMAGE:-softgym-local:py36-cuda92}
CONTAINER_ROOT=/workspace/softgym
SOFTGYM_HEADLESS=${SOFTGYM_HEADLESS:-1}
SOFTGYM_SKIP_PREFLIGHT=${SOFTGYM_SKIP_PREFLIGHT:-0}
SOFTGYM_USE_WSLG_GL=${SOFTGYM_USE_WSLG_GL:-0}
SOFTGYM_SOFTWARE_GL=${SOFTGYM_SOFTWARE_GL:-1}

DOCKER_REPO_ROOT=$REPO_ROOT
DOCKERFILE_LOCAL="$REPO_ROOT/docker/Dockerfile.local"
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    # Git Bash rewrites Linux-looking Docker args like /workspace/softgym
    # unless path conversion is disabled. Convert only the host paths.
    export MSYS_NO_PATHCONV=1
    DOCKER_REPO_ROOT=$(cygpath -w "$REPO_ROOT")
    DOCKERFILE_LOCAL=$(cygpath -w "$DOCKERFILE_LOCAL")
    ;;
esac

usage() {
  cat <<'USAGE'
Usage:
  docker/softgym-local.sh build
  docker/softgym-local.sh compile
  docker/softgym-local.sh doctor
  docker/softgym-local.sh example [EnvName]
  docker/softgym-local.sh run "python envs/random_env.py --env_name ClothDrop --headless 1"
  docker/softgym-local.sh shell

Default example environment: ClothDrop
Set SOFTGYM_HEADLESS=0 to render through WSLg/X11 instead of headless EGL.
USAGE
}

docker_common_args() {
  local args=(
    --rm
    --gpus all
    -e NVIDIA_VISIBLE_DEVICES=all
    -e NVIDIA_DRIVER_CAPABILITIES=all
    --user "$(id -u):$(id -g)"
    -e HOME=/tmp
    -e "SOFTGYM_PICKER_COLOR=${SOFTGYM_PICKER_COLOR:-}"
    -e "SOFTGYM_HORIZON=${SOFTGYM_HORIZON:-}"
    -v "$DOCKER_REPO_ROOT:$CONTAINER_ROOT"
    -w "$CONTAINER_ROOT"
  )

  # Keep STDIN open when SOFTGYM_STDIN=1 — required for the env_server IPC
  # bridge (eval/env_client.py pipes JSON commands to the container's stdin).
  # Harmless to omit for non-interactive uses (collect, compile, benchmark).
  if [[ "${SOFTGYM_STDIN:-0}" == "1" ]]; then
    args+=(-i)
  fi

  if [[ -n "${DISPLAY:-}" && -d /mnt/wslg/.X11-unix ]]; then
    args+=(
      -v /mnt/wslg/.X11-unix:/tmp/.X11-unix
      -e "DISPLAY=$DISPLAY"
      -e QT_X11_NO_MITSHM=1
      -e SDL_VIDEODRIVER=x11
    )
  elif [[ -n "${DISPLAY:-}" && -d /tmp/.X11-unix ]]; then
    args+=(
      -v /tmp/.X11-unix:/tmp/.X11-unix
      -e "DISPLAY=$DISPLAY"
      -e QT_X11_NO_MITSHM=1
      -e SDL_VIDEODRIVER=x11
    )
  fi

  if [[ "$SOFTGYM_SOFTWARE_GL" == "1" ]]; then
    args+=(
      -e LIBGL_ALWAYS_SOFTWARE=1
      -e MESA_LOADER_DRIVER_OVERRIDE=swrast
      -e MESA_GL_VERSION_OVERRIDE=3.3
      -e SOFTGYM_DISABLE_INTEROP=1
    )
  fi

  if [[ -e /dev/dxg ]]; then
    args+=(--device /dev/dxg)
  fi

  if [[ -d /mnt/wslg ]]; then
    args+=(
      -v /mnt/wslg:/mnt/wslg
      -e "WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-}"
      -e XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir
      -e "PULSE_SERVER=${PULSE_SERVER:-}"
    )
  fi

  local wslg_gl_path=/mnt/wslg/distro/usr/lib/x86_64-linux-gnu
  if [[ "$SOFTGYM_USE_WSLG_GL" == "1" && -d "$wslg_gl_path" ]]; then
    args+=(
      -e "LIBGL_DRIVERS_PATH=$wslg_gl_path/dri"
      -e "__EGL_VENDOR_LIBRARY_DIRS=$wslg_gl_path/glvnd/egl_vendor.d"
    )
  fi

  if [[ -d /usr/lib/wsl ]]; then
    args+=(
      -v /usr/lib/wsl:/usr/lib/wsl:ro
      -e "LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
    )
  fi

  printf '%s\n' "${args[@]}"
}

pyflex_env='export PYFLEXROOT=/workspace/softgym/PyFlex; export PYTHONPATH=/workspace/softgym:/workspace/softgym/PyFlex/bindings/build:${PYTHONPATH:-}; export LD_LIBRARY_PATH=/workspace/softgym/PyFlex/external/SDL2-2.0.4/lib/x64:${LD_LIBRARY_PATH:-}'

build_image() {
  docker build -f "$DOCKERFILE_LOCAL" -t "$IMAGE" "$DOCKER_REPO_ROOT"
}

ensure_image() {
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    build_image
  fi
}

run_container() {
  mapfile -t args < <(docker_common_args)
  docker run "${args[@]}" "$IMAGE" "$@"
}

compile_pyflex() {
  ensure_image
  run_container bash -lc "$pyflex_env; mkdir -p PyFlex/bindings/build && cd PyFlex/bindings/build && cmake -DPYBIND11_PYTHON_VERSION=3.6 .. && make -j\$(nproc)"
}

doctor() {
  ensure_image
  run_container bash -lc '
    set -x
    id
    env | grep -E "^(DISPLAY|WAYLAND_DISPLAY|XDG_RUNTIME_DIR|LD_LIBRARY_PATH|NVIDIA_|CUDA|LIBGL|MESA_|SOFTGYM_)" || true
    ls -la /dev/dxg /mnt/wslg /mnt/wslg/runtime-dir /tmp/.X11-unix /usr/lib/wsl/lib 2>/dev/null || true
    command -v nvidia-smi >/dev/null && nvidia-smi || true
    ldconfig -p | grep -E "libGL|libEGL|libGLX|libcuda|d3d12|dxcore|nvidia" | head -120 || true
    python - <<'"'"'PY'"'"'
import os
print("python ok")
for key in ("DISPLAY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR", "LD_LIBRARY_PATH"):
    print("{}={}".format(key, os.environ.get(key)))
PY
  '
}

check_headless_egl() {
  if [[ "$SOFTGYM_HEADLESS" != "1" || "$SOFTGYM_SKIP_PREFLIGHT" == "1" ]]; then
    return
  fi

  if ! run_container bash -lc "ldconfig -p | grep -q 'libEGL_nvidia'"; then
    cat >&2 <<'EOF'
Headless EGL is not available inside the SoftGym container.

CUDA is not enough for SoftGym video rendering: PyFlex also needs an EGL
OpenGL context. On WSL, try rendering through WSLg instead:

  SOFTGYM_HEADLESS=0 docker/softgym-local.sh example RopeFlatten

If you are on native Linux, install/configure NVIDIA Container Toolkit and make
sure Docker exposes NVIDIA graphics capabilities.
EOF
    exit 1
  fi
}

run_example() {
  local env_name=${1:-ClothDrop}
  if [[ ! "$env_name" =~ ^[A-Za-z0-9_]+$ ]]; then
    echo "Invalid environment name: $env_name" >&2
    exit 2
  fi

  ensure_image
  if ! compgen -G "$REPO_ROOT/PyFlex/bindings/build/pyflex*.so" >/dev/null; then
    compile_pyflex
  fi
  check_headless_egl

  mkdir -p "$REPO_ROOT/data"
  local quoted_env
  quoted_env=$(printf '%q' "$env_name")
  run_container bash -lc "$pyflex_env; python envs/random_env.py --env_name $quoted_env --headless $SOFTGYM_HEADLESS --num_variations 1 --save_video_dir ./data"
}

case "${1:-}" in
  build)
    build_image
    ;;
  compile)
    compile_pyflex
    ;;
  doctor)
    doctor
    ;;
  example)
    shift
    run_example "${1:-ClothDrop}"
    ;;
  run)
    shift
    if [[ $# -eq 0 ]]; then
      usage
      exit 2
    fi
    ensure_image
    run_container bash -lc "$pyflex_env; $*"
    ;;
  shell)
    ensure_image
    mapfile -t args < <(docker_common_args)
    docker run -it "${args[@]}" "$IMAGE" bash -lc "$pyflex_env; exec bash"
    ;;
  *)
    usage
    exit 2
    ;;
esac
