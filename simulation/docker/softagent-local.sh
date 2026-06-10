#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
BASE_IMAGE=${SOFTGYM_LOCAL_IMAGE:-softgym-local:py36-cuda92}
IMAGE=${SOFTAGENT_LOCAL_IMAGE:-softgym-softagent:py36-cuda92}
CONTAINER_ROOT=/workspace/softgym
SOFTAGENT_ROOT=$CONTAINER_ROOT/softagent
SOFTGYM_USE_WSLG_GL=${SOFTGYM_USE_WSLG_GL:-0}
SOFTGYM_SOFTWARE_GL=${SOFTGYM_SOFTWARE_GL:-1}

usage() {
  cat <<'USAGE'
Usage:
  docker/softagent-local.sh build
  docker/softagent-local.sh compile
  docker/softagent-local.sh shell
  docker/softagent-local.sh run "python experiments/run_softgym_benchmark.py --algorithm cem --env-name ClothFlatten"
  docker/softagent-local.sh benchmark [algorithm] [extra args...]
  docker/softagent-local.sh cem [extra args...]
  docker/softagent-local.sh curl [extra args...]
  docker/softagent-local.sh drq [extra args...]
  docker/softagent-local.sh planet [extra args...]
  docker/softagent-local.sh mvp [extra args...]

Examples:
  docker/softagent-local.sh cem --env-name ClothFlatten --test-episodes 1 --max-iters 2 --timestep-per-decision 200
  docker/softagent-local.sh drq --env-name ClothFlatten --num-train-steps 10000 --num-seed-steps 1000

Environment variables:
  SOFTGYM_LOCAL_IMAGE   Base image tag, default softgym-local:py36-cuda92
  SOFTAGENT_LOCAL_IMAGE SoftAgent image tag, default softgym-softagent:py36-cuda92
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
    -v "$REPO_ROOT:$CONTAINER_ROOT"
    -w "$SOFTAGENT_ROOT"
  )

  # Pin to a GPU only when CUDA_VISIBLE_DEVICES is explicitly set. Passing an
  # empty value hides all CUDA devices inside the container.
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    args+=(-e "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
  fi

  if [[ -n "${DISPLAY:-}" && -d /mnt/wslg/.X11-unix ]]; then
    args+=(
      -v /mnt/wslg/.X11-unix:/tmp/.X11-unix
      -e "DISPLAY=$DISPLAY"
      -e QT_X11_NO_MITSHM=1
    )
  elif [[ -n "${DISPLAY:-}" && -d /tmp/.X11-unix ]]; then
    args+=(
      -v /tmp/.X11-unix:/tmp/.X11-unix
      -e "DISPLAY=$DISPLAY"
      -e QT_X11_NO_MITSHM=1
    )
  fi

  if [[ "$SOFTGYM_SOFTWARE_GL" == "1" ]]; then
    args+=(
      -e LIBGL_ALWAYS_SOFTWARE=1
      -e MESA_LOADER_DRIVER_OVERRIDE=swrast
      -e MESA_GL_VERSION_OVERRIDE=${MESA_GL_VERSION_OVERRIDE:-4.5}
      -e MESA_GLSL_VERSION_OVERRIDE=${MESA_GLSL_VERSION_OVERRIDE:-450}
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

softagent_env='export PYFLEXROOT=/workspace/softgym/PyFlex; export PYTHONPATH=/workspace/softgym/softagent/rlpyt_cloth:/workspace/softgym/softagent:/workspace/softgym:/workspace/softgym/PyFlex/bindings/build:${PYTHONPATH:-}; export LD_LIBRARY_PATH=/workspace/softgym/PyFlex/external/SDL2-2.0.4/lib/x64:${LD_LIBRARY_PATH:-}; export PYTORCH_JIT=0; export MUJOCO_GL=osmesa; export MUJOCO_gl=egl; export EGL_GPU=${CUDA_VISIBLE_DEVICES:-0}'

quote_command() {
  local quoted=()
  local arg
  for arg in "$@"; do
    quoted+=("$(printf "%q" "$arg")")
  done
  printf "%s " "${quoted[@]}"
}

ensure_base_image() {
  if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
    "$SCRIPT_DIR/softgym-local.sh" build
  fi
}

build_image() {
  ensure_base_image
  docker build \
    --build-arg "BASE_IMAGE=$BASE_IMAGE" \
    -f "$REPO_ROOT/docker/Dockerfile.softagent" \
    -t "$IMAGE" \
    "$REPO_ROOT"
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
  run_container bash -lc "$softagent_env; mkdir -p ../PyFlex/bindings/build && cd ../PyFlex/bindings/build && cmake -DPYBIND11_PYTHON_VERSION=3.6 .. && make -j\$(nproc)"
}

ensure_pyflex() {
  if ! compgen -G "$REPO_ROOT/PyFlex/bindings/build/pyflex*.so" >/dev/null; then
    compile_pyflex
  fi
}

run_benchmark() {
  local algorithm=${1:-cem}
  shift || true
  ensure_image
  ensure_pyflex
  mkdir -p "$REPO_ROOT/data/softagent/$algorithm"

  local cmd=(
    python
    experiments/run_softgym_benchmark.py
    --algorithm "$algorithm"
    "$@"
  )
  run_container bash -lc "$softagent_env; $(quote_command "${cmd[@]}")"
}

case "${1:-}" in
  -h|--help|help)
    usage
    ;;
  build)
    build_image
    ;;
  compile)
    compile_pyflex
    ;;
  shell)
    ensure_image
    ensure_pyflex
    mapfile -t args < <(docker_common_args)
    docker run -it "${args[@]}" "$IMAGE" bash -lc "$softagent_env; exec bash"
    ;;
  run)
    shift
    if [[ $# -eq 0 ]]; then
      usage
      exit 2
    fi
    ensure_image
    ensure_pyflex
    run_container bash -lc "$softagent_env; $*"
    ;;
  benchmark)
    shift
    run_benchmark "${1:-cem}" "${@:2}"
    ;;
  cem|curl|drq|planet|mvp)
    algorithm=$1
    shift
    run_benchmark "$algorithm" "$@"
    ;;
  *)
    usage
    exit 2
    ;;
esac
