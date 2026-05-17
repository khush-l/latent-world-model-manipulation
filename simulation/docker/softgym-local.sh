#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
IMAGE=${SOFTGYM_LOCAL_IMAGE:-softgym-local:py36-cuda92}
CONTAINER_ROOT=/workspace/softgym

usage() {
  cat <<'USAGE'
Usage:
  docker/softgym-local.sh build
  docker/softgym-local.sh compile
  docker/softgym-local.sh example [EnvName]
  docker/softgym-local.sh run "python examples/random_env.py --env_name ClothDrop --headless 1"
  docker/softgym-local.sh shell

Default example environment: ClothDrop
USAGE
}

docker_common_args() {
  local args=(
    --rm
    --gpus all
    --user "$(id -u):$(id -g)"
    -e HOME=/tmp
    -v "$REPO_ROOT:$CONTAINER_ROOT"
    -w "$CONTAINER_ROOT"
  )

  if [[ -n "${DISPLAY:-}" && -d /tmp/.X11-unix ]]; then
    args+=(
      -v /tmp/.X11-unix:/tmp/.X11-unix
      -e "DISPLAY=$DISPLAY"
      -e QT_X11_NO_MITSHM=1
    )
  fi

  printf '%s\n' "${args[@]}"
}

pyflex_env='export PYFLEXROOT=/workspace/softgym/PyFlex; export PYTHONPATH=/workspace/softgym:/workspace/softgym/PyFlex/bindings/build:${PYTHONPATH:-}; export LD_LIBRARY_PATH=/workspace/softgym/PyFlex/external/SDL2-2.0.4/lib/x64:${LD_LIBRARY_PATH:-}'

build_image() {
  docker build -f "$REPO_ROOT/docker/Dockerfile.local" -t "$IMAGE" "$REPO_ROOT"
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

  mkdir -p "$REPO_ROOT/data"
  local quoted_env
  quoted_env=$(printf '%q' "$env_name")
  run_container bash -lc "$pyflex_env; python examples/random_env.py --env_name $quoted_env --headless 1 --num_variations 1 --save_video_dir ./data"
}

case "${1:-}" in
  build)
    build_image
    ;;
  compile)
    compile_pyflex
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
