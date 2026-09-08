#!/usr/bin/env bash
# Run a command inside the IsaacLab-Arena container, non-interactively.
#
# Mirrors the mounts and flags of the Arena repo's own docker/run_docker.sh, minus the X11 and
# interactive-TTY parts, so the simulator can be driven headlessly from a script.
#
# Mounts, all defined in env.sh:
#   $ARENA_DIR  -> /workspaces/isaaclab_arena   the checkout, so edits apply without a rebuild
#   $MODELS_DIR -> /models                      where the policy config's model_path resolves
#   $EVAL_DIR   -> /eval                        videos and logs written by the run
#
# BRAINCO_DEBUG and BRAINCO_GRASP_DEPTH are forwarded because the retargeting layer reads them
# inside the container.
#
# Usage: ./arena_run.sh "cd /workspaces/isaaclab_arena && /isaac-sim/python.sh ..."

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

[[ -d "$ARENA_DIR" ]] || { echo "Arena checkout not found at $ARENA_DIR. Run 1_baseline/setup.sh first." >&2; exit 1; }
[[ -d "$MODELS_DIR" ]] || { echo "Models directory not found at $MODELS_DIR. Run 1_baseline/setup.sh first." >&2; exit 1; }
mkdir -p "$EVAL_DIR"

docker run --rm \
  --privileged --ulimit memlock=-1 --ulimit stack=-1 \
  --ipc=host --net=host --runtime=nvidia --gpus=all \
  -v "${ARENA_DIR}":${CONTAINER_WORKDIR} \
  -v "${MODELS_DIR}":/models \
  -v "${EVAL_DIR}":/eval \
  -v /tmp:/tmp \
  -v "$HOME/.cache":/home/"$(id -un)"/.cache \
  --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --env OMNI_KIT_ALLOW_ROOT=1 \
  --env DOCKER_RUN_USER_ID="$(id -u)" --env DOCKER_RUN_USER_NAME="$(id -un)" \
  --env DOCKER_RUN_GROUP_ID="$(id -g)" --env DOCKER_RUN_GROUP_NAME="$(id -gn)" \
  --env ISAACLAB_PATH=${CONTAINER_WORKDIR}/submodules/IsaacLab \
  --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
  --env BRAINCO_DEBUG="${BRAINCO_DEBUG:-}" \
  --env BRAINCO_GRASP_DEPTH="${BRAINCO_GRASP_DEPTH:-}" \
  "${DOCKER_IMAGE}" "$*"
