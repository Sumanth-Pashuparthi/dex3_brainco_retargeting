#!/usr/bin/env bash
# Run a command inside the IsaacLab-Arena container, non-interactively, pinned to one GPU.
#
# Same mounts and flags as the other steps, plus the dataset and eval directories this pipeline
# reads and writes, and the plug-in mounted onto the container's working directory so
# `g1_apple_mimic.<module>` imports without a rebuild.
#
# Usage: ./arena_run.sh <gpu> "/isaac-sim/python.sh g1_apple_mimic/smoke_test.py ..."
#
# DOCKER_NAME and DOCKER_LABEL let the launcher tag each worker so status.sh and stop.sh can find
# them.

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

[[ -d "$ARENA_DIR" ]] || { echo "Arena checkout not found at $ARENA_DIR. Run 1_baseline/setup.sh first." >&2; exit 1; }
mkdir -p "$DATA_DIR" "$EVAL_DIR" "$LOG_DIR"

GPU=${1:?usage: arena_run.sh <gpu> <command>}
shift

docker run --rm \
  ${DOCKER_NAME:+--name "$DOCKER_NAME"} \
  ${DOCKER_LABEL:+--label "$DOCKER_LABEL"} \
  --privileged --ulimit memlock=-1 --ulimit stack=-1 \
  --ipc=host --net=host --runtime=nvidia --gpus "device=${GPU}" \
  -v "${ARENA_DIR}":${CONTAINER_WORKDIR} \
  -v "$(pwd)/g1_apple_mimic":${CONTAINER_WORKDIR}/g1_apple_mimic \
  -v "${MODELS_DIR}":/models \
  -v "$(dirname "$DATA_DIR")":/datasets \
  -v "$(dirname "$EVAL_DIR")":/eval \
  -v /tmp:/tmp \
  -v "$HOME/.cache":/home/"$(id -un)"/.cache \
  --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --env OMNI_KIT_ALLOW_ROOT=1 \
  --env DOCKER_RUN_USER_ID="$(id -u)" --env DOCKER_RUN_USER_NAME="$(id -un)" \
  --env DOCKER_RUN_GROUP_ID="$(id -g)" --env DOCKER_RUN_GROUP_NAME="$(id -gn)" \
  --env ISAACLAB_PATH=${CONTAINER_WORKDIR}/submodules/IsaacLab \
  --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
  --env HDF5_USE_FILE_LOCKING=FALSE \
  --env PYTHONPATH=${CONTAINER_WORKDIR} \
  "${DOCKER_IMAGE}" "cd ${CONTAINER_WORKDIR} && $*"
