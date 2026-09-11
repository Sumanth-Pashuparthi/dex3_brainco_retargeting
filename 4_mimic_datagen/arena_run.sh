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

# Isaac Lab mirrors remote files under $TMPDIR, keyed by URL path: object USDs from S3 land in
# /tmp/Assets/..., the WBC ONNX policy from GitHub in /tmp/nvidia-isaac/.... Existing USDs are never
# re-downloaded and copy errors are ignored; the ONNX is re-copied with OVERWRITE every start. /tmp is
# bind-mounted, so a mirror written by a container that ran as root breaks both: unreadable USDs spawn
# as empty prims ("No contact sensors added to the prim ... no rigid bodies") and the ONNX overwrite
# fails ("Unable to copy file ... Is the Nucleus Server running?"). Refuse to start in that state; the
# fix runs inside the container, which has passwordless sudo.
#
# SKIP_TMP_GUARD=1 turns the check off. The repair itself has to run in the container, so without
# a bypass the guard would reject the very command that fixes it.
MIRRORS=(/tmp/Assets /tmp/nvidia-isaac /tmp/isaaclab)
if [[ "${SKIP_TMP_GUARD:-0}" != "1" ]]; then
  for m in "${MIRRORS[@]}"; do
    [[ -d "$m" ]] || continue
    if [[ -n "$(find "$m" ! -writable -print -quit 2>/dev/null)" ]]; then
      echo "ERROR: $m contains files this user cannot read/write (left by a root container run)." >&2
      echo "  fix: SKIP_TMP_GUARD=1 DOCKER_LABEL= DOCKER_NAME= ./arena_run.sh $GPU \"sudo chown -R \$(id -u):\$(id -g) ${MIRRORS[*]}\"" >&2
      exit 1
    fi
  done
fi

# No --privileged (upstream run_docker.sh has it): a privileged container gets every /dev/nvidia*
# node, so --gpus "device=N" is silently ignored, all GPUs are visible, and Kit puts its RTX renderer
# on whichever GPU Vulkan enumerates first, regardless of --device. With nine workers that stacked
# nine renderers on one GPU; when it filled up, RTX failed *silently* and the recorded camera frames
# were all black. Without --privileged the nvidia runtime exposes only the requested GPU, which also
# pins the renderer. NVIDIA_VISIBLE_DEVICES is set explicitly because the image bakes in "all".
docker run --rm \
  ${DOCKER_NAME:+--name "$DOCKER_NAME"} \
  ${DOCKER_LABEL:+--label "$DOCKER_LABEL"} \
  ${DOCKER_PRIVILEGED:+--privileged} --ulimit memlock=-1 --ulimit stack=-1 \
  --ipc=host --net=host --runtime=nvidia --gpus "device=${GPU}" \
  --env NVIDIA_VISIBLE_DEVICES="${GPU}" --env NVIDIA_DRIVER_CAPABILITIES=all \
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
