#!/usr/bin/env bash
# Prepare an existing Arena checkout for the Mimic pipeline. Idempotent.
#
# Everything heavy is already installed by the earlier steps, so this only:
#   1. checks Isaac Sim, the Arena checkout and the Revo2 embodiment are present
#   2. applies annotate_demos.patch, without which --auto annotation crashes
#
# The plug-in itself is not copied anywhere: arena_run.sh mounts g1_apple_mimic/ onto the
# container's working directory, so edits here apply on the next run with no rebuild.

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

fail() { echo "ERROR: $*" >&2; exit 1; }

echo "[1/2] checking prerequisites"
[[ -d "$ARENA_DIR" ]] || fail "Arena checkout not found at $ARENA_DIR. Run 1_baseline/setup.sh."
docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1 || fail "Docker image $DOCKER_IMAGE not built. Run 1_baseline/setup.sh."
[[ -d "$ARENA_DIR/isaaclab_arena/embodiments/g1_brainco" ]] \
  || fail "Revo2 embodiment not installed. Run 2_retargeting/install.sh."
echo "  Arena checkout, container image and Revo2 embodiment present"

echo "[2/2] patching annotate_demos.py"
TARGET="$ARENA_DIR/isaaclab_arena/scripts/imitation_learning/annotate_demos.py"
[[ -f "$TARGET" ]] || fail "not found: $TARGET"
if grep -qF "_as_tensor" "$TARGET"; then
  echo "  already patched"
elif [[ -d "$ARENA_DIR/.git" ]] && git -C "$ARENA_DIR" apply "$(pwd)/annotate_demos.patch"; then
  echo "  applied annotate_demos.patch (git apply)"
elif patch -p1 -d "$ARENA_DIR" --forward --silent < "$(pwd)/annotate_demos.patch"; then
  echo "  applied annotate_demos.patch (patch -p1; Arena tree is not a git checkout)"
else
  fail "patch did not apply; $TARGET may have changed upstream"
fi

mkdir -p "$DATA_DIR" "$EVAL_DIR" "$LOG_DIR"
echo
echo "ready. Data goes to $DATA_DIR"
echo "next: ./prepare_source.sh   (needs the rollouts from step 3 at $ROLLOUTS_HDF5)"
