#!/usr/bin/env bash
# One-time installation for building the robot asset.
#
# A subset of what the baseline step needs: the simulator, but no policy checkpoint and no
# inference server, because nothing here runs a policy. If you already ran the baseline setup this
# finds everything in place and exits immediately.
#
# Usage: ./setup.sh

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

# ---------------------------------------------------------------- prerequisites
step "Checking prerequisites"
missing=0
for tool in docker git python3; do
  command -v "$tool" >/dev/null 2>&1 || { echo "  MISSING: $tool"; missing=1; }
done
python3 -c "import numpy" 2>/dev/null || { echo "  MISSING: numpy  (pip install numpy)"; missing=1; }
[[ $missing -eq 0 ]] || { echo "Install the tools above and re-run."; exit 1; }
echo "  ok"

# ---------------------------------------------------------------- isaac sim image
step "Isaac Sim base image: $ISAAC_SIM_IMAGE"
if docker image inspect "$ISAAC_SIM_IMAGE" >/dev/null 2>&1; then
  echo "  already present"
else
  echo "  ~29 GB from NGC; run 'docker login nvcr.io' first."
  echo "  Username is exactly \$oauthtoken, password is your NGC API key."
  for attempt in 1 2 3; do
    if docker pull "$ISAAC_SIM_IMAGE"; then break; fi
    echo "  pull attempt $attempt failed, retrying"
    [[ $attempt -lt 3 ]] || { echo "  giving up"; exit 1; }
  done
fi

# ---------------------------------------------------------------- arena
step "IsaacLab-Arena ($ARENA_BRANCH)"
mkdir -p "$WORK_DIR"
if [[ -d "$ARENA_DIR/.git" ]]; then
  echo "  already cloned at $ARENA_DIR"
else
  # release/0.2.1, not main: main's Dockerfile references an internal NVIDIA registry.
  git clone -b "$ARENA_BRANCH" --recursive "$ARENA_REPO" "$ARENA_DIR"
fi
# --recursive matters here beyond the usual reasons: the Dex3 URDF the geometry analysis compares
# against lives inside the Isaac-GR00T submodule.
git -C "$ARENA_DIR" submodule update --init --recursive

step "Arena Docker image"
if docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
  echo "  already built"
else
  echo "  Building, ~20 minutes."
  docker build --pull \
    --progress=plain \
    --build-arg WORKDIR="$CONTAINER_WORKDIR" \
    --build-arg INSTALL_GROOT=true \
    -t "$DOCKER_IMAGE" \
    --file "$ARENA_DIR/docker/Dockerfile.isaaclab_arena" \
    "$ARENA_DIR"
fi

# ---------------------------------------------------------------- sanity
step "Source hand URDF"
if [[ -f "$DEX3_URDF" ]]; then
  echo "  found: $DEX3_URDF"
else
  echo "  NOT FOUND: $DEX3_URDF"
  echo "  The geometry analysis needs it. Check that submodules cloned:"
  echo "    git -C $ARENA_DIR submodule update --init --recursive"
  exit 1
fi

cat <<EOF

Setup complete.

  Arena       $ARENA_DIR
  asset out   $ARENA_DIR/$ASSET_SUBDIR/$ASSET_NAME

Next:
  ./build_asset.sh                   # URDF -> USD plus corrections, expect 41 articulated joints
  ./install.sh                       # install the embodiment into the Arena checkout
  python3 test_retargeting.py        # check the layer without a simulator
  ./analyze.sh                       # optional: re-derive the tables from the two URDFs
EOF
