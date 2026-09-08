#!/usr/bin/env bash
# Install the Revo2 embodiment into an IsaacLab-Arena checkout.
#
# Copies the package into isaaclab_arena/embodiments/ and appends the import that makes
# @register_asset run, so `--embodiment g1_wbc_agile_joint_brainco` resolves on the CLI.
#
# Usage: ./install.sh [/path/to/IsaacLab-Arena]     (defaults to $ARENA_DIR from env.sh)

set -euo pipefail
SRC=$(cd "$(dirname "$0")" && pwd)
source "$SRC/env.sh"

ARENA=${1:-$ARENA_DIR}
DST="$ARENA/isaaclab_arena/embodiments"
[[ -d "$DST" ]] || { echo "not an IsaacLab-Arena checkout: $ARENA" >&2; exit 1; }

# The package carries its own copy of the measured tables, because it runs inside the container
# where this folder is not importable. Refreshing it here is what keeps the two from drifting.
cp "$SRC/hand_specs.py" "$SRC/g1_brainco/hand_specs.py"

cp -r "$SRC/g1_brainco" "$DST/"
echo "installed $DST/g1_brainco"

INIT="$DST/__init__.py"
if grep -qF "g1_brainco.g1_brainco" "$INIT"; then
  echo "registration already present in $INIT"
else
  printf '\n%s\n' "from .g1_brainco.g1_brainco import *  # noqa: F401,F403" >> "$INIT"
  echo "appended registration to $INIT"
fi

ASSET="$ARENA/$ASSET_SUBDIR/$ASSET_NAME"
if [[ -f "$ASSET" ]]; then
  echo "robot asset present: $ASSET"
else
  echo
  echo "WARNING: robot asset missing at $ASSET"
  echo "Run ./build_asset.sh — the embodiment spawns exactly this path."
fi
