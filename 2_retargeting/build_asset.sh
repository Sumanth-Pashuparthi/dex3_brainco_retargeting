#!/usr/bin/env bash
# Build the G1 + BrainCo Revo2 robot asset: URDF -> USD, then apply the corrections the converter
# output needs. Patching always follows conversion, so the two are one step.
#
# Conversion runs inside the container because it needs Isaac Sim. Patching runs on the host: the
# corrections are plain text edits to the .usda layers, so they show up in a diff.
#
# Idempotent. Safe to re-run.
#
# Usage: ./build_asset.sh

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

[[ -d "$ARENA_DIR" ]] || { echo "Arena checkout not found at $ARENA_DIR. Run ./setup.sh first." >&2; exit 1; }

# The converter resolves mesh filenames relative to the URDF, so the whole directory is staged
# somewhere the container can see it. /tmp is bind-mounted.
echo "Staging URDF and meshes to $URDF_STAGING"
rm -rf "$URDF_STAGING"
cp -r urdf "$URDF_STAGING"

cp convert_urdf_to_usd.py "$ARENA_DIR/"

./arena_run.sh "cd ${CONTAINER_WORKDIR} && \
PYTHONUNBUFFERED=1 /isaac-sim/python.sh convert_urdf_to_usd.py \
  --urdf ${URDF_STAGING}/g1_29dof_with_brainco_hand.urdf \
  --out ${CONTAINER_WORKDIR}/${ASSET_SUBDIR}/${ASSET_NAME}"

echo
echo "Patching converted USD"
python3 patch_converted_usd.py "$ARENA_DIR/$ASSET_SUBDIR"

cat <<EOF

Asset at $ARENA_DIR/$ASSET_SUBDIR/$ASSET_NAME

Expect 41 articulated joints: 29 body + 6 per hand.
EOF
