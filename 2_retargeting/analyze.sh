#!/usr/bin/env bash
# Compare the two hands and print the tables retargeting uses. Host-side, NumPy only.
# Also writes media/aperture_curves.png.
#
# Usage: ./analyze.sh [--skip_thumb_search]

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

[[ -f "$DEX3_URDF" ]] || {
  echo "Source hand URDF not found: $DEX3_URDF" >&2
  echo "Run ./setup.sh, or set DEX3_URDF to a G1 + Dex3-1 URDF." >&2
  exit 1
}

exec python3 analyze_hand_geometry.py --dex3_urdf "$DEX3_URDF" --plot media/aperture_curves.png "$@"
