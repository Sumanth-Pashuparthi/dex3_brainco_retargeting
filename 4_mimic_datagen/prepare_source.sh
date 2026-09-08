#!/usr/bin/env bash
# Step 1: successful rollouts (50-D joint actions) -> Mimic source demos (23-D Pink actions).
#
# Runs on the host: the converter needs only h5py, numpy and scipy, no simulator.
#
# Usage: ./prepare_source.sh [rollouts.hdf5]      (defaults to $ROLLOUTS_HDF5)
#
# Knobs worth knowing, all passed through as environment variables:
#   PAD_STEPS      hold-pose/open-hand steps appended so the apple can settle (default 40)
#   LOOKAHEAD      wrist target at t = measured pose at t+LOOKAHEAD (default 1)
#   SCRIPTED_PLACE 1 to replace the policy's carry/release with a scripted one (default 1)

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

IN=${1:-$ROLLOUTS_HDF5}
OUT="$DATA_DIR/$(basename "$SOURCE_HDF5")"
[[ -f "$IN" ]] || { echo "rollouts not found: $IN. Record them with 3_g1_brainco_inference." >&2; exit 1; }
[[ -f "${IN%.hdf5}.meta.json" ]] || { echo "missing sidecar: ${IN%.hdf5}.meta.json (written alongside the rollouts)" >&2; exit 1; }
mkdir -p "$DATA_DIR"

EXTRA=()
[[ "${SCRIPTED_PLACE:-1}" == "1" ]] && EXTRA+=(--scripted_place)

python3 g1_apple_mimic/convert_joint_demos_to_pink_source.py "$IN" "$OUT" \
  --target revo2 \
  --pad_steps "${PAD_STEPS:-40}" \
  --lookahead "${LOOKAHEAD:-1}" \
  "${EXTRA[@]}"

echo
echo "wrote $OUT"
echo "next: ./run_validate.sh"
