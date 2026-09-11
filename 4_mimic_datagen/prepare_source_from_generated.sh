#!/usr/bin/env bash
# Step 1 (alternative): Mimic *generated* demos -> Mimic *source* demos for another round.
#
# Use this instead of prepare_source.sh when the harvested rollouts are gone but a generated dataset
# (shards or merged) exists. Every generated episode is a successful 23-D Pink demo with its exact
# initial_state, which is all annotate_demos.py reads. Runs on the host: h5py, numpy, scipy only.
#
# Usage: ./prepare_source_from_generated.sh [generated.hdf5]   (default: $MERGED_HDF5)
#   NUM_SOURCES  demos to keep, farthest-point sampled on apple XY (env.sh default 24)
#   TIME_SCALE   time-stretch factor; 1.0 keeps the original speed (env.sh default 2.0)
#   PAD_STEPS    settle steps appended after the last motion (default 60)
#   SOURCE_DEMOS explicit comma-separated demo indices, overrides NUM_SOURCES

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

IN=${1:-$DATA_DIR/$(basename "$MERGED_HDF5")}
OUT="$DATA_DIR/$(basename "$SOURCE_HDF5")"
[[ -f "$IN" ]] || { echo "generated dataset not found: $IN" >&2; exit 1; }
PY=${MERGE_PY:-$GR00T_DIR/.venv/bin/python}
[[ -x "$PY" ]] || PY=python3
"$PY" -c "import h5py, scipy" 2>/dev/null || { echo "h5py/scipy not importable with $PY; set MERGE_PY" >&2; exit 1; }

if [[ -f "$OUT" ]]; then
  mv "$OUT" "$OUT.bak.$(date +%s)"
  echo "existing $OUT moved aside"
fi

EXTRA=()
[[ -n "${SOURCE_DEMOS:-}" ]] && EXTRA+=(--demos "$SOURCE_DEMOS")
"$PY" g1_apple_mimic/make_source_from_generated.py "$IN" "$OUT" \
  --num_sources "$NUM_SOURCES" --time_scale "$TIME_SCALE" --pad_steps "${PAD_STEPS:-60}" "${EXTRA[@]}"

echo
echo "wrote $OUT"
echo "next: ./run_annotate.sh"
