#!/usr/bin/env bash
# Step 5: combine the per-worker shards into one dataset, renumbered demo_0..N-1.
#
# Usage: ./run_merge.sh [gpu]

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

SHARDS=()
for f in "$DATA_DIR"/gen_w*.hdf5; do
  [[ -f "$f" ]] && SHARDS+=("$C_DATA_DIR/$(basename "$f")")
done
[[ ${#SHARDS[@]} -gt 0 ]] || { echo "no gen_w*.hdf5 shards in $DATA_DIR" >&2; exit 1; }

echo "merging ${#SHARDS[@]} shards -> $MERGED_HDF5"
./arena_run.sh "${1:-0}" \
  "/isaac-sim/python.sh isaaclab_arena/scripts/imitation_learning/merge_demos.py \
    --overwrite -o ${MERGED_HDF5} ${SHARDS[*]}"

echo
python3 dataset_stats.py "$DATA_DIR/$(basename "$MERGED_HDF5")"
echo "next: ./run_convert.sh"
