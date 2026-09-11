#!/usr/bin/env bash
# Step 5: combine the per-worker shards into one dataset, renumbered demo_0..N-1, optionally dropping
# the first step of every demo.
#
# Both scripts are pure h5py, so this runs on the host: no container, no GPU. Arena's merge_demos.py
# is used for its cross-shard schema validation (--dry_run); trim_first_step.py trims each shard in
# parallel (recompressing camera chunks is the slow part, ~1 min/GB per process) and then merges the
# parts with a raw chunk copy. With TRIM_STEPS=0 (default) the shards are raw-copied in seconds.
#
# Usage: ./run_merge.sh
#   TRIM_STEPS=1   drop the stale step-0 camera frame. Needed for shards generated before
#                  G1StaticAppleMimicEnv._refresh_camera_obs existed (the round-1 upload, seeds
#                  1000-1005); shards from the fixed env have a correct step 0 and need no trim.
#   TRIM_JOBS=N    parallel trim processes (default: one per shard, max 12)

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

PY=${MERGE_PY:-$GR00T_DIR/.venv/bin/python}
[[ -x "$PY" ]] || PY=python3
"$PY" -c "import h5py" 2>/dev/null || { echo "h5py not importable with $PY; set MERGE_PY" >&2; exit 1; }

SHARDS=()
for f in "$DATA_DIR"/gen_w*.hdf5; do
  [[ -f "$f" ]] && SHARDS+=("$f")
done
[[ ${#SHARDS[@]} -gt 0 ]] || { echo "no gen_w*.hdf5 shards in $DATA_DIR" >&2; exit 1; }

OUT="$DATA_DIR/$(basename "$MERGED_HDF5")"
echo "[1/2] validating ${#SHARDS[@]} shards (merge_demos.py --dry_run)"
HDF5_USE_FILE_LOCKING=FALSE "$PY" "$ARENA_DIR/isaaclab_arena/scripts/imitation_learning/merge_demos.py" \
  --dry_run -o "$OUT" "${SHARDS[@]}"

echo "[2/2] merging, trimming ${TRIM_STEPS:-0} leading step(s) -> $OUT"
"$PY" trim_first_step.py --overwrite --steps "${TRIM_STEPS:-0}" --jobs "${TRIM_JOBS:-0}" -o "$OUT" "${SHARDS[@]}"

echo
"$PY" dataset_stats.py "$OUT"
echo "next: ./run_convert.sh"
