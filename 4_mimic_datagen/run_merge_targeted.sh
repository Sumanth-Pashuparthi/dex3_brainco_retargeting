#!/usr/bin/env bash
# Step 5b: combine the targeted gen_t* shards into their own dataset, then convert and merge with
# round 2 at the LeRobot level.
#
#   ./run_merge_targeted.sh [name=g1_apple_mimic_targeted]
#
# Why not merge the HDF5s: round 2's merged HDF5 is ~210 GB and a 1000-demo round adds ~126 GB, so
# a combined HDF5 would need ~340 GB of free space plus the shards. The LeRobot form of the same
# data is under 2 GB, so the rounds are merged there instead (merge_lerobot.py) and the HDF5s stay
# separate. Round 2's dataset and LeRobot tree are left untouched, so the old baseline stays
# reproducible.
#
# Steps: validate + merge gen_t* -> <name>.hdf5, convert it to LeRobot, merge round 2's LeRobot
# tree with it, recompute normalisation stats over the combination.

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

NAME=${1:-g1_apple_mimic_targeted}
BASE_NAME=${BASE_NAME:-g1_apple_mimic_generated}
COMBINED_NAME=${COMBINED_NAME:-g1_apple_mimic_r2r3}

PY=${MERGE_PY:-$GR00T_DIR/.venv/bin/python}
[[ -x "$PY" ]] || PY=python3
"$PY" -c "import h5py" 2>/dev/null || { echo "h5py not importable with $PY; set MERGE_PY" >&2; exit 1; }

SHARDS=()
for f in "$DATA_DIR"/gen_t*.hdf5; do
  [[ -f "$f" ]] && SHARDS+=("$f")
done
[[ ${#SHARDS[@]} -gt 0 ]] || { echo "no gen_t*.hdf5 shards in $DATA_DIR. Run ./run_generate_targeted.sh first." >&2; exit 1; }

OUT="$DATA_DIR/${NAME}.hdf5"
echo "[1/4] validating ${#SHARDS[@]} targeted shards"
HDF5_USE_FILE_LOCKING=FALSE "$PY" "$ARENA_DIR/isaaclab_arena/scripts/imitation_learning/merge_demos.py" \
  --dry_run -o "$OUT" "${SHARDS[@]}"

echo "[2/4] merging -> $OUT"
"$PY" trim_first_step.py --overwrite --steps "${TRIM_STEPS:-0}" --jobs "${TRIM_JOBS:-0}" -o "$OUT" "${SHARDS[@]}"
"$PY" dataset_stats.py "$OUT"

echo "[3/4] converting the targeted round to LeRobot"
./run_convert.sh "$NAME"

echo "[4/4] merging round 2 + targeted at the LeRobot level -> $DATA_DIR/$COMBINED_NAME/lerobot"
BASE_LEROBOT="$DATA_DIR/$BASE_NAME/lerobot"
NEW_LEROBOT="$DATA_DIR/$NAME/lerobot"
[[ -f "$BASE_LEROBOT/meta/info.json" ]] || { echo "round-2 LeRobot tree missing at $BASE_LEROBOT" >&2; exit 1; }
./run_merge_lerobot.sh "$DATA_DIR/$COMBINED_NAME/lerobot" "$BASE_LEROBOT" "$NEW_LEROBOT"

echo
echo "train on: $DATA_DIR/$COMBINED_NAME/lerobot"
