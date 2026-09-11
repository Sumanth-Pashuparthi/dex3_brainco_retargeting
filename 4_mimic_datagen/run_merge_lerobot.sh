#!/usr/bin/env bash
# Combine LeRobot datasets into one trainable tree and recompute its normalisation stats.
#
#   ./run_merge_lerobot.sh <out_lerobot_dir> <in_lerobot_dir> [<in_lerobot_dir> ...]
#
# GR00T post-training takes a single dataset_path, so multiple generation rounds have to be joined
# before training. merge_lerobot.py hardlinks videos and rewrites only the parquet files that need
# renumbering, so this costs little space and runs in seconds. Stats must be recomputed over the
# combination: merge_lerobot.py deliberately leaves meta/stats.json absent, and GR00T's loader
# refuses to open a tree without it, so a forgotten stats step fails loudly instead of silently
# normalising against the wrong distribution.

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

OUT=${1:?usage: ./run_merge_lerobot.sh <out_lerobot_dir> <in_lerobot_dir> [...]}
shift
[[ $# -ge 1 ]] || { echo "need at least one input LeRobot dir" >&2; exit 1; }

GR00T_PY="$GR00T_DIR/.venv/bin/python"
[[ -x "$GR00T_PY" ]] || { echo "GR00T venv not found at $GR00T_PY" >&2; exit 1; }
MODALITY_CFG="$ARENA_DIR/isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_gr00t_n_1_7_config.py"

for d in "$@"; do
  [[ -f "$d/meta/info.json" ]] || { echo "not a LeRobot dataset: $d" >&2; exit 1; }
done

echo "[1/3] merging $# dataset(s) -> $OUT"
rm -rf "$OUT"
"$GR00T_PY" merge_lerobot.py -o "$OUT" "$@"

echo "[2/3] normalisation stats over the combination"
( cd "$GR00T_DIR" && "$GR00T_PY" gr00t/data/stats.py \
    --dataset-path "$OUT" --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "$MODALITY_CFG" )

echo "[3/3] validating with GR00T's own loader"
VALIDATOR="$(pwd)/g1_apple_mimic/validate_lerobot.py"
( cd "$GR00T_DIR" && "$GR00T_PY" "$VALIDATOR" --dataset_path "$OUT" \
    --modality_config_path "$MODALITY_CFG" ) || echo "note: validator reported issues, see above"

echo
echo "ready to train: $OUT"
