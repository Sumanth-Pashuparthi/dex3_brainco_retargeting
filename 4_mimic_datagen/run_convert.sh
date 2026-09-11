#!/usr/bin/env bash
# Step 6: merged HDF5 -> GR00T-LeRobot dataset -> normalisation stats -> loader validation.
#
# The converter itself needs only h5py/numpy/torch/pandas/cv2/yaml, no Isaac Lab, and the Arena
# container has no pandas, so it runs on the host in the GR00T venv with Arena on PYTHONPATH.
#
# Output: $DATA_DIR/<name>/lerobot, a LeRobot v2 tree with meta/modality.json and meta/stats.json,
# which is what GR00T post-training consumes and what gets pushed to the Hub.
#
# Usage: ./run_convert.sh [name]        (default g1_apple_mimic_generated)

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

NAME=${1:-g1_apple_mimic_generated}
GR00T_PY="$GR00T_DIR/.venv/bin/python"
MODALITY_CFG="$ARENA_DIR/isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_gr00t_n_1_7_config.py"
H_HDF5="$DATA_DIR/${NAME}.hdf5"
H_LEROBOT="$DATA_DIR/${NAME}/lerobot"
LOG="$LOG_DIR/convert_${NAME}.log"

[[ -x "$GR00T_PY" ]] || { echo "GR00T venv not found at $GR00T_PY. Run 1_baseline/setup.sh." >&2; exit 1; }
[[ -f "$H_HDF5" ]] || { echo "merged dataset missing: $H_HDF5. Run ./run_merge.sh." >&2; exit 1; }

echo "[1/3] HDF5 -> LeRobot: $H_LEROBOT"
# The converter prompts before overwriting; clearing the directory keeps this non-interactive.
rm -rf "$H_LEROBOT"
mkdir -p "$DATA_DIR/$NAME"
CFG="$LOG_DIR/lerobot_config_${NAME}.yaml"
sed -e "s#^data_root:.*#data_root: \"${DATA_DIR}\"#" \
    -e "s#^hdf5_name:.*#hdf5_name: \"${NAME}.hdf5\"#" g1_apple_mimic/lerobot_config.yaml > "$CFG"
( cd "$ARENA_DIR" && HDF5_USE_FILE_LOCKING=FALSE PYTHONPATH="$ARENA_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$GR00T_PY" isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py --yaml_file "$CFG" ) 2>&1 | tee "$LOG"
[[ -f "$H_LEROBOT/meta/info.json" ]] || { echo "conversion failed, see $LOG" >&2; exit 1; }

echo "[2/3] normalisation stats -> meta/{stats,relative_stats}.json"
( cd "$GR00T_DIR" && "$GR00T_PY" gr00t/data/stats.py \
    --dataset-path "$H_LEROBOT" --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "$MODALITY_CFG" ) 2>&1 | tee -a "$LOG"

echo "[3/3] validate with GR00T's own loader"
# Resolve the validator path before the cd: $(pwd) inside the subshell would be $GR00T_DIR.
VALIDATOR="$(pwd)/g1_apple_mimic/validate_lerobot.py"
( cd "$GR00T_DIR" && "$GR00T_PY" "$VALIDATOR" \
    --dataset_path "$H_LEROBOT" --modality_config_path "$MODALITY_CFG" ) 2>&1 | tee -a "$LOG"

echo
echo "LeRobot dataset ready: $H_LEROBOT  ($(du -sh "$H_LEROBOT" | cut -f1))"
echo "next: ./push_to_hub.sh <hf-username>/<dataset-name>"
