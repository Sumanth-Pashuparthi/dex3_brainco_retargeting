#!/usr/bin/env bash
# Step 3: auto-annotate the source demos with the grasp_<arm> subtask signal.
#
# Replays every demo through the same environment and records where the subtask boundary falls.
# --device cpu matches the device the rollouts were recorded on; on cuda the replay diverges enough
# that the grasp does not reproduce and the demo is dropped.
#
# This is the step that needs annotate_demos.patch (applied by setup.sh).
#
# Usage: ./run_annotate.sh [gpu]

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

GPU=${1:-0}
LOG="$LOG_DIR/annotate.log"

echo "annotate: $SOURCE_HDF5 -> $ANNOTATED_HDF5 (gpu $GPU, device ${ANNOTATE_DEVICE:-cpu})"
echo "log: $LOG"

# run_patched.py: Arena's asset library queries the Lightwheel API at import and the SDK's 10 s
# timeout is too short for that service; the wrapper raises it and retries, then runs the script.
DOCKER_LABEL=g1_apple_mimic=annotate DOCKER_NAME=g1_apple_mimic_annotate \
./arena_run.sh "$GPU" \
  "/isaac-sim/python.sh g1_apple_mimic/run_patched.py isaaclab_arena/scripts/imitation_learning/annotate_demos.py \
    --headless --device ${ANNOTATE_DEVICE:-cpu} --auto \
    --input_file ${SOURCE_HDF5} \
    --output_file ${ANNOTATED_HDF5} \
    ${ENV_ARGS}" 2>&1 | tee "$LOG"

echo
python3 dataset_stats.py "$DATA_DIR/$(basename "$ANNOTATED_HDF5")" || true
echo "next: ./run_generate.sh 200 2 \"0\""
