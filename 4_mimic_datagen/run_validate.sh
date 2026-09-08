#!/usr/bin/env bash
# Step 2: check the source demos before spending simulator time on them.
#
# Verifies action shape (T, 23), success flags, that exactly one arm closes, and that the apple and
# plate are present in initial_state. Prints which arm did the pick, which is what MIMIC_ARM wants.
#
# Usage: ./run_validate.sh [gpu]

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

GPU=${1:-0}
exec ./arena_run.sh "$GPU" \
  "/isaac-sim/python.sh g1_apple_mimic/validate_source_demos.py ${SOURCE_HDF5} \
    --object ${OBJECT} --destination ${DESTINATION}"
