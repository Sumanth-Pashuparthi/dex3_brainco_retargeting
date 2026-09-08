#!/usr/bin/env bash
# Step 0: harvest successful episodes from the retargeted policy.
#
#   ./run_harvest.sh [num_demos=40] [max_episodes=800] [gpu=0]
#
# This is the substitute for teleoperation. Arena's recorder already stamps the success termination
# term on each episode and EXPORT_SUCCEEDED_ONLY drops the rest, so a plain rollout loop with
# terminations left enabled yields success-filtered demos with no changes to Arena.
#
# At the measured 0.06 success rate, expect roughly 6 demos per 100 episodes and about 15 s per
# episode. --num_envs 32 runs them in parallel; drop it if VRAM is tight.
#
# Needs 1_baseline/run_policy_server.sh running on the host.
#
# Writes $ROLLOUTS_HDF5 plus a .meta.json sidecar recording the joint order, which
# prepare_source.sh reads to rebuild the wrist trajectories.

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

NUM_DEMOS=${1:-40}
MAX_EPISODES=${2:-800}
GPU=${3:-0}

mkdir -p "$(dirname "$ROLLOUTS_HDF5")"
LOG="$LOG_DIR/harvest.log"

echo "harvesting up to $NUM_DEMOS demos from at most $MAX_EPISODES episodes -> $ROLLOUTS_HDF5"
echo "log: $LOG"

DOCKER_LABEL=g1_apple_mimic=harvest DOCKER_NAME=g1_apple_mimic_harvest \
./arena_run.sh "$GPU" \
  "/isaac-sim/python.sh g1_apple_mimic/record_policy_demos.py \
    --headless --enable_cameras --num_envs ${HARVEST_NUM_ENVS:-32} \
    --policy_type ${POLICY_TYPE} \
    --policy_config_yaml_path ${POLICY_CONFIG} \
    --remote_host ${POLICY_HOST} --remote_port ${POLICY_PORT} \
    --dataset_file ${C_ROLLOUTS_HDF5} \
    --num_demos ${NUM_DEMOS} --max_episodes ${MAX_EPISODES} \
    --dex3_actions ${GRASP_DEPTH_RANGE:+--grasp_depth_range $GRASP_DEPTH_RANGE} \
    galileo_g1_static_pick_and_place \
    --object ${OBJECT} --destination ${DESTINATION} \
    --embodiment ${HARVEST_EMBODIMENT}" 2>&1 | tee "$LOG"

echo
python3 dataset_stats.py "$ROLLOUTS_HDF5" || true
echo "next: ./prepare_source.sh"
