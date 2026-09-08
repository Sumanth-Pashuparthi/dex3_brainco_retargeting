#!/usr/bin/env bash
# Run the frozen GR00T N1.7 policy on the G1 + BrainCo Revo2 robot and report the success rate.
#
# Identical to 1_baseline/run_eval.sh apart from --embodiment, which is the point: policy, task,
# scene, success criterion, episode length, control rate and evaluation script are all the same, so
# the difference in the numbers is attributable to the hand swap and the retargeting layer.
#
# Rates from runs shorter than 100 episodes are not meaningful here. One configuration scored 1/10
# and then 0/20 on repeat.
#
# Usage: ./run_eval.sh [num_episodes]      (default 100)

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

EPISODES=${1:-100}

./arena_run.sh "cd ${CONTAINER_WORKDIR} && \
PYTHONUNBUFFERED=1 /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --policy_type ${POLICY_TYPE} \
  --policy_config_yaml_path ${POLICY_CONFIG} \
  --remote_host ${POLICY_HOST} --remote_port ${POLICY_PORT} \
  --num_episodes ${EPISODES} --enable_cameras --device cuda \
  ${TASK} --embodiment ${EMBODIMENT}"
