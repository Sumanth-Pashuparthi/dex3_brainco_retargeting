#!/usr/bin/env bash
# Evaluate the frozen GR00T N1.7 policy on the stock G1 + Dex3-1 apple-to-plate task.
#
# Nothing here is modified: task, embodiment, checkpoint and success criterion are all upstream
# IsaacLab-Arena. This produces the reference number.
#
# PYTHONUNBUFFERED=1 matters. Metrics are emitted with plain print(), which is block-buffered when
# stdout is not a terminal, so the numbers are lost if the container exits before the buffer flushes.
#
# The first ~4 minutes are Isaac Sim startup with no output. That is normal.
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
  ${TASK_ARGS}"
