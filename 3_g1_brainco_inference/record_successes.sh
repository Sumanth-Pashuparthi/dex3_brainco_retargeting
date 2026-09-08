#!/usr/bin/env bash
# Record one labelled clip per successful episode, plus the first few failures.
#
# At a few percent success, concatenating every episode into one video is useless: a 20-episode run
# is 6000 frames of near-identical failures and, in three attempts, contained no success at all.
#
# Usage: ./record_successes.sh [num_episodes] [keep_failures]     (default 100, 3)

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

EPISODES=${1:-100}
KEEP_FAILURES=${2:-3}

cp record_success_episodes.py "$ARENA_DIR/"

./arena_run.sh "cd ${CONTAINER_WORKDIR} && \
PYTHONUNBUFFERED=1 /isaac-sim/python.sh record_success_episodes.py \
  --policy_type ${POLICY_TYPE} \
  --policy_config_yaml_path ${POLICY_CONFIG} \
  --remote_host ${POLICY_HOST} --remote_port ${POLICY_PORT} \
  --record_episodes ${EPISODES} --keep_failures ${KEEP_FAILURES} \
  --out_dir /eval/revo2_success \
  --enable_cameras --device cuda \
  ${TASK} --embodiment ${EMBODIMENT}"

echo
echo "Clips in $EVAL_DIR/revo2_success/"
echo "Check them by eye before believing the count; see README.md."
