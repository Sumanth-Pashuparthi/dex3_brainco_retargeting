#!/usr/bin/env bash
# Evaluate a fine-tuned checkpoint on the Revo2 robot and report the success rate.
#
# Start ./run_policy_server.sh with the checkpoint you want first, then:
#
#   ./run_eval.sh 20                      # 20 episodes at the stock deterministic spawn
#   EVAL_XY_RANGE_M=0.05 ./run_eval.sh 40 # 40 episodes with the +/-5 cm spawn jitter
#
# This is step 3's evaluation with different weights behind it. Task, embodiment, success term,
# episode length and control rate all come from env.sh unchanged, so a number produced here is
# comparable to the 0.06 that step 3 measured -- but only at the same EVAL_XY_RANGE_M. The default
# is 0.0 for exactly that reason.
#
# Two caveats that decide whether a number is worth quoting:
#   - n matters. At n=10 the 95% interval is about +/-0.3, wide enough to hide any effect this
#     work produced. The sweep in results/ pools repeats to get to 40-140.
#   - the success term also fires when the apple is *pushed* into the plate region, so audit the
#     videos in $EVAL_DIR before treating a rate as pick-and-place.

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

EPISODES=${1:-20}
STAMP=$(date '+%Y%m%d_%H%M%S')
OUT_SUBDIR="rollouts_${STAMP}"

if ! command -v ss >/dev/null || ! ss -ltn 2>/dev/null | grep -q ":${POLICY_PORT} "; then
  echo "no policy server on port ${POLICY_PORT}; start ./run_policy_server.sh <step> first" >&2
  exit 1
fi

mkdir -p "$EVAL_DIR/$OUT_SUBDIR"
echo "evaluating $EPISODES episodes  jitter=${EVAL_XY_RANGE_M} m  episode=${EPISODE_LENGTH_S} s"
echo "  -> $EVAL_DIR/$OUT_SUBDIR"

./arena_run.sh "cd ${CONTAINER_WORKDIR} && \
PYTHONUNBUFFERED=1 /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --policy_type ${POLICY_TYPE} \
  --policy_config_yaml_path ${POLICY_CONFIG} \
  --remote_host ${POLICY_HOST} --remote_port ${POLICY_PORT} \
  --num_episodes ${EPISODES} --enable_cameras --device cuda \
  --apple_xy_range_m ${EVAL_XY_RANGE_M} \
  --episode_length_s ${EPISODE_LENGTH_S} \
  --output_dir /eval/${OUT_SUBDIR} \
  ${TASK} --embodiment ${EMBODIMENT}" 2>&1 | tee "$EVAL_DIR/$OUT_SUBDIR/eval.log"

echo
grep -aoE 'SUCCESS RATE: [0-9]+/[0-9]+ = [0-9.]+' "$EVAL_DIR/$OUT_SUBDIR/eval.log" | tail -1 \
  || echo "no success rate line found; see $EVAL_DIR/$OUT_SUBDIR/eval.log"
