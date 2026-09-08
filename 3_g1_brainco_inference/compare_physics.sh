#!/usr/bin/env bash
# Log the as-loaded physics of both robots and diff them.
#
# This is the instrumentation that found the real problem. It dumps every per-DoF property PhysX
# actually loaded — stiffness, damping, armature, friction, effort and velocity limits, joint
# limits, link masses — plus the pelvis trajectory, for the baseline and the Revo2 robot, and diffs
# the two.
#
# Usage: ./compare_physics.sh [episodes]      (default 8)

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

EPISODES=${1:-8}
OUT=${OUT:-$EVAL_DIR/physics_compare}
mkdir -p "$OUT"

cp log_pelvis_drift.py "$ARENA_DIR/"

for EMB in "$BASELINE_EMBODIMENT" "$EMBODIMENT"; do
  echo "== $EMB"
  ./arena_run.sh "cd ${CONTAINER_WORKDIR} && \
PYTHONUNBUFFERED=1 /isaac-sim/python.sh log_pelvis_drift.py \
  --policy_type ${POLICY_TYPE} \
  --policy_config_yaml_path ${POLICY_CONFIG} \
  --remote_host ${POLICY_HOST} --remote_port ${POLICY_PORT} \
  --episodes ${EPISODES} --enable_cameras --device cuda \
  ${TASK} --embodiment ${EMB}" | grep -E "^(PHYS|ROOTEP|ROOT_TRACE)" > "$OUT/drift_$EMB.log"
done

echo
echo "== per-DoF differences"
if diff "$OUT/drift_${BASELINE_EMBODIMENT}.log" "$OUT/drift_${EMBODIMENT}.log" \
     --unchanged-group-format='' --old-line-format='  baseline: %L' --new-line-format='  revo2:    %L' \
     | grep -E "PHYSDOF"; then
  echo "  (differences above)"
else
  echo "  none: every authored per-DoF property matches"
fi

echo
echo "Logs in $OUT/"
