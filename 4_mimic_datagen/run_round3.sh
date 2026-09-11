#!/usr/bin/env bash
# Round 3, end to end: plan -> targeted generation -> merge -> LeRobot -> merge with round 2.
#
#   setsid nohup ./run_round3.sh > /dev/null 2>&1 &
#
# Detached on purpose: generation is several hours across three GPUs and must survive the shell
# that started it. Everything is logged to $LOG_DIR/round3_driver.log and the current stage is
# written to $EVAL_DIR/round3_state.json, so progress can be read without attaching to anything.
#
# Stages can be skipped individually when resuming after a failure:
#   SKIP_PLAN=1 SKIP_GEN=1 ./run_round3.sh     # only redo the merge/convert half
#
# Knobs: TOTAL_DEMOS (1000), WORKERS_PER_GPU (3), GPUS ("0 1 2").

set -euo pipefail
cd "$(dirname "$0")"
source ./round3.env
source ./env.sh

TOTAL_DEMOS=${TOTAL_DEMOS:-1000}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-3}
GPUS=${GPUS:-"0 1 2"}
PLAN="$EVAL_DIR/round3_plan.json"
DRIVER_LOG="$LOG_DIR/round3_driver.log"
STATE="$EVAL_DIR/round3_state.json"

mkdir -p "$LOG_DIR" "$EVAL_DIR" "$DATA_DIR"
exec > >(tee -a "$DRIVER_LOG") 2>&1

state() {
  printf '{"stage": "%s", "detail": "%s", "at": "%s", "pid": %d}\n' \
    "$1" "${2:-}" "$(date '+%F %T')" "$$" > "$STATE"
  echo
  echo "=== [$(date '+%F %T')] $1 ${2:+- $2}"
}

fail() { state "failed" "$1"; echo "ROUND 3 FAILED: $1" >&2; exit 1; }

echo "################ round 3 driver, started $(date '+%F %T'), pid $$"
echo "  demos     : $TOTAL_DEMOS"
echo "  workers   : $WORKERS_PER_GPU per GPU on [$GPUS]"
echo "  data      : $DATA_DIR"
echo "  logs      : $LOG_DIR"

# --- stage 1: the spawn-region plan -----------------------------------------------------------
if [[ "${SKIP_PLAN:-0}" != "1" ]]; then
  state "planning" "$TOTAL_DEMOS demos over the weak spawn regions"
  python3 plan_targeted.py --total "$TOTAL_DEMOS" --out "$PLAN" || fail "plan_targeted.py"
fi
[[ -f "$PLAN" ]] || fail "no plan at $PLAN"

# --- stage 2: targeted generation -------------------------------------------------------------
if [[ "${SKIP_GEN:-0}" != "1" ]]; then
  state "generating" "$WORKERS_PER_GPU workers per GPU on [$GPUS]"
  ./run_generate_targeted.sh "$PLAN" "$WORKERS_PER_GPU" "$GPUS" || fail "run_generate_targeted.sh"

  SHARDS=$(ls "$DATA_DIR"/gen_t*.hdf5 2>/dev/null | wc -l)
  [[ $SHARDS -gt 0 ]] || fail "generation produced no gen_t*.hdf5 shards"
  echo "generation finished with $SHARDS shards"
fi

# --- stage 3: merge, convert, and join with round 2 --------------------------------------------
state "merging" "shards -> HDF5 -> LeRobot -> merged with round 2"
./run_merge_targeted.sh || fail "run_merge_targeted.sh"

COMBINED="$DATA_DIR/${COMBINED_NAME:-g1_apple_mimic_r2r3}/lerobot"
[[ -f "$COMBINED/meta/info.json" ]] || fail "no combined dataset at $COMBINED"

state "done" "$COMBINED"
python3 - "$COMBINED" <<'PY'
import json, sys
info = json.load(open(f"{sys.argv[1]}/meta/info.json"))
print(f"  episodes : {info['total_episodes']}")
print(f"  frames   : {info['total_frames']}")
PY
echo
echo "################ round 3 complete at $(date '+%F %T')"
echo "train on: $COMBINED"
