#!/usr/bin/env bash
# Step 4: generate demos with Isaac Lab Mimic, spread over independent worker processes.
#
#   ./run_generate.sh [total_demos=200] [workers_per_gpu=2] [gpus="0"]
#
# G1DecoupledWBCPinkAction asserts num_envs == 1, so Mimic here cannot use --num_envs N. Throughput
# comes from separate Isaac Sim processes instead: one env each, its own seed, its own output shard.
# Budget ~8-12 GB of VRAM and one CPU-heavy process per worker.
#
# Each worker writes gen_w<i>.hdf5 and a live status_w<i>.json. Watch with ./status.sh, combine with
# ./run_merge.sh.

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

# Internal worker mode, re-entered through nohup by the launcher below.
if [[ "${1:-}" == "--worker" ]]; then
  GPU=$2; WID=$3; TRIALS=$4; SEED=$5
  CAM_FLAG=$([[ "${GEN_CAMERAS:-1}" == "0" ]] && echo "" || echo "--enable_cameras")
  DOCKER_LABEL=g1_apple_mimic=gen DOCKER_NAME="g1_apple_mimic_gen_w${WID}" \
  exec ./arena_run.sh "$GPU" \
    "/isaac-sim/python.sh g1_apple_mimic/generate_dataset_seeded.py \
      --headless ${CAM_FLAG} --device ${GEN_DEVICE:-cuda} --num_envs 1 \
      --generation_num_trials ${TRIALS} --datagen_seed ${SEED} \
      --max_num_failures ${MAX_NUM_FAILURES:-50} \
      --input_file ${ANNOTATED_HDF5} \
      --output_file ${C_DATA_DIR}/gen_w${WID}.hdf5 \
      --status_file ${C_EVAL_DIR}/status_w${WID}.json \
      ${ENV_ARGS}" > "$LOG_DIR/gen_w${WID}.log" 2>&1
fi

TOTAL=${1:-200}
WPG=${2:-2}
GPUS=${3:-0}
BASE_SEED=${BASE_SEED:-1000}

[[ -f "$DATA_DIR/$(basename "$ANNOTATED_HDF5")" ]] \
  || { echo "annotated file missing. Run ./run_annotate.sh first." >&2; exit 1; }
docker ps --filter label=g1_apple_mimic=gen -q | grep -q . \
  && { echo "workers already running. Use ./stop.sh first." >&2; exit 1; }

read -ra GPU_ARR <<< "$GPUS"
NW=$(( ${#GPU_ARR[@]} * WPG ))
PER=$(( TOTAL / NW ))
REM=$(( TOTAL - PER * NW ))

echo "total=$TOTAL workers=$NW ($WPG per GPU on: $GPUS) -> $PER each (+1 for the first $REM)"
rm -f "$EVAL_DIR"/status_w*.json
printf '{"total": %d, "workers": %d, "gpus": "%s", "started": "%s"}\n' \
  "$TOTAL" "$NW" "$GPUS" "$(date '+%F %T')" > "$EVAL_DIR/plan.json"

WID=0
for GPU in "${GPU_ARR[@]}"; do
  for _ in $(seq 1 "$WPG"); do
    N=$PER; [[ $WID -lt $REM ]] && N=$((PER + 1))
    SHARD="$DATA_DIR/gen_w${WID}.hdf5"
    [[ -f "$SHARD" ]] && mv "$SHARD" "$SHARD.bak.$(date +%s)"
    nohup "$0" --worker "$GPU" "$WID" "$N" "$((BASE_SEED + WID))" \
      > "$LOG_DIR/launcher_w${WID}.log" 2>&1 &
    echo "  worker $WID on GPU $GPU: $N demos, seed $((BASE_SEED + WID)) (pid $!)"
    WID=$((WID + 1))
    # Isaac Sim spikes ~20 GB RSS while it warms the shader cache; starting them together thrashes.
    sleep "${STAGGER_S:-45}"
  done
done

echo
echo "launched. Watch with ./status.sh, combine with ./run_merge.sh"
