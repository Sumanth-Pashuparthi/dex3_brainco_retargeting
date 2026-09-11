#!/usr/bin/env bash
# Step 4: generate demos with Isaac Lab Mimic, spread over independent worker processes.
#
#   ./run_generate.sh [total_demos=200] [workers_per_gpu=2] [gpus="0"]
#
# G1DecoupledWBCPinkAction asserts num_envs == 1, so Mimic here cannot use --num_envs N. Throughput
# comes from separate Isaac Sim processes instead: one env each, its own seed, its own output shard.
# Budget ~10 GB of VRAM, ~14 GB of host RAM and one CPU-heavy process per worker; host RAM is what
# limits the count on this box (see the RAM guard below), not the GPUs.
#
# Each worker writes gen_w<i>.hdf5 and a live status_w<i>.json. Watch with ./status.sh, combine with
# ./run_merge.sh.

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

# Internal worker mode, re-entered through nohup by the launcher below.
#
# Start-up is the fragile part: Arena's asset library talks to the Lightwheel API and Kit resolves
# object USDs from S3 at scene creation, and either can hiccup (run_patched.py covers the first). A
# worker that dies before writing its shard is retried up to WORKER_RETRIES times; one that dies with
# a shard on disk is not, so partial data is never overwritten.
if [[ "${1:-}" == "--worker" ]]; then
  GPU=$2; WID=$3; TRIALS=$4; SEED=$5
  CAM_FLAG=$([[ "${GEN_CAMERAS:-1}" == "0" ]] && echo "" || echo "--enable_cameras")
  SHARD="$DATA_DIR/gen_w${WID}.hdf5"
  for ATTEMPT in $(seq 1 "${WORKER_RETRIES:-4}"); do
    [[ $ATTEMPT -gt 1 ]] && { echo "[worker $WID] attempt $ATTEMPT" >&2; sleep 30; }
    DOCKER_LABEL=g1_apple_mimic=gen DOCKER_NAME="g1_apple_mimic_gen_w${WID}" \
    ./arena_run.sh "$GPU" \
      "/isaac-sim/python.sh g1_apple_mimic/run_patched.py g1_apple_mimic/generate_dataset_seeded.py \
        --headless ${CAM_FLAG} --device ${GEN_DEVICE:-cuda} --num_envs 1 \
        --generation_num_trials ${TRIALS} --datagen_seed ${SEED} \
        --max_num_failures ${MAX_NUM_FAILURES:-50} \
        --rerenders_on_reset ${RERENDERS_ON_RESET:-3} \
        --input_file ${ANNOTATED_HDF5} \
        --output_file ${C_DATA_DIR}/gen_w${WID}.hdf5 \
        --status_file ${C_EVAL_DIR}/status_w${WID}.json \
        ${ENV_ARGS}" > "$LOG_DIR/gen_w${WID}.attempt${ATTEMPT}.log" 2>&1 && RC=0 || RC=$?
    ln -sfn "gen_w${WID}.attempt${ATTEMPT}.log" "$LOG_DIR/gen_w${WID}.log"
    [[ $RC -eq 0 || -f "$SHARD" ]] && exit $RC
    echo "[worker $WID] exited $RC before writing $SHARD" >&2
  done
  exit 1
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

# RAM guard. A worker holds ~14 GB of host RAM at steady state and more while it starts, and this
# box has 7 GB of swap: twice, launching a full set of workers on top of an unrelated training job
# took the machine past MemAvailable=0 into a hard hang (journal: "Under memory pressure, flushing
# caches", then nothing). Before each launch, wait until MemAvailable >= MIN_FREE_GB plus one
# worker's worth; if it does not get there within RAM_WAIT_S, stop launching and say so. Workers
# already running keep going; the plan is per-worker so the total simply comes out short.
MIN_FREE_GB=${MIN_FREE_GB:-40}
WORKER_GB=${WORKER_GB:-20}
RAM_WAIT_S=${RAM_WAIT_S:-600}
mem_avail_gb() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }

WID=0
for GPU in "${GPU_ARR[@]}"; do
  for _ in $(seq 1 "$WPG"); do
    N=$PER; [[ $WID -lt $REM ]] && N=$((PER + 1))
    NEED=$((MIN_FREE_GB + WORKER_GB)); WAITED=0
    while [[ $(mem_avail_gb) -lt $NEED ]]; do
      if [[ $WAITED -ge $RAM_WAIT_S ]]; then
        echo "  RAM guard: MemAvailable=$(mem_avail_gb) GB < $NEED GB after ${WAITED}s; not launching workers $WID..$((NW - 1))." >&2
        echo "  Re-run later with a smaller workers_per_gpu, or free memory (check: ps -eo rss,args --sort=-rss | head)." >&2
        break 2
      fi
      [[ $WAITED -eq 0 ]] && echo "  RAM guard: MemAvailable=$(mem_avail_gb) GB, waiting for $NEED GB before worker $WID"
      sleep 30; WAITED=$((WAITED + 30))
    done
    SHARD="$DATA_DIR/gen_w${WID}.hdf5"
    [[ -f "$SHARD" ]] && mv "$SHARD" "$SHARD.bak.$(date +%s)"
    nohup "$0" --worker "$GPU" "$WID" "$N" "$((BASE_SEED + WID))" \
      > "$LOG_DIR/launcher_w${WID}.log" 2>&1 &
    echo "  worker $WID on GPU $GPU: $N demos, seed $((BASE_SEED + WID)) (pid $!, MemAvailable $(mem_avail_gb) GB)"
    WID=$((WID + 1))
    # Isaac Sim spikes ~20 GB RSS while it warms the shader cache; starting them together thrashes.
    sleep "${STAGGER_S:-45}"
  done
done

echo
echo "launched $WID of $NW workers. Watch with ./status.sh, combine with ./run_merge.sh"
