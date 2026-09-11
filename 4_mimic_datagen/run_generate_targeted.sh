#!/usr/bin/env bash
# Step 4b: generate demos aimed at the spawn regions where the trained policy fails.
#
#   ./run_generate_targeted.sh <plan.json> [workers_per_gpu=2] [gpus="0"]
#
# Same machinery as run_generate.sh -- one Isaac Sim per worker, own seed, own shard, retries on a
# start-up death, RAM guard before each launch -- except each worker samples the apple from one
# rectangle of the spawn box instead of the full symmetric +/-APPLE_XY_RANGE_M. The plan comes from
# plan_targeted.py, which sizes each rectangle's demo count from the eval success rate there.
#
# Shards are gen_t<i>.hdf5 so a targeted round never overwrites the round-2 gen_w* shards.
# Watch with ./status.sh, then ./run_merge_targeted.sh.
#
#   DRY_RUN=1 ./run_generate_targeted.sh plan.json    print the worker table and exit

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

# Isaac Sim creates the output file during start-up, so a crash before any demo is written leaves a
# ~96-byte stub. "the shard exists" therefore cannot mean "the worker succeeded": a real shard holds
# at least one ~150 MB demo. Without this the retry loop treats a start-up crash as done and
# silently drops that worker's whole allocation.
shard_has_demos() {
  local f=$1
  [[ -f "$f" ]] || return 1
  [[ $(stat -c %s "$f" 2>/dev/null || echo 0) -gt 1048576 ]]
}

# Internal worker mode, re-entered through nohup by the launcher below.
if [[ "${1:-}" == "--worker" ]]; then
  GPU=$2; WID=$3; TRIALS=$4; SEED=$5; XMIN=$6; XMAX=$7; YMIN=$8; YMAX=$9
  CAM_FLAG=$([[ "${GEN_CAMERAS:-1}" == "0" ]] && echo "" || echo "--enable_cameras")
  SHARD="$DATA_DIR/gen_t${WID}.hdf5"
  # The spawn-box flags must trail ${ENV_ARGS}: example_environment is an argparse subparser, so
  # only flags placed after the environment name reach the environment's own add_cli_args.
  for ATTEMPT in $(seq 1 "${WORKER_RETRIES:-4}"); do
    [[ $ATTEMPT -gt 1 ]] && { echo "[worker $WID] attempt $ATTEMPT" >&2; sleep 30; }
    DOCKER_LABEL=g1_apple_mimic=gen DOCKER_NAME="g1_apple_mimic_gen_t${WID}" \
    ./arena_run.sh "$GPU" \
      "/isaac-sim/python.sh g1_apple_mimic/run_patched.py g1_apple_mimic/generate_dataset_seeded.py \
        --headless ${CAM_FLAG} --device ${GEN_DEVICE:-cuda} --num_envs 1 \
        --generation_num_trials ${TRIALS} --datagen_seed ${SEED} \
        --max_num_failures ${MAX_NUM_FAILURES:-50} \
        --rerenders_on_reset ${RERENDERS_ON_RESET:-3} \
        --input_file ${ANNOTATED_HDF5} \
        --output_file ${C_DATA_DIR}/gen_t${WID}.hdf5 \
        --status_file ${C_EVAL_DIR}/status_t${WID}.json \
        ${ENV_ARGS} \
        --apple_x_off_min_m ${XMIN} --apple_x_off_max_m ${XMAX} \
        --apple_y_off_min_m ${YMIN} --apple_y_off_max_m ${YMAX}" \
        > "$LOG_DIR/gen_t${WID}.attempt${ATTEMPT}.log" 2>&1 && RC=0 || RC=$?
    ln -sfn "gen_t${WID}.attempt${ATTEMPT}.log" "$LOG_DIR/gen_t${WID}.log"
    [[ $RC -eq 0 ]] || shard_has_demos "$SHARD" && exit $RC
    rm -f "$SHARD"
    echo "[worker $WID] exited $RC before writing $SHARD" >&2
  done
  exit 1
fi

PLAN=${1:?usage: ./run_generate_targeted.sh <plan.json> [workers_per_gpu] [gpus]}
WPG=${2:-2}
GPUS=${3:-0}
BASE_SEED=${BASE_SEED:-3000}

[[ -f "$PLAN" ]] || { echo "no plan at $PLAN. Run plan_targeted.py first." >&2; exit 1; }
[[ -f "$DATA_DIR/$(basename "$ANNOTATED_HDF5")" ]] \
  || { echo "annotated file missing. Run ./run_annotate.sh first." >&2; exit 1; }
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  docker ps --filter label=g1_apple_mimic=gen -q | grep -q . \
    && { echo "workers already running. Use ./stop.sh first." >&2; exit 1; }
fi

read -ra GPU_ARR <<< "$GPUS"
NW=$(( ${#GPU_ARR[@]} * WPG ))

# One line per worker: trials x_min x_max y_min y_max, offsets in metres. Each region is split
# across NW workers so every GPU gets a slice of every region rather than one region each.
mapfile -t WORK < <(NW="$NW" python3 - "$PLAN" <<'PY'
import json, os, sys
plan = json.load(open(sys.argv[1]))
nw = int(os.environ["NW"])
rows = []
for region in plan["regions"]:
    demos = int(region["demos"])
    if demos <= 0:
        continue
    # Clamp the sentinel-widened outer bounds back to the physical box.
    x = [max(-9.0, min(9.0, v)) for v in region["x_off_cm"]]
    y = [max(-9.0, min(9.0, v)) for v in region["y_off_cm"]]
    base, rem = divmod(demos, nw)
    for k in range(nw):
        n = base + (1 if k < rem else 0)
        if n:
            rows.append(f"{n} {x[0]/100:.4f} {x[1]/100:.4f} {y[0]/100:.4f} {y[1]/100:.4f}")
print("\n".join(rows))
PY
)

TOTAL=$(printf '%s\n' "${WORK[@]}" | awk '{s+=$1} END {print s}')
echo "plan    : $PLAN"
echo "workers : $NW concurrent ($WPG per GPU on: $GPUS), ${#WORK[@]} shards in waves"
printf '%s\n' "${WORK[@]}" | awk '{printf "  %4d demos  x=[%+.3f,%+.3f] y=[%+.3f,%+.3f] m\n", $1,$2,$3,$4,$5}'
echo "total   : $TOTAL demos"
if [[ "${DRY_RUN:-0}" == "1" ]]; then echo; echo "DRY_RUN=1, nothing launched."; exit 0; fi

rm -f "$EVAL_DIR"/status_t*.json
printf '{"plan": "%s", "total": %d, "workers": %d, "gpus": "%s", "started": "%s"}\n' \
  "$PLAN" "$TOTAL" "$NW" "$GPUS" "$(date '+%F %T')" > "$EVAL_DIR/plan_targeted.json"

# RAM guard, as in run_generate.sh: a worker holds ~14 GB at steady state and spikes to ~20 GB
# while Isaac Sim warms its shader cache, and this box has little swap.
MIN_FREE_GB=${MIN_FREE_GB:-40}
WORKER_GB=${WORKER_GB:-20}
RAM_WAIT_S=${RAM_WAIT_S:-600}
mem_avail_gb() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }

WID=0
IDX=0
while [[ $IDX -lt ${#WORK[@]} ]]; do
  PIDS=()
  for GPU in "${GPU_ARR[@]}"; do
    for _ in $(seq 1 "$WPG"); do
      [[ $IDX -lt ${#WORK[@]} ]] || break
      read -ra SPEC <<< "${WORK[$IDX]}"
      NEED=$((MIN_FREE_GB + WORKER_GB)); WAITED=0
      while [[ $(mem_avail_gb) -lt $NEED ]]; do
        if [[ $WAITED -ge $RAM_WAIT_S ]]; then
          echo "  RAM guard: MemAvailable=$(mem_avail_gb) GB < $NEED GB after ${WAITED}s; stopping launches at worker $WID." >&2
          echo "  Re-run later with a smaller workers_per_gpu, or free memory." >&2
          break 3
        fi
        [[ $WAITED -eq 0 ]] && echo "  RAM guard: MemAvailable=$(mem_avail_gb) GB, waiting for $NEED GB before worker $WID"
        sleep 30; WAITED=$((WAITED + 30))
      done
      SHARD="$DATA_DIR/gen_t${WID}.hdf5"
      [[ -f "$SHARD" ]] && mv "$SHARD" "$SHARD.bak.$(date +%s)"
      nohup "$0" --worker "$GPU" "$WID" "${SPEC[0]}" "$((BASE_SEED + WID))" \
        "${SPEC[1]}" "${SPEC[2]}" "${SPEC[3]}" "${SPEC[4]}" \
        > "$LOG_DIR/launcher_t${WID}.log" 2>&1 &
      PIDS+=("$!")
      echo "  worker $WID on GPU $GPU: ${SPEC[0]} demos, seed $((BASE_SEED + WID)), x=[${SPEC[1]},${SPEC[2]}] y=[${SPEC[3]},${SPEC[4]}] (MemAvailable $(mem_avail_gb) GB)"
      WID=$((WID + 1)); IDX=$((IDX + 1))
      sleep "${STAGGER_S:-45}"
    done
  done
  [[ ${#PIDS[@]} -gt 0 ]] || break
  echo "  waiting on ${#PIDS[@]} workers..."
  wait "${PIDS[@]}" || echo "  note: a worker exited non-zero, see $LOG_DIR/gen_t*.log"
done

echo
echo "launched $WID workers. Shards: $DATA_DIR/gen_t*.hdf5"
echo "next: ./run_merge_targeted.sh"
