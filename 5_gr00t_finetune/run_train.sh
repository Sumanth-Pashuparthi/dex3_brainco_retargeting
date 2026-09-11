#!/usr/bin/env bash
# Fine-tune GR00T N1.7 on the Mimic-generated Revo2 demonstrations.
#
#   ./run_train.sh                  # VARIANT=r2r3, the run the report's final numbers come from
#   VARIANT=r1r2 ./run_train.sh     # the first fine-tune, 1668 demos, 5k steps
#   VARIANT=r2r3-base ./run_train.sh   # same data from the general N1.7 base instead of the expert
#   SMOKE=1 ./run_train.sh          # 1 GPU, 3 steps, foreground: checks the chain, not the result
#
# Resumable: re-running with the same VARIANT picks up the last checkpoint in OUT_DIR, because
# the HF trainer is told resume_from_checkpoint and skips the dataloader fast-forward. That is how
# the 10k-step run was completed across three sessions.
#
# Takes about 14 h for 10k steps on three RTX 5090s. Runs in the background and logs to
# $LOG_DIR/finetune_$RUN_NAME.log; ./status.sh summarises progress.

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

if [ "${SMOKE:-0}" = "1" ]; then
  GPUS=${SMOKE_GPU:-0}; NGPU=1; MAX_STEPS=3; GLOBAL_BATCH=4
  SAVE_STEPS=2; SAVE_TOTAL_LIMIT=1; WORKERS=2
  RUN_NAME="${RUN_NAME}_smoke"; OUT_DIR="$CKPT_ROOT/$RUN_NAME"
fi

LOG="$LOG_DIR/finetune_${RUN_NAME}.log"

# Preflight. Each of these has cost a multi-hour run at some point, so they are checked up front
# rather than discovered after the model has loaded.
[ -f "$DATASET/meta/info.json" ] || {
  echo "dataset missing: $DATASET" >&2
  echo "  build it with 4_mimic_datagen/run_generate.sh then run_convert.sh, or point DATASET at it" >&2
  exit 1; }
[ -x "$GR00T_PY" ] || {
  echo "GR00T venv python missing: $GR00T_PY  (run 1_baseline/setup.sh, or set GR00T_DIR)" >&2
  exit 1; }
[ -f "$MODALITY_CFG" ] || {
  echo "modality config missing: $MODALITY_CFG  (run 1_baseline/setup.sh, or set ARENA_DIR)" >&2
  exit 1; }
if pgrep -f "launch_finetune.py.*--output-dir $OUT_DIR" >/dev/null; then
  echo "a fine-tune is already writing to $OUT_DIR; stop it first" >&2
  exit 1
fi

# GLOBAL_BATCH is split evenly across ranks, so it has to divide the GPU count.
if [ $((GLOBAL_BATCH % NGPU)) -ne 0 ]; then
  echo "GLOBAL_BATCH=$GLOBAL_BATCH is not divisible by $NGPU GPUs" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$LOG_DIR"

RESUMING=no
if compgen -G "$OUT_DIR/checkpoint-*" >/dev/null; then RESUMING=yes; fi

cat <<EOF
fine-tune GR00T N1.7
  variant     $VARIANT
  dataset     $DATASET
  base model  $BASE_MODEL
  tuning      $TUNE_ARGS
  steps       $MAX_STEPS   global batch $GLOBAL_BATCH ($((GLOBAL_BATCH / NGPU))/GPU on $NGPU GPUs)
  lr          $LR
  output      $OUT_DIR    (resuming: $RESUMING)
  log         $LOG
EOF

CMD=( "$GR00T_PY" -m torch.distributed.run --nproc_per_node="$NGPU" --standalone
  gr00t/experiment/launch_finetune.py
  --base-model-path "$BASE_MODEL"
  --dataset-path "$DATASET"
  --output-dir "$OUT_DIR"
  --experiment-name "$RUN_NAME"
  --modality-config-path "$MODALITY_CFG"
  --embodiment-tag "$EMBODIMENT_TAG"
  --global-batch-size "$GLOBAL_BATCH"
  --max-steps "$MAX_STEPS"
  --num-gpus "$NGPU"
  --save-steps "$SAVE_STEPS"
  --save-total-limit "$SAVE_TOTAL_LIMIT"
  --learning-rate "$LR"
  --dataloader-num-workers "$WORKERS"
  # The head camera is the only sensor; jitter it hard so the policy does not key on sim lighting.
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
  --no-use-wandb )
# shellcheck disable=SC2206  # TUNE_ARGS is deliberately word-split into separate flags
CMD+=( ${TUNE_ARGS} )
[ "${SMOKE:-0}" = "1" ] && CMD+=( --save-only-model )

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export HF_HUB_ENABLE_HF_TRANSFER=0

cd "$GR00T_DIR"
# Appended, not truncated: a resumed run must not erase the loss history of the segment before it.
{ echo "# $(date '+%F %T') variant=$VARIANT resuming=$RESUMING"
  echo "# CUDA_VISIBLE_DEVICES=$GPUS ${CMD[*]}"; } >> "$LOG"

if [ "${SMOKE:-0}" = "1" ]; then
  "${CMD[@]}" 2>&1 | tee -a "$LOG"
  echo "smoke fine-tune finished; see $OUT_DIR"
else
  nohup "${CMD[@]}" >> "$LOG" 2>&1 &
  echo "launched pid $! -- progress: ./status.sh"
fi
