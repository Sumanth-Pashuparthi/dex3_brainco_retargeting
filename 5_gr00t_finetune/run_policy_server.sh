#!/usr/bin/env bash
# Serve a fine-tuned checkpoint to the simulator over ZeroMQ.
#
# Same server as 1_baseline/run_policy_server.sh; the only difference is which weights it loads,
# which is the whole experiment. Pass a checkpoint directory, or a step number to pick one out of
# the current variant's output directory:
#
#   ./run_policy_server.sh 10000                        # $OUT_DIR/checkpoint-10000
#   ./run_policy_server.sh /path/to/checkpoint-5000     # any directory
#   VARIANT=r1r2 ./run_policy_server.sh 5000
#
# Takes about a minute to load the 3B model, then holds ~7 GB of VRAM. Leave it running; the
# evaluation script connects to it. Startup is complete when it prints that it is listening.

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

ARG=${1:-}
[ -n "$ARG" ] || { echo "usage: $0 <step|checkpoint_dir>" >&2; exit 1; }

# A bare number means "that step of the variant I am configured for".
if [[ "$ARG" =~ ^[0-9]+$ ]]; then
  CKPT="$OUT_DIR/checkpoint-$ARG"
else
  CKPT="$ARG"
fi

[ -f "$CKPT/config.json" ] || {
  echo "not a checkpoint: $CKPT" >&2
  echo "available under $OUT_DIR:" >&2
  ls -d "$OUT_DIR"/checkpoint-* 2>/dev/null >&2 || echo "  (none)" >&2
  exit 1; }
[ -f "$MODALITY_CFG" ] || { echo "modality config missing: $MODALITY_CFG" >&2; exit 1; }

if command -v ss >/dev/null && ss -ltn 2>/dev/null | grep -q ":${POLICY_PORT} "; then
  echo "port ${POLICY_PORT} is already in use; stop the old server or set POLICY_PORT" >&2
  exit 1
fi

# The modality config must be the one the checkpoint was trained with, or the action head will be
# handed the wrong number of joints and fail silently into garbage actions rather than erroring.
echo "serving $CKPT on ${POLICY_HOST}:${POLICY_PORT}"
cd "$GR00T_DIR"
exec "$GR00T_PY" gr00t/eval/run_gr00t_server.py \
  --modality-config-path "$MODALITY_CFG" \
  --model-path "$CKPT" \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda --host 0.0.0.0 --port "$POLICY_PORT"
