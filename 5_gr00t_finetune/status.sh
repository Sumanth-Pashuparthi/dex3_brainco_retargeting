#!/usr/bin/env bash
# Summarise a running or finished fine-tune: step, loss, throughput, ETA, saved checkpoints.
#
#   ./status.sh                  # the current VARIANT
#   VARIANT=r1r2 ./status.sh
#
# Reads the checkpoint's own trainer_state.json rather than scraping the log, so it works on a
# finished run too.

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

LOG="$LOG_DIR/finetune_${RUN_NAME}.log"

echo "variant $VARIANT  ->  $OUT_DIR"
if pgrep -f "launch_finetune.py.*--output-dir $OUT_DIR" >/dev/null; then
  echo "state:  RUNNING (pids: $(pgrep -df ' ' -f "launch_finetune.py.*--output-dir $OUT_DIR" | tr '\n' ' '))"
else
  echo "state:  not running"
fi

LAST=$(ls -d "$OUT_DIR"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1 || true)
if [ -z "$LAST" ]; then
  echo "no checkpoints yet"
else
  echo "checkpoints: $(ls -d "$OUT_DIR"/checkpoint-* | sed 's/.*checkpoint-//' | sort -n | tr '\n' ' ')"
  python3 - "$OUT_DIR/checkpoint-$LAST/trainer_state.json" <<'PY'
import json, sys
st = json.load(open(sys.argv[1]))
hist = [h for h in st["log_history"] if "loss" in h]
step, mx = st.get("global_step", 0), st.get("max_steps", 0)
print(f"progress:    {step}/{mx} steps ({100*step/mx:.1f}%)" if mx else f"progress: {step} steps")
if hist:
    tail = hist[-20:]
    print(f"loss:        {hist[0]['loss']:.4f} at step {hist[0]['step']} "
          f"-> {sum(h['loss'] for h in tail)/len(tail):.4f} (mean of last {len(tail)} logs)")
PY
fi

if [ -f "$LOG" ]; then
  echo "log:         $LOG"
  # The trainer's own progress line carries the rate and its ETA; the last one is current.
  grep -aoE '[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+, +[0-9.]+s?/?i?t?[^]]*\]' "$LOG" | tail -1 \
    | sed 's/^/rate:        /' || true
  echo "--- last 3 log lines ---"
  tail -n 3 "$LOG"
fi
