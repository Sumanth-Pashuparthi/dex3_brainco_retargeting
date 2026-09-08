#!/usr/bin/env bash
# Live view of a generation run: GPUs, containers, per-worker counters, demos on disk.

set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh

echo "===== GPUs ====="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv

echo
echo "===== containers ====="
docker ps --filter label=g1_apple_mimic --format 'table {{.Names}}\t{{.Status}}'

echo
echo "===== workers ====="
[[ -f "$EVAL_DIR/plan.json" ]] && cat "$EVAL_DIR/plan.json"
python3 - "$EVAL_DIR" <<'PY'
import glob, json, os, sys
tot_s = tot_a = tot_t = 0
paths = sorted(glob.glob(os.path.join(sys.argv[1], "status_w*.json")),
               key=lambda s: int(s.rsplit("_w", 1)[1].split(".")[0]))
for p in paths:
    try:
        s = json.load(open(p))
    except Exception as e:
        print(os.path.basename(p), "unreadable:", e)
        continue
    rate = 100.0 * s["success"] / s["attempts"] if s["attempts"] else 0.0
    print(f"{os.path.basename(p):18s} {s['state']:9s} {s['success']:4d}/{s['target']:<4d} succ  "
          f"attempts={s['attempts']:<5d} ({rate:5.1f}%)  {s['elapsed_s']/60:6.1f} min  seed={s.get('seed')}")
    tot_s += s["success"]; tot_a += s["attempts"]; tot_t += s["target"]
if tot_t:
    rate = 100.0 * tot_s / tot_a if tot_a else 0.0
    print(f"{'TOTAL':18s} {'':9s} {tot_s:4d}/{tot_t:<4d} succ  attempts={tot_a} ({rate:.1f}%)")
PY

echo
echo "===== demos on disk ====="
python3 dataset_stats.py --brief "$DATA_DIR"/*.hdf5 2>/dev/null || echo "none yet"
