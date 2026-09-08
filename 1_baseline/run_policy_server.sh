#!/usr/bin/env bash
# Start the GR00T N1.7 inference server on the host.
#
# The server runs outside the container, in Isaac-GR00T's own uv environment, and the simulator
# talks to it over ZeroMQ. Startup takes ~1 minute while the 3B model loads onto the GPU; it then
# holds about 6.6 GB of VRAM. Leave it running for the whole session.
#
# Usage: ./run_policy_server.sh

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

MODALITY="$ARENA_DIR/isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_gr00t_n_1_7_config.py"
CKPT="$MODELS_DIR/$CKPT_NAME"

[[ -d "$CKPT" ]] || { echo "Checkpoint not found: $CKPT. Run ./setup.sh first." >&2; exit 1; }
[[ -f "$MODALITY" ]] || { echo "Modality config not found: $MODALITY. Run ./setup.sh first." >&2; exit 1; }
[[ -d "$GROOT_DIR" ]] || { echo "Isaac-GR00T not found: $GROOT_DIR. Run ./setup.sh first." >&2; exit 1; }

if command -v ss >/dev/null && ss -ltn 2>/dev/null | grep -q ":${POLICY_PORT} "; then
  echo "Something is already listening on port ${POLICY_PORT}."
  echo "If that is an older server, stop it; otherwise set POLICY_PORT to a free port."
  exit 1
fi

echo "Serving $CKPT_NAME on ${POLICY_HOST}:${POLICY_PORT}"
cd "$GROOT_DIR"
exec uv run python gr00t/eval/run_gr00t_server.py \
  --modality-config-path "$MODALITY" \
  --model-path "$CKPT" \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda --host 0.0.0.0 --port "$POLICY_PORT"
