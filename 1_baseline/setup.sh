#!/usr/bin/env bash
# One-time installation for the baseline evaluation.
#
# Pulls the Isaac Sim base image, clones and builds IsaacLab-Arena, clones Isaac-GR00T at the
# pinned commit, downloads the policy checkpoint, and points the policy config at it.
#
# Every step is idempotent and skipped if already done, so it is safe to re-run after a failure.
# Nothing is installed outside $WORK_DIR (default ~/g1_baseline) except the two Docker images.
#
# Usage: ./setup.sh

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------- prerequisites
step "Checking prerequisites"
missing=0
for tool in docker git; do
  have "$tool" || { echo "  MISSING: $tool"; missing=1; }
done
if ! have huggingface-cli && ! have hf; then
  echo "  MISSING: huggingface-cli  (pip install -U 'huggingface_hub[cli]')"
  missing=1
fi
have uv || { echo "  MISSING: uv  (curl -LsSf https://astral.sh/uv/install.sh | sh)"; missing=1; }
[[ $missing -eq 0 ]] || { echo "Install the tools above and re-run."; exit 1; }

if ! docker info 2>/dev/null | grep -qi nvidia; then
  echo "  WARNING: the NVIDIA container runtime was not detected in 'docker info'."
  echo "  The simulator needs it. See nvidia-container-toolkit if the run step fails."
fi
echo "  ok"

# ---------------------------------------------------------------- isaac sim image
step "Isaac Sim base image: $ISAAC_SIM_IMAGE"
if docker image inspect "$ISAAC_SIM_IMAGE" >/dev/null 2>&1; then
  echo "  already present"
else
  echo "  This is a ~29 GB pull from NGC and needs 'docker login nvcr.io' first."
  echo "  Username is exactly \$oauthtoken, password is your NGC API key."
  # A single layer here is ~9 GB and can look stalled for minutes; retry rather than give up.
  for attempt in 1 2 3; do
    if docker pull "$ISAAC_SIM_IMAGE"; then break; fi
    echo "  pull attempt $attempt failed, retrying"
    [[ $attempt -lt 3 ]] || { echo "  giving up"; exit 1; }
  done
fi

# ---------------------------------------------------------------- arena
step "IsaacLab-Arena ($ARENA_BRANCH)"
mkdir -p "$WORK_DIR"
if [[ -d "$ARENA_DIR/.git" ]]; then
  echo "  already cloned at $ARENA_DIR"
else
  # release/0.2.1, not main: main's Dockerfile references an internal NVIDIA registry that is not
  # publicly pullable. This branch uses the public Isaac Sim image above.
  git clone -b "$ARENA_BRANCH" --recursive "$ARENA_REPO" "$ARENA_DIR"
fi
git -C "$ARENA_DIR" submodule update --init --recursive

step "Arena Docker image"
if docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
  echo "  already built"
else
  echo "  Building, ~20 minutes."
  docker build --pull \
    --progress=plain \
    --build-arg WORKDIR="$CONTAINER_WORKDIR" \
    --build-arg INSTALL_GROOT=true \
    -t "$DOCKER_IMAGE" \
    --file "$ARENA_DIR/docker/Dockerfile.isaaclab_arena" \
    "$ARENA_DIR"
fi

# ---------------------------------------------------------------- gr00t
step "Isaac-GR00T ($GROOT_COMMIT), for the inference server"
# The server runs on the host, outside the container, because it needs its own Python environment.
# Arena vendors Isaac-GR00T as a submodule too, but that copy is not pinned to the commit this
# checkpoint was validated against, so clone it separately.
if [[ -d "$GROOT_DIR/.git" ]]; then
  echo "  already cloned at $GROOT_DIR"
else
  git clone "$GROOT_REPO" "$GROOT_DIR"
fi
git -C "$GROOT_DIR" fetch --all --quiet
git -C "$GROOT_DIR" checkout --quiet "$GROOT_COMMIT"
echo "  at $(git -C "$GROOT_DIR" log -1 --format='%h %s')"

step "Isaac-GR00T Python environment"
(cd "$GROOT_DIR" && uv sync)

# ---------------------------------------------------------------- checkpoint
step "Policy checkpoint: $CKPT_REPO"
mkdir -p "$MODELS_DIR" "$EVAL_DIR"
if [[ -f "$MODELS_DIR/$CKPT_NAME/config.json" ]]; then
  echo "  already downloaded"
else
  echo "  ~14 GB from Hugging Face."
  HF=$(command -v hf || command -v huggingface-cli)
  "$HF" download "$CKPT_REPO" --local-dir "$MODELS_DIR/$CKPT_NAME"
fi

# ---------------------------------------------------------------- config
step "Pointing the policy config at the checkpoint"
CFG="$ARENA_DIR/$POLICY_CONFIG"
[[ -f "$CFG" ]] || { echo "  policy config not found at $CFG"; exit 1; }
# /models is where arena_run.sh mounts $MODELS_DIR inside the container.
if grep -q "model_path:.*/models/$CKPT_NAME" "$CFG"; then
  echo "  already set"
else
  sed -i "s|^\(\s*model_path:\).*|\1 /models/$CKPT_NAME|" "$CFG"
  echo "  set to /models/$CKPT_NAME"
fi
grep -n "model_path:" "$CFG"

cat <<EOF

Setup complete.

  Arena         $ARENA_DIR
  Isaac-GR00T   $GROOT_DIR
  checkpoint    $MODELS_DIR/$CKPT_NAME
  output        $EVAL_DIR

Next, in two terminals:
  ./run_policy_server.sh          # leave running
  ./run_eval.sh 100               # expect success_rate 0.65
EOF
