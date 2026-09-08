# Shared configuration for every script in this folder. Sourced, not executed.
#
# Override any of these from the shell before running, e.g.
#   WORK_DIR=/data/g1 ./setup.sh
#
# Defaults install everything under ~/g1_baseline so nothing lands outside one directory.

WORK_DIR=${WORK_DIR:-$HOME/g1_baseline}

# Pinned versions. These are the exact ones this baseline was measured on.
ISAAC_SIM_IMAGE=${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.0-dev2}
ARENA_BRANCH=${ARENA_BRANCH:-release/0.2.1}
ARENA_REPO=${ARENA_REPO:-https://github.com/isaac-sim/IsaacLab-Arena.git}
GROOT_COMMIT=${GROOT_COMMIT:-4b1dca9}
GROOT_REPO=${GROOT_REPO:-https://github.com/NVIDIA/Isaac-GR00T.git}
CKPT_REPO=${CKPT_REPO:-nvidia/GN1x-Tuned-Arena-G1-Static-PickNPlace}

# Layout produced by setup.sh.
ARENA_DIR=${ARENA_DIR:-$WORK_DIR/IsaacLab-Arena}
GROOT_DIR=${GROOT_DIR:-$WORK_DIR/Isaac-GR00T}
MODELS_DIR=${MODELS_DIR:-$WORK_DIR/models}
EVAL_DIR=${EVAL_DIR:-$WORK_DIR/eval}
CKPT_NAME=${CKPT_NAME:-GN1x-Tuned-Arena-G1-Static-PickNPlace}

# Inference server. 5556 rather than 5555 so it does not collide with an existing GR00T server.
POLICY_HOST=${POLICY_HOST:-localhost}
POLICY_PORT=${POLICY_PORT:-5556}

# Paths as seen from inside the container.
CONTAINER_WORKDIR=/workspaces/isaaclab_arena
DOCKER_IMAGE=${DOCKER_IMAGE:-isaaclab_arena:latest}

# The task under evaluation, and the stock Dex3-1 embodiment. Kept here so the evaluation and the
# video recording cannot drift apart.
POLICY_TYPE=isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy
POLICY_CONFIG=isaaclab_arena_gr00t/policy/config/g1_static_apple_gr00t_closedloop_config.yaml
TASK_ARGS="galileo_g1_static_pick_and_place \
  --object apple_01_objaverse_robolab \
  --destination clay_plates_hot3d_robolab \
  --embodiment g1_wbc_agile_joint"
