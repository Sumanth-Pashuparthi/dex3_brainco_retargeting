# Shared configuration for every script in this folder. Sourced, not executed.
#
# Defaults match the other steps. Installation lives in 1_baseline/setup.sh and 2_retargeting/;
# nothing here installs anything.

WORK_DIR=${WORK_DIR:-$HOME/g1_baseline}

# Pinned versions. These are the exact ones this asset was built with.
ISAAC_SIM_IMAGE=${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.0-dev2}
ARENA_BRANCH=${ARENA_BRANCH:-release/0.2.1}
ARENA_REPO=${ARENA_REPO:-https://github.com/isaac-sim/IsaacLab-Arena.git}

ARENA_DIR=${ARENA_DIR:-$WORK_DIR/IsaacLab-Arena}
MODELS_DIR=${MODELS_DIR:-$WORK_DIR/models}
EVAL_DIR=${EVAL_DIR:-$WORK_DIR/eval}

CONTAINER_WORKDIR=/workspaces/isaaclab_arena
DOCKER_IMAGE=${DOCKER_IMAGE:-isaaclab_arena:latest}

# The inference server started by 1_baseline/run_policy_server.sh.
POLICY_HOST=${POLICY_HOST:-localhost}
POLICY_PORT=${POLICY_PORT:-5556}

POLICY_TYPE=isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy
POLICY_CONFIG=isaaclab_arena_gr00t/policy/config/g1_static_apple_gr00t_closedloop_config.yaml

# The task. Identical to the baseline in every argument except --embodiment, which is the point.
TASK="galileo_g1_static_pick_and_place \
  --object apple_01_objaverse_robolab \
  --destination clay_plates_hot3d_robolab"
EMBODIMENT=${EMBODIMENT:-g1_wbc_agile_joint_brainco}
BASELINE_EMBODIMENT=g1_wbc_agile_joint
