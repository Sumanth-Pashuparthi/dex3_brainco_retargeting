# Shared configuration for every script in this folder. Sourced, not executed.
#
# Defaults match the other steps, so if you already ran an earlier setup.sh this one finds
# everything in place and does nothing.

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

# Where the converted robot is written, relative to the Arena checkout. The embodiment definition in
# step 3 spawns exactly this path, so changing it means changing BRAINCO_G1_USD there too.
ASSET_SUBDIR=assets/brainco/g1_29dof_with_brainco_hand
ASSET_NAME=g1_29dof_with_brainco_hand.usda

# Staging path for the URDF. The converter resolves mesh filenames relative to the URDF, so the
# whole urdf/ directory is copied somewhere the container can see it. /tmp is bind-mounted.
URDF_STAGING=${URDF_STAGING:-/tmp/brainco_asset}

# The source hand's URDF, for the geometry comparison. It ships inside Arena's Isaac-GR00T submodule
# as part of the whole-body-control model data.
DEX3_URDF=${DEX3_URDF:-$ARENA_DIR/submodules/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/control/robot_model/model_data/g1/g1_29dof_with_hand.urdf}
