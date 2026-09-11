# Shared configuration for every script in this folder. Sourced, not executed.
#
# Defaults match the other steps. Isaac Sim, the Arena checkout and the GR00T venv are installed by
# 1_baseline/setup.sh; the Revo2 asset and embodiment by 2_retargeting. setup.sh here only adds the
# Mimic plug-in on top.

WORK_DIR=${WORK_DIR:-$HOME/g1_baseline}

# Pinned versions, identical to the other steps.
ISAAC_SIM_IMAGE=${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.0-dev2}
ARENA_BRANCH=${ARENA_BRANCH:-release/0.2.1}
ARENA_REPO=${ARENA_REPO:-https://github.com/isaac-sim/IsaacLab-Arena.git}

ARENA_DIR=${ARENA_DIR:-$WORK_DIR/IsaacLab-Arena}
MODELS_DIR=${MODELS_DIR:-$WORK_DIR/models}
GR00T_DIR=${GR00T_DIR:-$WORK_DIR/Isaac-GR00T}

CONTAINER_WORKDIR=/workspaces/isaaclab_arena
DOCKER_IMAGE=${DOCKER_IMAGE:-isaaclab_arena:latest}

# Host-side data, eval and log directories. Nothing here is committed; see .gitignore.
DATA_DIR=${DATA_DIR:-$WORK_DIR/datasets/g1_apple_mimic}
EVAL_DIR=${EVAL_DIR:-$WORK_DIR/eval/g1_apple_mimic}
LOG_DIR=${LOG_DIR:-$WORK_DIR/logs/g1_apple_mimic}

# Container-side equivalents, mounted by arena_run.sh.
C_DATA_DIR=/datasets/g1_apple_mimic
C_EVAL_DIR=/eval/g1_apple_mimic

# The three files the pipeline moves through.
SOURCE_HDF5=${SOURCE_HDF5:-$C_DATA_DIR/source_demos.hdf5}
ANNOTATED_HDF5=${ANNOTATED_HDF5:-$C_DATA_DIR/source_annotated.hdf5}
MERGED_HDF5=${MERGED_HDF5:-$C_DATA_DIR/g1_apple_mimic_generated.hdf5}

# The rollouts run_harvest.sh records, and the input to prepare_source.sh. Both paths point at the
# same file; arena_run.sh mounts the parent of DATA_DIR as /datasets, so they stay in step.
ROLLOUTS_NAME=${ROLLOUTS_NAME:-revo2_demos.hdf5}
ROLLOUTS_HDF5=${ROLLOUTS_HDF5:-$WORK_DIR/datasets/rollouts/$ROLLOUTS_NAME}
C_ROLLOUTS_HDF5=/datasets/rollouts/$ROLLOUTS_NAME

# The inference server started by 1_baseline/run_policy_server.sh, used by run_harvest.sh.
POLICY_HOST=${POLICY_HOST:-localhost}
POLICY_PORT=${POLICY_PORT:-5556}
POLICY_TYPE=isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy
POLICY_CONFIG=isaaclab_arena_gr00t/policy/config/g1_static_apple_gr00t_closedloop_config.yaml
# The deployment embodiment from step 3. Rollouts are recorded on it, not on the Pink variant.
HARVEST_EMBODIMENT=${HARVEST_EMBODIMENT:-g1_wbc_agile_joint_brainco}

# The Mimic-capable environment registered by the plug-in, and the task it wraps.
ENV_NAME=galileo_g1_static_apple_mimic
EXT_ENV="g1_apple_mimic.environment:GalileoG1StaticAppleMimicEnvironment"
OBJECT=apple_01_objaverse_robolab
DESTINATION=clay_plates_hot3d_robolab

# Which arm performs the pick. prepare_source.sh reports which one the rollouts used.
MIMIC_ARM=${MIMIC_ARM:-left}
# Apple spawn jitter during generation, in metres (half-range, uniform in x and y). The stock env has
# none, so without this every generated demo would repeat the same scene and add nothing over the
# source. The first 200-demo round used 0.02, which is a ~15 px shift in the head camera and looks
# identical episode to episode; 0.05 gives a 10 x 10 cm box. Orientation is deliberately not
# randomised: Mimic transforms the grasp with the full object pose, so a yawed apple would rotate the
# approach direction around it.
APPLE_XY_RANGE_M=${APPLE_XY_RANGE_M:-0.05}

# prepare_source_from_generated.sh: how many generated demos become sources for the next round, and
# the time-stretch applied to them. Mimic executes one source step per env step, so generated demos
# move exactly as fast as their sources; the first round's 5 s episodes came from a GR00T rollout plus
# a scripted place squeezed under the task's 6 s cap (wrist ~0.6 m/s). 2.0 halves every velocity.
NUM_SOURCES=${NUM_SOURCES:-24}
TIME_SCALE=${TIME_SCALE:-2.0}
# g1_wbc_agile_pink_brainco = Revo2 hands on the Pink IK interface; g1_wbc_agile_pink = stock Dex3.
EMBODIMENT=${EMBODIMENT:-g1_wbc_agile_pink_brainco}

# Common tail of every Arena command for this environment.
ENV_ARGS="--external_environment_class_path ${EXT_ENV} --mimic ${ENV_NAME} \
  --object ${OBJECT} --destination ${DESTINATION} --embodiment ${EMBODIMENT} \
  --mimic_arm ${MIMIC_ARM} --apple_xy_range_m ${APPLE_XY_RANGE_M}"
