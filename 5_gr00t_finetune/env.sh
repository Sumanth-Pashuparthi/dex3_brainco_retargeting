# Shared configuration for every script in this folder. Sourced, not executed.
#
# Override anything from the shell, e.g.
#   WORK_DIR=/data/g1 VARIANT=r2r3 ./run_train.sh
#
# Nothing here is an absolute path to one machine. Every location derives from WORK_DIR, which
# defaults to the same ~/g1_baseline the earlier steps install into, so whichever step you ran
# first has already put the checkout, the venv and the datasets where these scripts look.
#
# Large artefacts (datasets, checkpoints, rollout videos) can be moved off the WORK_DIR disk
# without touching the rest: set DATA_DIR, CKPT_ROOT and EVAL_DIR independently. The runs in
# results/ were produced with CKPT_ROOT and DATA_DIR pointed at a RAID array for exactly that
# reason, which is why no path in this folder assumes they live under WORK_DIR.

WORK_DIR=${WORK_DIR:-$HOME/g1_baseline}

# Installed by 1_baseline/setup.sh (Arena, the GR00T checkout and its venv) and extended by
# 2_retargeting (the Revo2 asset and embodiment). GR00T_DIR must be an N1.7-capable checkout:
# launch_finetune.py and the Gr00tPolicy server both come from it.
ARENA_DIR=${ARENA_DIR:-$WORK_DIR/IsaacLab-Arena}
GR00T_DIR=${GR00T_DIR:-$WORK_DIR/Isaac-GR00T}
GR00T_PY=${GR00T_PY:-$GR00T_DIR/.venv/bin/python}

DATA_DIR=${DATA_DIR:-$WORK_DIR/datasets/g1_apple_mimic}
CKPT_ROOT=${CKPT_ROOT:-$WORK_DIR/ckpt}
EVAL_DIR=${EVAL_DIR:-$WORK_DIR/eval/g1_apple_mimic}
LOG_DIR=${LOG_DIR:-$WORK_DIR/logs/g1_apple_mimic}

# ---------------------------------------------------------------------------- what to train
#
# VARIANT selects one of the three runs the report compares. Each names a dataset produced by
# 4_mimic_datagen and a starting checkpoint; everything else about the recipe is identical, so the
# variants are comparable to each other.
#
#   r1r2        rounds 1-2 only (1668 demos). The first fine-tune. 5k steps.
#   r2r3        rounds 1-3 merged (2647 demos). The second fine-tune. 10k steps.
#   r2r3-base   same data as r2r3 but starting from the general N1.7 base rather than NVIDIA's
#               apple-to-plate expert. Isolates how much of the result is inherited from the
#               expert checkpoint rather than learned from our demos. Staged, not yet run.
VARIANT=${VARIANT:-r2r3}

# The task-tuned checkpoint the first two variants start from. This is the same checkpoint steps 1
# and 3 evaluate frozen, so fine-tuning starts from the measured 0.06 rather than from scratch.
EXPERT_MODEL=${EXPERT_MODEL:-nvidia/GN1x-Tuned-Arena-G1-Static-PickNPlace}
BASE_N17_MODEL=${BASE_N17_MODEL:-nvidia/GR00T-N1.7-3B}

case "$VARIANT" in
  r1r2)
    DATASET_NAME=${DATASET_NAME:-g1_apple_mimic_generated}
    BASE_MODEL=${BASE_MODEL:-$EXPERT_MODEL}
    MAX_STEPS=${MAX_STEPS:-5000}
    GLOBAL_BATCH=${GLOBAL_BATCH:-192}
    RUN_NAME=${RUN_NAME:-g1_apple_ft_r1r2}
    ;;
  r2r3)
    DATASET_NAME=${DATASET_NAME:-g1_apple_mimic_r2r3}
    BASE_MODEL=${BASE_MODEL:-$EXPERT_MODEL}
    MAX_STEPS=${MAX_STEPS:-10000}
    GLOBAL_BATCH=${GLOBAL_BATCH:-240}
    RUN_NAME=${RUN_NAME:-g1_apple_ft_r2r3}
    ;;
  r2r3-base)
    DATASET_NAME=${DATASET_NAME:-g1_apple_mimic_r2r3}
    BASE_MODEL=${BASE_MODEL:-$BASE_N17_MODEL}
    MAX_STEPS=${MAX_STEPS:-10000}
    GLOBAL_BATCH=${GLOBAL_BATCH:-240}
    RUN_NAME=${RUN_NAME:-g1_apple_ft_r2r3_from_base}
    ;;
  *)
    echo "unknown VARIANT '$VARIANT' (expected r1r2, r2r3 or r2r3-base)" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac

DATASET=${DATASET:-$DATA_DIR/$DATASET_NAME/lerobot}
OUT_DIR=${OUT_DIR:-$CKPT_ROOT/$RUN_NAME}

# The modality config decides what the model sees and how long a chunk it predicts. Taken from
# Arena rather than restated here, so training and deployment cannot drift: action_horizon 40 in
# this file is the same 40 the policy server hands to the action-chunk buffer at evaluation.
MODALITY_CFG=${MODALITY_CFG:-$ARENA_DIR/isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_gr00t_n_1_7_config.py}
EMBODIMENT_TAG=${EMBODIMENT_TAG:-new_embodiment}

# Arena's documented recipe for this task: adapt the vision tower, the projector and the flow
# matching head, leave the language model frozen. The LLM is what knows "pick up the apple"; the
# hand swap changes what the arm has to do, not what the instruction means.
TUNE_ARGS=${TUNE_ARGS:-"--no-tune-llm --tune-visual --tune-projector --tune-diffusion-model"}
LR=${LR:-1e-4}
SAVE_STEPS=${SAVE_STEPS:-1000}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-4}
WORKERS=${WORKERS:-8}

GPUS=${GPUS:-0,1,2}
NGPU=$(echo "$GPUS" | tr ',' '\n' | grep -c .)

# ---------------------------------------------------------------------------- evaluation
#
# The evaluation half of this step is deliberately the *same* script and config as step 3. The
# whole point of the comparison is that only the weights change, so the task, embodiment, success
# term and episode length are all inherited rather than redefined.
POLICY_HOST=${POLICY_HOST:-localhost}
POLICY_PORT=${POLICY_PORT:-5556}
POLICY_TYPE=isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy
POLICY_CONFIG=${POLICY_CONFIG:-isaaclab_arena_gr00t/policy/config/g1_static_apple_gr00t_closedloop_config.yaml}
EMBODIMENT=${EMBODIMENT:-g1_wbc_agile_joint_brainco}
TASK=${TASK:-"galileo_g1_static_pick_and_place \
  --object apple_01_objaverse_robolab \
  --destination clay_plates_hot3d_robolab"}

# Apple spawn jitter at evaluation, metres (half-range). NOT comparable across values: 0.0 is the
# stock deterministic spawn that steps 1 and 3 measured, and is the only setting that can be
# compared to their numbers. 0.05 matches the generation jitter and is what the checkpoint sweep
# uses. See the README table before quoting any of these against each other.
EVAL_XY_RANGE_M=${EVAL_XY_RANGE_M:-0.0}
EPISODE_LENGTH_S=${EPISODE_LENGTH_S:-14.0}

CONTAINER_WORKDIR=/workspaces/isaaclab_arena
DOCKER_IMAGE=${DOCKER_IMAGE:-isaaclab_arena:latest}

# ---------------------------------------------------------------------------- distributed
#
# Three PCIe GPUs with no NVLink (nvidia-smi topo reports NODE). NCCL's default peer-to-peer path
# dies with "CUDA error: an illegal memory access was encountered" in the very first all-reduce;
# these five route the collective through host memory instead. Two of them are not enough -- the
# resume of the r2r3 run crashed on exactly that, having inherited only P2P and IB from an older
# launcher. Harmless on an NVLink box, so they are unconditional.
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-1}
export NCCL_PROTO=${NCCL_PROTO:-Simple}
export NCCL_ALGO=${NCCL_ALGO:-Ring}
