# Record demonstrations from a POLICY ROLLOUT instead of human teleoperation.
#
# This is `isaaclab_arena/scripts/imitation_learning/record_demos.py` with the teleop device
# replaced by the same policy-loading path `isaaclab_arena/evaluation/policy_runner.py` uses.
# Everything downstream (HDF5 schema, LeRobot conversion, GR00T post-training) is untouched.
#
# Why this works without any change to Arena:
#
#   * `RecorderManager.record_pre_reset` (submodules/IsaacLab/.../managers/recorder_manager.py)
#     already reads the `success` termination term and stamps `success` on the episode, and
#     `DatasetExportMode.EXPORT_SUCCEEDED_ONLY` then drops every failed episode. So a rollout
#     harness that simply leaves terminations enabled gets success-filtered demos for free.
#   * `ManagerBasedRLEnv.step` calls `record_pre_step` / `record_post_step` / `record_pre_reset`
#     itself, so no per-step recorder bookkeeping is needed here.
#
# BrainCo Revo2 additions, both optional and both off by default:
#
#   --dex3_actions        Also record `processed_actions_dex3`: the executed joint targets
#                         re-expressed in the 43-DoF Dex3 space of
#                         `isaaclab_arena_gr00t/embodiments/g1/43dof_joint_space.yaml`.
#                         The Revo2 articulation has 41 joints, so raw `processed_actions` is
#                         (T, 41) and would fail the converter's
#                         `joints.shape[1] == len(action_joints_config)` assertion. This term
#                         emits (T, 43) so the stock g1_static_apple converter config, modality
#                         config and 43-DoF policy action space all stay unchanged.
#   --grasp_depth_range   Resample `hand_retarget._DEPTH` once per episode from [min, max].
#                         `aperture_match` reads that module global at call time, so this is a
#                         pure-runtime knob. Raises the yield of successful Revo2 episodes and
#                         diversifies the grasp, and because the recorded action targets reflect
#                         whichever depth succeeded, the finetuned policy learns to command that
#                         grasp itself.
#
# Usage (inside the Arena container, GR00T server already listening):
#
#   /isaac-sim/python.sh /path/to/record_policy_demos.py \
#     --headless --enable_cameras \
#     --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy \
#     --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/g1_static_apple_gr00t_closedloop_config.yaml \
#     --remote_host localhost --remote_port 5556 \
#     --dataset_file /datasets/rollouts/revo2_demos.hdf5 \
#     --num_demos 200 --max_episodes 4000 \
#     --dex3_actions --grasp_depth_range 1.0 1.5 \
#     galileo_g1_static_pick_and_place \
#     --object apple_01_objaverse_robolab \
#     --destination clay_plates_hot3d_robolab \
#     --embodiment g1_wbc_agile_joint_brainco

from __future__ import annotations

import argparse
import json
import os

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser


def build_parser() -> argparse.ArgumentParser:
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument(
        "--dataset_file", type=str, required=True, help="File path to export recorded demos (.hdf5)."
    )
    parser.add_argument(
        "--num_demos",
        type=int,
        default=200,
        help="Stop once this many SUCCESSFUL demos have been exported.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=0,
        help=(
            "Hard cap on attempted episodes, so a low success rate cannot run forever. "
            "0 disables the cap."
        ),
    )
    parser.add_argument(
        "--num_success_steps",
        type=int,
        default=0,
        help=(
            "If > 0, remove the `success` termination and instead conclude a demo after this many "
            "consecutive successful steps, mirroring record_demos.py so the episode keeps a few "
            "stable frames past the success edge (the static_apple doc recommends 10). "
            "Requires --num_envs 1. If 0, the `success` termination ends the episode immediately."
        ),
    )
    parser.add_argument(
        "--dex3_actions",
        action="store_true",
        default=False,
        help=(
            "Record `processed_actions_dex3` (43-DoF pseudo-Dex3 view of the executed joint "
            "targets). Required for the BrainCo/Revo2 embodiment; harmless for the Dex3 baseline."
        ),
    )
    parser.add_argument(
        "--no_lowerbody_cmds",
        action="store_true",
        default=False,
        help=(
            "Skip adding the `action` observation group. Only use this if the embodiment already "
            "has one (the Pink embodiments do); without it the LeRobot conversion asserts on a "
            "missing 'action' group."
        ),
    )
    parser.add_argument(
        "--grasp_depth_range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help=(
            "Resample the BrainCo grasp-depth multiplier uniformly from [MIN, MAX] at every "
            "episode reset. Only meaningful for the BrainCo embodiment."
        ),
    )
    return parser


# ---------------------------------------------------------------------------------------------
# Simulator startup must happen before any isaaclab.envs / gym import.
# ---------------------------------------------------------------------------------------------

_args_parser = build_parser()
args_cli, _ = _args_parser.parse_known_args()

from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext  # noqa: E402
from isaaclab_arena_gr00t.utils.groot_path import ensure_groot_deps_in_path  # noqa: E402

# ---------------------------------------------------------------------------------------------
# Lower-body command observations, sliced for the 50-D JOINT action layout.
#
# The LeRobot converter reads these from the HDF5 `action` group:
#     convert_hdf5_to_lerobot.py:212  assert "action" in trajectory.keys()
#     convert_hdf5_to_lerobot.py:214  trajectory["action"][hdf5_keys[teleop_key]]
# and the GR00T action modality
# (isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_gr00t_n_1_7_config.py) lists
# `base_height_command` and `navigate_command` among its 7 action keys, so training NEEDS them.
#
# That `action` observation group only exists on the *Pink* embodiments
# (`G1WBCPinkObservationsCfg.action = ActionLowerBodyCfg`, g1.py:761). `G1WBCJointObservationsCfg`
# has only `policy` and `wbc`, which is exactly why
# `ArenaEnvRecorderManagerCfg.record_post_step_flat_policy_action_observations` no-ops on the
# joint embodiments -- and why converting a joint-embodiment recording would trip the assert above.
#
# Arena's own terms cannot be reused here: `g1_observations.extract_action_components`
# (isaaclab_arena_g1/g1_env/mdp/g1_observations.py:53) slices with the PINK constants
# (NAVIGATE_CMD_START_IDX = 16, BASE_HEIGHT_CMD_START_IDX = 19), which on the 50-D joint vector
# land inside the body joint targets and would silently record nonsense. The terms below instead
# call the action term's own accessors
# (`G1DecoupledWBCJointAction.get_navigation_cmd_from_actions`, .../g1_decoupled_wbc_joint_action.py:163),
# which are defined relative to the END of the action vector and so are correct for both layouts.


def _joint_action_term(env):
    for term_name in env.action_manager.active_terms:
        term = env.action_manager.get_term(term_name)
        if hasattr(term, "get_navigation_cmd_from_actions"):
            return term
    raise RuntimeError(
        "no action term exposing get_navigation_cmd_from_actions; "
        "--record_lowerbody_cmds only applies to the G1 WBC embodiments"
    )


def navigate_cmd_obs(env):
    term = _joint_action_term(env)
    return term.get_navigation_cmd_from_actions(env.action_manager.action.clone())


def base_height_cmd_obs(env):
    term = _joint_action_term(env)
    return term.get_base_height_cmd_from_actions(env.action_manager.action.clone())


def torso_orientation_rpy_cmd_obs(env):
    term = _joint_action_term(env)
    return term.get_torso_orientation_rpy_cmd_from_actions(env.action_manager.action.clone())


def _make_dex3_recorder_cfgs():
    """Build the `processed_actions_dex3` recorder term.

    Imported lazily (inside the app context) because it pulls in isaaclab.managers.
    """
    import numpy as np
    import torch
    from isaaclab.managers import RecorderTerm, RecorderTermCfg
    from isaaclab.utils import configclass

    from isaaclab_arena.embodiments.g1_brainco.brainco_action import _load_43dof_order
    from isaaclab_arena.embodiments.g1_brainco.hand_retarget import (
        BRAINCO_ACTIVE_ORDER,
        DEX3_GROOT_ORDER,
        inverse_batch,
    )

    order43 = _load_43dof_order()
    body_names = [n for n in order43 if "_hand_" not in n]
    sides = ("left", "right")

    class PostStepDex3ProcessedActionsRecorder(RecorderTerm):
        """Executed joint targets, re-expressed in the 43-DoF Dex3 action space.

        Mirrors `isaaclab_arena/embodiments/g1_brainco/g1_brainco.py::_joint_state_43dof`, but
        applied to the action term's `processed_actions` (the commanded targets) rather than the
        measured joint positions. Body joints pass through by name; each hand's 6 Revo2 targets
        are inverted to the 7 pseudo-Dex3 targets that request the same finger closure.

        On a Dex3 articulation the hand joints are already present by name, so the mapping is a
        pure by-name reindex into `43dof_joint_space.yaml` order and the term is a no-op in
        substance.
        """

        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self._resolved = False

        def _resolve(self):
            asset = self._env.scene["robot"]
            names = list(asset.joint_names)
            self._names = names

            missing = [n for n in body_names if n not in names]
            if missing:
                raise RuntimeError(f"articulation is missing expected body joints: {missing}")
            self._body_idx = [names.index(n) for n in body_names]

            # Prefer real Dex3 joints when present; fall back to Revo2 + inverse retargeting.
            self._is_dex3 = all(
                f"{side}_{j}" in names for side in sides for j in DEX3_GROOT_ORDER
            )
            if self._is_dex3:
                self._hand_idx = {
                    side: [names.index(f"{side}_{j}") for j in DEX3_GROOT_ORDER] for side in sides
                }
            else:
                self._hand_idx = {
                    side: [names.index(f"{side}_{j}") for j in BRAINCO_ACTIVE_ORDER]
                    for side in sides
                }
            self._resolved = True

        def record_post_step(self):
            if not self._resolved:
                self._resolve()

            processed = None
            for term_name in self._env.action_manager.active_terms:
                term = self._env.action_manager.get_term(term_name)
                if getattr(term, "processed_actions", None) is None:
                    continue
                processed = term.processed_actions
                break
            if processed is None:
                return None, None

            out = torch.zeros(
                (processed.shape[0], len(order43)), dtype=processed.dtype, device=processed.device
            )
            for name, idx in zip(body_names, self._body_idx):
                out[:, order43[name]] = processed[:, idx]

            if self._is_dex3:
                for side in sides:
                    for j, jname in enumerate(DEX3_GROOT_ORDER):
                        out[:, order43[f"{side}_{jname}"]] = processed[:, self._hand_idx[side][j]]
            else:
                cmd = processed.detach().cpu().numpy().astype(np.float64)
                for side in sides:
                    pseudo = inverse_batch(cmd[:, self._hand_idx[side]], side)
                    for j, jname in enumerate(DEX3_GROOT_ORDER):
                        out[:, order43[f"{side}_{jname}"]] = torch.as_tensor(
                            pseudo[:, j], dtype=processed.dtype, device=processed.device
                        )

            return "processed_actions_dex3", out

    @configclass
    class PostStepDex3ProcessedActionsRecorderCfg(RecorderTermCfg):
        class_type: type[RecorderTerm] = PostStepDex3ProcessedActionsRecorder

    return PostStepDex3ProcessedActionsRecorderCfg


def main() -> None:  # noqa: C901
    import random

    import gymnasium as gym
    import torch
    import tqdm
    from isaaclab.managers import DatasetExportMode
    from isaaclab.utils import configclass

    from isaaclab_arena.evaluation.policy_runner import get_policy_cls
    from isaaclab_arena.evaluation.policy_runner_cli import add_policy_runner_arguments
    from isaaclab_arena.metrics.metrics_logger import metrics_to_plain_python_types
    from isaaclab_arena.utils.isaaclab_utils.recorders import ArenaEnvRecorderManagerCfg
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import reapply_viewer_cfg
    from isaaclab_arena_environments.cli import (
        get_arena_builder_from_cli,
        get_isaaclab_arena_environments_cli_parser,
    )

    global args_cli
    args_parser = build_parser()
    add_policy_runner_arguments(args_parser)
    args_cli, _ = args_parser.parse_known_args()

    policy_cls = get_policy_cls(args_cli.policy_type)
    args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
    args_parser = policy_cls.add_args_to_parser(args_parser)
    args_cli = args_parser.parse_args()

    if not args_cli.enable_cameras:
        raise SystemExit(
            "--enable_cameras is required: the LeRobot converter reads ego RGB from "
            "camera_obs/robot_head_cam_rgb, and ArenaEnvRecorderManagerCfg only records it when "
            "cameras are on."
        )
    if args_cli.num_success_steps > 0 and args_cli.num_envs != 1:
        raise SystemExit("--num_success_steps > 0 requires --num_envs 1.")

    output_dir = os.path.dirname(os.path.abspath(args_cli.dataset_file))
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    os.makedirs(output_dir, exist_ok=True)

    # -- environment config, with recorders wired in before the env is instantiated -----------
    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()

    # The recorder stores observations term-by-term; the converter looks up
    # obs/robot_joint_pos by name, so the policy group must not be concatenated.
    env_cfg.observations.policy.concatenate_terms = False

    # -- lower-body command observations -> HDF5 `action` group ---------------------------------
    # ObservationManager enumerates groups from `self.cfg.__dict__`
    # (submodules/IsaacLab/.../managers/observation_manager.py:492), so assigning a new attribute
    # on the instance is enough to register the group.
    if not args_cli.no_lowerbody_cmds:
        if getattr(env_cfg.observations, "action", None) is not None:
            print("[record] embodiment already has an `action` obs group; leaving it alone")
        else:
            from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg

            @configclass
            class LowerBodyCmdObsCfg(ObservationGroupCfg):
                """Names must match the `teleop_*_name_sim` keys in the converter config."""

                navigate_cmd = ObservationTermCfg(func=navigate_cmd_obs)
                base_height_cmd = ObservationTermCfg(func=base_height_cmd_obs)
                torso_orientation_rpy_cmd = ObservationTermCfg(func=torso_orientation_rpy_cmd_obs)

                def __post_init__(self):
                    self.enable_corruption = False
                    self.concatenate_terms = False

            env_cfg.observations.action = LowerBodyCmdObsCfg()
            print("[record] added `action` obs group (navigate/base_height/torso_rpy cmds)")

    success_term = None
    if args_cli.num_success_steps > 0:
        if not hasattr(env_cfg.terminations, "success"):
            raise SystemExit("--num_success_steps > 0 needs a `success` termination term.")
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None

    if args_cli.dex3_actions:
        dex3_cfg_cls = _make_dex3_recorder_cfgs()

        @configclass
        class PolicyDemoRecorderManagerCfg(ArenaEnvRecorderManagerCfg):
            record_post_step_dex3_processed_actions = dex3_cfg_cls()

        env_cfg.recorders = PolicyDemoRecorderManagerCfg()
    else:
        env_cfg.recorders = ArenaEnvRecorderManagerCfg()

    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    env = gym.make(env_name, cfg=env_cfg)
    reapply_viewer_cfg(env)
    base_env = env.unwrapped

    # -- optional per-episode grasp-depth randomization ----------------------------------------
    hand_retarget = None
    if args_cli.grasp_depth_range is not None:
        from isaaclab_arena.embodiments.g1_brainco import hand_retarget as _hr

        hand_retarget = _hr
        lo, hi = args_cli.grasp_depth_range
        print(f"[record] grasp depth will be resampled from [{lo}, {hi}] each episode", flush=True)

    def resample_grasp_depth() -> float | None:
        if hand_retarget is None:
            return None
        lo, hi = args_cli.grasp_depth_range
        depth = random.uniform(lo, hi)
        hand_retarget._DEPTH = depth
        return depth

    # -- policy ---------------------------------------------------------------------------------
    policy = policy_cls.from_args(args_cli)

    obs, _ = env.reset()
    policy.reset()
    task_description = (
        args_cli.language_instruction
        or base_env.cfg.isaaclab_arena_env.task.get_task_description()
    )
    policy.set_task_description(task_description)
    print(f"[record] task description: {task_description!r}", flush=True)

    current_depth = resample_grasp_depth()

    # Sidecar metadata: the articulation joint order is what `processed_actions` columns mean,
    # and it is otherwise unrecoverable from the HDF5.
    meta = {
        "env_name": env_name,
        "embodiment": getattr(args_cli, "embodiment", None),
        "task_description": task_description,
        "dataset_file": args_cli.dataset_file,
        "joint_names": list(base_env.scene["robot"].joint_names),
        "num_joints": len(base_env.scene["robot"].joint_names),
        "action_dim": int(base_env.action_manager.total_action_dim),
        "dex3_actions_recorded": bool(args_cli.dex3_actions),
        "grasp_depth_range": args_cli.grasp_depth_range,
        "num_success_steps": args_cli.num_success_steps,
    }
    meta_path = os.path.join(output_dir, output_file_name + ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[record] wrote {meta_path}", flush=True)

    # -- rollout loop ---------------------------------------------------------------------------
    exported = 0
    attempted = 0
    success_step_count = 0
    per_episode_depth: list[tuple[float | None, bool]] = []

    pbar = tqdm.tqdm(total=args_cli.num_demos, desc="Demos", unit="demo")
    try:
        while exported < args_cli.num_demos:
            if args_cli.max_episodes and attempted >= args_cli.max_episodes:
                print(
                    f"[record] hit --max_episodes {args_cli.max_episodes} with {exported} demos",
                    flush=True,
                )
                break

            with torch.inference_mode():
                actions = policy.get_action(env, obs)
                obs, _, terminated, truncated, _ = env.step(actions)

            manual_export = False
            if success_term is not None:
                if bool(success_term.func(base_env, **success_term.params)[0]):
                    success_step_count += 1
                    if success_step_count >= args_cli.num_success_steps:
                        base_env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                        base_env.recorder_manager.set_success_to_episodes(
                            [0],
                            torch.tensor([[True]], dtype=torch.bool, device=base_env.device),
                        )
                        base_env.recorder_manager.export_episodes([0])
                        manual_export = True
                else:
                    success_step_count = 0

            episode_over = manual_export or bool(terminated.any()) or bool(truncated.any())
            if episode_over:
                attempted += 1
                new_exported = base_env.recorder_manager.exported_successful_episode_count
                succeeded = new_exported > exported
                per_episode_depth.append((current_depth, succeeded))
                if new_exported > exported:
                    pbar.update(new_exported - exported)
                    exported = new_exported
                pbar.set_postfix_str(
                    f"{exported}/{attempted} attempts"
                    + (f" depth={current_depth:.3f}" if current_depth is not None else "")
                )

                success_step_count = 0
                if manual_export:
                    # The `success` termination was removed, so the env will not auto-reset here.
                    base_env.recorder_manager.reset()
                    obs, _ = env.reset()
                    policy.reset()
                else:
                    env_ids = (terminated | truncated).nonzero().flatten()
                    policy.reset(env_ids=env_ids)
                current_depth = resample_grasp_depth()
    finally:
        pbar.close()

    print(f"[record] exported {exported} successful demos out of {attempted} attempts", flush=True)
    if attempted:
        print(f"[record] success rate: {exported / attempted:.1%}", flush=True)
    if hand_retarget is not None:
        succ = [d for d, ok in per_episode_depth if ok and d is not None]
        if succ:
            print(
                f"[record] successful grasp depths: n={len(succ)} "
                f"min={min(succ):.3f} mean={sum(succ) / len(succ):.3f} max={max(succ):.3f}",
                flush=True,
            )
        print(f"[record] per-episode (depth, success): {per_episode_depth}", flush=True)

    if hasattr(base_env.cfg, "metrics") and base_env.cfg.metrics is not None:
        from isaaclab_arena.metrics.metrics import compute_metrics

        print(f"[record] metrics: {metrics_to_plain_python_types(compute_metrics(base_env))}")

    if policy.is_remote:
        policy.shutdown_remote(kill_server=False)
    env.close()
    print(f"[record] demos saved to: {args_cli.dataset_file}", flush=True)


if __name__ == "__main__":
    ensure_groot_deps_in_path()
    with SimulationAppContext(args_cli):
        main()
