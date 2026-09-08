"""Build the Mimic env once, step it, and print the Mimic API surface. No dataset needed.

    /isaac-sim/python.sh g1_apple_mimic/smoke_test.py --headless --device cuda \
      --external_environment_class_path g1_apple_mimic.environment:GalileoG1StaticAppleMimicEnvironment \
      --mimic galileo_g1_static_apple_mimic --mimic_arm left
"""

from __future__ import annotations

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

parser = get_isaaclab_arena_cli_parser()
parser.add_argument("--steps", type=int, default=5)
add_example_environments_cli_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_mimic.envs  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLMimicEnv


def main():
    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()
    print("env_name:", env_name)
    print("mimic env class:", arena_builder.get_entry_point())
    print("datagen_config:", env_cfg.datagen_config)
    for eef, subtasks in env_cfg.subtask_configs.items():
        print(f"eef {eef!r}: {len(subtasks)} subtask(s):", [(s.object_ref, s.subtask_term_signal) for s in subtasks])
    print("subtask_terms obs group:", env_cfg.observations.subtask_terms)

    success_term = env_cfg.terminations.success
    env_cfg.terminations = None
    env_cfg.observations.policy.concatenate_terms = False
    env = gym.make(env_name, cfg=env_cfg).unwrapped
    assert isinstance(env, ManagerBasedRLMimicEnv), type(env)

    obs, _ = env.reset()
    print("obs groups:", list(obs.keys()))
    print("subtask_terms:", {k: v.tolist() for k, v in obs["subtask_terms"].items()})
    with torch.inference_mode():
        for _ in range(args_cli.steps):
            # hold pose: reconstruct the current eef targets from the Mimic API itself
            left = env.get_robot_eef_pose("left")
            right = env.get_robot_eef_pose("right")
            body = torch.zeros(7, device=env.device)
            action = env.target_eef_pose_to_action(
                {"left": left[0], "right": right[0]},
                {"left": torch.zeros(1, device=env.device)[0], "right": torch.zeros(1, device=env.device)[0], "body": body},
            )
            obs, *_ = env.step(action.unsqueeze(0))
    print("action dim:", action.shape)

    # Hand check: close the left hand for 25 steps, report commanded vs measured finger joints and the
    # recorder-facing processed_actions (must be 43-D Dex3 space for the LeRobot converter).
    robot = env.scene["robot"]
    names = list(robot.joint_names)
    finger_idx = [i for i, n in enumerate(names) if n.startswith("left_") and ("hand_" in n or "thumb" in n or "proximal" in n)]
    term = env.action_manager.get_term("g1_action")
    with torch.inference_mode():
        for _ in range(25):
            left = env.get_robot_eef_pose("left")
            right = env.get_robot_eef_pose("right")
            action = env.target_eef_pose_to_action(
                {"left": left[0], "right": right[0]},
                {"left": torch.ones(1, device=env.device)[0], "right": torch.zeros(1, device=env.device)[0], "body": torch.zeros(7, device=env.device)},
            )
            obs, *_ = env.step(action.unsqueeze(0))
    q = robot.data.joint_pos
    q = q if isinstance(q, torch.Tensor) else __import__("warp").to_torch(q)
    print("articulation joints:", len(names))
    print("left finger joints after 25 closed steps:", {names[i]: round(float(q[0, i]), 3) for i in finger_idx})
    print("processed_actions (recorded):", tuple(term.processed_actions.shape), "obs robot_joint_pos:", tuple(obs["policy"]["robot_joint_pos"].shape))
    print("get_subtask_term_signals:", {k: v.tolist() for k, v in env.get_subtask_term_signals().items()})
    print("get_object_poses keys:", list(env.get_object_poses().keys()))
    print("left eef pose:\n", env.get_robot_eef_pose("left")[0])
    print("success term now:", success_term.func(env, **success_term.params).tolist())
    round_trip = env.action_to_target_eef_pose(action.unsqueeze(0))
    print("action_to_target_eef_pose keys:", list(round_trip.keys()))
    env.close()
    print("SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
    simulation_app.close()
