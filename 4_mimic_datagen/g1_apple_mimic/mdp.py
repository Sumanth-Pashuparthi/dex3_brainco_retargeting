"""Observation terms used as Isaac Lab Mimic subtask-termination signals."""

from __future__ import annotations

import torch

import warp as wp
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def object_grasped_by_hand(
    env: ManagerBasedEnv,
    object_cfg: SceneEntityCfg,
    wrist_link_name: str,
    spawn_z_env: float,
    lift_height_m: float = 0.03,
    max_hand_dist_m: float = 0.18,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``True`` once the object has been lifted off the shelf while close to the given wrist link.

    This is the ``grasp_<arm>`` subtask-termination signal for Mimic: false during the reach phase,
    true from the moment the apple leaves the shelf in the hand. It stays true while carrying; Mimic
    only uses the 0->1 edge.

    Args:
        env: The environment.
        object_cfg: Scene entity of the pick-up object.
        wrist_link_name: Robot body name of the active hand's wrist (e.g. ``left_wrist_yaw_link``).
        spawn_z_env: Resting z of the object in the env-local frame (shelf spawn height).
        lift_height_m: Object must be at least this far above ``spawn_z_env``.
        max_hand_dist_m: Object origin must be within this distance of the wrist link.
        robot_cfg: Scene entity of the robot.

    Returns:
        Bool tensor of shape ``(num_envs,)``.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    obj_pos_w = wp.to_torch(obj.data.root_pos_w)
    obj_z_env = obj_pos_w[:, 2] - env.scene.env_origins[:, 2]
    lifted = obj_z_env > (spawn_z_env + lift_height_m)

    link_idx = robot.data.body_names.index(wrist_link_name)
    wrist_pos_w = wp.to_torch(robot.data.body_link_state_w)[:, link_idx, :3]
    near_hand = torch.linalg.vector_norm(obj_pos_w - wrist_pos_w, dim=1) < max_hand_dist_m

    return torch.logical_and(lifted, near_hand)
