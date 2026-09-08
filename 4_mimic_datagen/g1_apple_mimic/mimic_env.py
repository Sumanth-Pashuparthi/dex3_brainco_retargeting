"""G1 Mimic env with automatic subtask-termination signals for the static apple task."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab_arena.embodiments.g1.g1 import G1MimicEnv


class G1StaticAppleMimicEnv(G1MimicEnv):
    """``G1MimicEnv`` + ``get_subtask_term_signals`` so ``annotate_demos.py --auto`` works headless.

    Every term of the ``subtask_terms`` observation group (added by
    :class:`g1_apple_mimic.task.G1StaticApplePickPlaceTask`) is exposed as a signal.
    """

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Eef pose in the frame the action term expects.

        With the Revo2 Pink action term (``anchor_targets=True``) that is the fixed source-pelvis frame
        ``ANCHOR_T_W``; otherwise fall back to Arena's live-pelvis-frame observation.
        """
        term = self.action_manager.get_term("g1_action")
        anchor = getattr(term, "anchor_T_w", None)
        if anchor is None:
            return super().get_robot_eef_pose(eef_name, env_ids)
        import numpy as np
        import warp as wp
        from scipy.spatial.transform import Rotation as R

        if env_ids is None:
            env_ids = slice(None)
        robot = self.scene["robot"]
        link = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link", "body": "right_wrist_yaw_link"}[eef_name]
        b = list(robot.data.body_names).index(link)
        pos = wp.to_torch(robot.data.body_link_pos_w)[env_ids, b].cpu().numpy()
        q_xyzw = wp.to_torch(robot.data.body_link_quat_w)[env_ids, b].cpu().numpy()  # this build: (x, y, z, w)
        anchor_inv = np.linalg.inv(anchor)
        out = np.tile(np.eye(4), (pos.shape[0], 1, 1))
        for i in range(pos.shape[0]):
            T = np.eye(4)
            T[:3, :3] = R.from_quat(q_xyzw[i]).as_matrix()
            T[:3, 3] = pos[i]
            out[i] = anchor_inv @ T
        return torch.as_tensor(out, dtype=torch.float32, device=self.device)

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        """Object poses in the same fixed frame as :meth:`get_robot_eef_pose` when the action term is anchored.

        Arena's default expresses objects in the *live* pelvis frame; with anchored wrist targets the eef poses are
        in the fixed source-pelvis frame, so the object poses must be too (Mimic composes eef-relative-to-object
        transforms from the two). Falls back to Arena's behaviour for non-anchored embodiments.
        """
        term = self.action_manager.get_term("g1_action")
        anchor = getattr(term, "anchor_T_w", None)
        if anchor is None:
            return super().get_object_poses(env_ids)
        import numpy as np
        from scipy.spatial.transform import Rotation as R

        if env_ids is None:
            env_ids = slice(None)
        anchor_inv = np.linalg.inv(anchor)
        out = {}
        for obj_name, obj_state in self.scene.get_state(is_relative=True)["rigid_object"].items():
            rp = obj_state["root_pose"][env_ids].cpu().numpy()  # env-local (x, y, z, qx, qy, qz, qw)
            mats = np.tile(np.eye(4), (rp.shape[0], 1, 1))
            for i in range(rp.shape[0]):
                T = np.eye(4)
                T[:3, :3] = R.from_quat(rp[i, 3:7]).as_matrix()
                T[:3, 3] = rp[i, :3]
                mats[i] = anchor_inv @ T
            out[obj_name] = torch.as_tensor(mats, dtype=torch.float32, device=self.device)
        return out

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        subtask_terms = self.obs_buf["subtask_terms"]
        return {name: flags[env_ids] for name, flags in subtask_terms.items()}

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Shape-normalising wrapper around ``G1MimicEnv.target_eef_pose_to_action``.

        ``actions_to_gripper_actions`` yields 0-D hand actions and a (7,) body action. Mimic's
        interpolation path (``WaypointTrajectory.add_waypoint_sequence_for_target_pose`` with
        ``num_interpolation_steps > 0``) re-shapes them via ``unsqueeze(0).repeat((n, 1))`` so the hand
        actions arrive here as ``(1,)`` and the parent's extra ``unsqueeze(0)`` makes ``torch.cat`` fail
        ("got 2 and 1"). Flatten everything back to the shapes the parent expects.
        """
        gripper_action_dict = {
            "left": gripper_action_dict["left"].reshape(()),
            "right": gripper_action_dict["right"].reshape(()),
            "body": gripper_action_dict["body"].reshape(-1)[-7:],
        }
        target_eef_pose_dict = {k: v.reshape(4, 4) for k, v in target_eef_pose_dict.items()}
        return super().target_eef_pose_to_action(
            target_eef_pose_dict=target_eef_pose_dict,
            gripper_action_dict=gripper_action_dict,
            action_noise_dict=action_noise_dict,
            env_id=env_id,
        )
