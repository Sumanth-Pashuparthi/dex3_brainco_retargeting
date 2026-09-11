"""G1 Mimic env with automatic subtask-termination signals for the static apple task."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab_arena.embodiments.g1.g1 import G1MimicEnv


class G1StaticAppleMimicEnv(G1MimicEnv):
    """``G1MimicEnv`` + ``get_subtask_term_signals`` so ``annotate_demos.py --auto`` works headless.

    Every term of the ``subtask_terms`` observation group (added by
    :class:`g1_apple_mimic.task.G1StaticApplePickPlaceTask`) is exposed as a signal.

    Also fixes the stale first camera frame of every generated episode (see :meth:`_refresh_camera_obs`).
    """

    # -- stale-frame fix -------------------------------------------------------------------------------

    def reset(self, *args, **kwargs):
        obs, extras = super().reset(*args, **kwargs)
        self._refresh_camera_obs()
        return self.obs_buf, extras

    def reset_to(self, *args, **kwargs):
        out = super().reset_to(*args, **kwargs)
        self._refresh_camera_obs()
        return (self.obs_buf, out[1]) if isinstance(out, tuple) else out

    def _refresh_camera_obs(self) -> None:
        """Re-render after a reset so ``obs_buf["camera_obs"]`` shows the reset scene, not the previous episode.

        In this Isaac Lab, ``sim.render()`` no longer drives RTX; the camera pumps the renderer itself in
        ``ensure_isaac_rtx_render_update()``, which is de-duplicated per ``(sim, physics_step_count)``. A
        reset does not advance the physics step count, so the camera read that ``reset()`` performs is a
        no-op pump and returns the annotator frame from the *last step of the previous episode* (or the
        pre-reset scene for the first one). ``num_rerenders_on_reset`` cannot help: it loops ``sim.render()``.
        The recorder copies that ``obs_buf["camera_obs"]`` as step 0 of the episode, so without this every
        generated demo starts with an image of the wrong scene.

        Here: defeat the dedup key, pump the renderer a few times so RTX has converged on the new state, mark
        the sensors outdated and recompute the ``camera_obs`` group in place.
        """
        if not getattr(self, "has_rtx_sensors", False) or "camera_obs" not in self.observation_manager.active_terms:
            return
        import isaaclab_physx.renderers.isaac_rtx_renderer_utils as rtx

        for _ in range(max(2, int(getattr(self.cfg, "num_rerenders_on_reset", 0) or 0))):
            rtx._last_render_update_key = (0, -1)
            rtx.ensure_isaac_rtx_render_update()
        for sensor in self.scene.sensors.values():
            sensor.reset()
        self.obs_buf["camera_obs"] = self.observation_manager.compute_group("camera_obs")

    # -- Mimic API --------------------------------------------------------------------------------------

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
