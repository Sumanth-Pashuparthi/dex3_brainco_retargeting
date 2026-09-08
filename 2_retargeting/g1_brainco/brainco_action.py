"""Action term for the G1 wearing BrainCo Revo2 hands, driven by the Dex3-trained GR00T policy.

The baseline robot has 43 actuated joints (29 body + 7 per Dex3 hand); this robot has 41
(29 body + 6 per Revo2 hand). The checkpoint is frozen and still emits a 43-DoF Dex3 chunk, and
``action_dim`` does not depend on the asset, so the interface stays in source-embodiment space and
this term absorbs the difference:

    1. hand block of the incoming action -> retargeted to Revo2 joint targets, written directly to
       the 6 hand DoF per side (the whole-body controller has no opinion about fingers)
    2. measured Revo2 joints -> inverted to pseudo-Dex3 angles, used to synthesise the 43-DoF joint
       state the controller's G1 model expects
    3. body joints pass through the unmodified whole-body controller, so balance behaviour is
       identical to the baseline

``postprocess_actions`` is reused as-is: it already skips controller joints that are absent from
the articulation, so the Dex3 finger entries fall away and only the 29 body joints are written from
the controller solution.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch
import yaml

import isaaclab.utils.math as math_utils
import warp as wp
from isaaclab.utils import configclass

from isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_joint_action import G1DecoupledWBCJointAction
from isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_joint_action_cfg import G1DecoupledWBCJointActionCfg
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.run_policy import postprocess_actions

from .hand_retarget import BRAINCO_ACTIVE_ORDER, DEX3_GROOT_ORDER, inverse_batch, retarget_batch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

_SIDES = ("left", "right")
_DEBUG = os.environ.get("BRAINCO_DEBUG") == "1"


def _load_43dof_order() -> dict[str, int]:
    """Joint-name -> index map of the 43-DoF space the GR00T action chunk is written into."""
    root = os.path.abspath(__file__)
    for _ in range(4):
        root = os.path.dirname(root)
    path = os.path.join(root, "isaaclab_arena_gr00t/embodiments/g1/43dof_joint_space.yaml")
    with open(path) as f:
        return yaml.safe_load(f)["joints"]


_ORDER43 = _load_43dof_order()


class G1BraincoWBCJointAction(G1DecoupledWBCJointAction):
    """Dex3-space actions in, Revo2 joint targets out."""

    def __init__(self, cfg: G1BraincoWBCJointActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        asset_names = list(self._asset.joint_names)

        self._body_names = [n for n in _ORDER43 if "_hand_" not in n]
        missing = [n for n in self._body_names if n not in asset_names]
        if missing:
            raise RuntimeError(f"Revo2 G1 articulation is missing expected body joints: {missing}")
        self._body_idx_asset = [asset_names.index(n) for n in self._body_names]

        self._dex3_idx43 = {
            side: [_ORDER43[f"{side}_{j}"] for j in DEX3_GROOT_ORDER] for side in _SIDES
        }
        self._revo2_idx_asset = {}
        for side in _SIDES:
            names = [f"{side}_{j}" for j in BRAINCO_ACTIVE_ORDER]
            absent = [n for n in names if n not in asset_names]
            if absent:
                raise RuntimeError(f"Revo2 G1 articulation is missing hand joints: {absent}")
            self._revo2_idx_asset[side] = [asset_names.index(n) for n in names]

        self._action_to_wbc = [(_ORDER43[n], self.wbc_g1_joints_order[n]) for n in _ORDER43]
        self._num_wbc_joints = len(self.wbc_g1_joints_order)

    def _synthetic_wbc_joint_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Measured articulation state expressed in the controller's 43-DoF Dex3 joint space."""
        data = self._asset.data
        q = wp.to_torch(data.joint_pos).cpu().numpy()
        dq = wp.to_torch(data.joint_vel).cpu().numpy()
        q_def = wp.to_torch(data.default_joint_pos).cpu().numpy()
        n = q.shape[0]

        q43 = np.zeros((n, self._num_wbc_joints))
        dq43 = np.zeros((n, self._num_wbc_joints))
        qd43 = np.zeros((n, self._num_wbc_joints))

        for name, idx_asset in zip(self._body_names, self._body_idx_asset):
            w = self.wbc_g1_joints_order[name]
            q43[:, w] = q[:, idx_asset]
            dq43[:, w] = dq[:, idx_asset]
            qd43[:, w] = q_def[:, idx_asset]

        # Finger velocities stay zero: the controller uses finger entries only as part of the
        # upper-body posture target and never differentiates them for balance.
        for side in _SIDES:
            pseudo = inverse_batch(q[:, self._revo2_idx_asset[side]], side)
            for j, jname in enumerate(DEX3_GROOT_ORDER):
                q43[:, self.wbc_g1_joints_order[f"{side}_{jname}"]] = pseudo[:, j]
        return q43, dq43, qd43

    def _prepare_observations(self, q43: np.ndarray, dq43: np.ndarray, qd43: np.ndarray) -> dict:
        """Controller observation dict, mirroring run_policy.prepare_observations."""
        data = self._asset.data
        n = q43.shape[0]

        root_pos_w = wp.to_torch(data.root_link_pos_w).cpu().numpy()
        root_quat_xyzw = wp.to_torch(data.root_link_quat_w).cpu().numpy()
        root_quat_wxyz = np.concatenate((root_quat_xyzw[:, 3:4], root_quat_xyzw[:, :3]), axis=1)
        base_pose_w = np.concatenate((root_pos_w, root_quat_wxyz), axis=1)
        base_vel_b = np.concatenate(
            (
                wp.to_torch(data.root_link_lin_vel_b).cpu().numpy(),
                wp.to_torch(data.root_link_ang_vel_b).cpu().numpy(),
            ),
            axis=1,
        )

        torso = wp.to_torch(data.body_link_state_w)[:, data.body_names.index("torso_link"), :]
        torso_quat_xyzw = torso[:, 3:7]
        torso_quat_wxyz = torch.cat((torso_quat_xyzw[:, 3:4], torso_quat_xyzw[:, :3]), dim=1)
        torso_ang_vel_b = math_utils.quat_apply_inverse(torso_quat_xyzw, torso[:, -3:])

        return {
            "q": q43,
            "dq": dq43,
            "default_q": qd43,
            "ddq": np.zeros((n, self._num_wbc_joints)),
            "tau_est": np.zeros((n, self._num_wbc_joints)),
            "floating_base_pose": base_pose_w,
            "floating_base_vel": base_vel_b,
            "floating_base_acc": np.zeros((n, 6)),
            "projected_gravity_b": wp.to_torch(data.projected_gravity_b).cpu().numpy(),
            "root_ang_vel_b": wp.to_torch(data.root_ang_vel_b).cpu().numpy(),
            "torso_quat": torso_quat_wxyz.cpu().numpy(),
            "torso_ang_vel": torso_ang_vel_b.cpu().numpy(),
        }

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions[:, : self.action_dim]
        actions_clone = actions.clone()

        self.set_wbc_goal(
            self.get_navigation_cmd_from_actions(actions_clone),
            self.get_base_height_cmd_from_actions(actions_clone),
            self.get_torso_orientation_rpy_cmd_from_actions(actions_clone),
        )
        self.wbc_policy.set_goal(self._wbc_goal)

        action_np = actions_clone.detach().cpu().numpy()

        revo2_targets = {
            side: retarget_batch(action_np[:, self._dex3_idx43[side]], side) for side in _SIDES
        }

        q43, dq43, qd43 = self._synthetic_wbc_joint_state()
        self.wbc_policy.set_observation(self._prepare_observations(q43, dq43, qd43))

        target43 = np.zeros((action_np.shape[0], self._num_wbc_joints))
        for idx_action, idx_wbc in self._action_to_wbc:
            target43[:, idx_wbc] = action_np[:, idx_action]
        upper = target43[:, self.robot_model.get_joint_group_indices("upper_body")]

        wbc_action = self.wbc_policy.get_action(upper)
        processed = postprocess_actions(
            wbc_action, self._asset.data, self.wbc_g1_joints_order, self.device
        )

        # Fingers are not part of the controller solution for this articulation.
        for side in _SIDES:
            processed[:, self._revo2_idx_asset[side]] = torch.as_tensor(
                revo2_targets[side], dtype=processed.dtype, device=processed.device
            )
        self._processed_actions = processed

        if _DEBUG:
            self._log_debug(action_np, revo2_targets, processed)

    def _log_debug(self, action_np, revo2_targets, processed):
        """Trace the retargeting boundary. Enabled with BRAINCO_DEBUG=1."""
        self._dbg_step = getattr(self, "_dbg_step", 0) + 1
        if self._dbg_step == 1:
            names43 = {v: k for k, v in _ORDER43.items()}
            nz = np.nonzero(np.abs(action_np[0]) > 1e-6)[0]
            print(f"[brainco] action_dim={action_np.shape[1]} nonzero={len(nz)}", flush=True)
            for i in nz:
                print(
                    f"[brainco]   a[{i:2d}] = {action_np[0, i]:+.4f}  "
                    f"{names43.get(i, '<wbc cmd>')}",
                    flush=True,
                )
        if self._dbg_step % 20 != 1:
            return

        data = self._asset.data
        q = wp.to_torch(data.joint_pos).cpu().numpy()
        names = list(self._asset.joint_names)
        # Full root position, not just z: z alone cannot distinguish the controller tracking its
        # commanded base height from the robot wandering away from the table, which is at +x.
        root = wp.to_torch(data.root_link_pos_w).cpu().numpy()[0]
        # Left hand, because this task is left-handed: the right-hand action is identically zero.
        arm = "left_shoulder_pitch_joint"
        arm_cmd = action_np[0, _ORDER43[arm]]
        arm_now = q[0, names.index(arm)]
        arm_out = processed[0, names.index(arm)].item()
        dex3_cmd = action_np[0, self._dex3_idx43["left"]]
        revo2_cmd = revo2_targets["left"][0]
        revo2_now = q[0, self._revo2_idx_asset["left"]]
        # A large persistent command/measurement gap means the fingers are being stopped, by the
        # object or by too little torque; a zero gap in free air means any failure is downstream.
        print(
            f"[brainco] step {self._dbg_step:4d} "
            f"root {root[0]:+.3f},{root[1]:+.3f},{root[2]:+.3f} | "
            f"{arm} cmd {arm_cmd:+.3f} wbc_out {arm_out:+.3f} meas {arm_now:+.3f} | "
            f"dex3_L {np.array2string(dex3_cmd, precision=2)} -> "
            f"revo2_cmd {np.array2string(revo2_cmd, precision=2)} "
            f"meas {np.array2string(revo2_now, precision=2)} "
            f"err {np.array2string(revo2_cmd - revo2_now, precision=2)}",
            flush=True,
        )


@configclass
class G1BraincoWBCJointActionCfg(G1DecoupledWBCJointActionCfg):
    class_type: type = G1BraincoWBCJointAction
