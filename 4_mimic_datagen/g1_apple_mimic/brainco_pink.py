"""``g1_wbc_agile_pink_brainco``: Pink-IK (Mimic-capable) embodiment for the G1 wearing BrainCo Revo2 hands.

The deployment embodiment (``g1_wbc_agile_joint_brainco``) drives the Revo2 robot from 43-DoF Dex3-space joint targets.
Isaac Lab Mimic instead needs the 23-D Pink action interface (hand states + wrist poses + WBC
commands) so it can transform wrist waypoints per object. This module glues the two together:

* arms/waist: the stock Arena Pink IK on the Dex3 G1 robot model (identical arm kinematics), lower
  body: the AGILE WBC fed a *synthetic* 43-DoF Dex3 joint state of the Revo2 articulation (same trick
  as ``G1BraincoWBCJointAction``);
* hands: the binary hand state becomes the fixed Dex3 finger command the retargeted policy actually
  emitted in the source demos, retargeted to the 6 Revo2 joints per side with the very same
  ``hand_retarget.retarget_batch`` used at deployment (so the sim grasp is the deployed grasp);
* recording: ``processed_actions`` (what the HDF5 recorder stores) is the 43-DoF Dex3-space target
  in ``43dof_joint_space.yaml`` order, and ``robot_joint_pos`` is the 43-DoF pseudo-Dex3 state, so the
  Arena HDF5->LeRobot converter and the GR00T G1 modality config work unchanged and the finetuned
  policy plugs straight into the retargeting deployment path.

Use with any Arena imitation_learning script through ``g1_apple_mimic.environment`` and
``--embodiment g1_wbc_agile_pink_brainco``.
"""

from __future__ import annotations

import numpy as np
import os
import torch
from scipy.spatial.transform import Rotation as R
from typing import TYPE_CHECKING

import warp as wp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.g1.g1 import (
    _DEFAULT_G1_CAMERA_OFFSET,
    G1WBCAgilePinkEmbodiment,
    G1WBCPinkObservationsCfg,
    _remove_waist_from_pink_ik_action_config,
)
from isaaclab_arena.embodiments.g1_brainco.brainco_action import G1BraincoWBCJointAction, _load_43dof_order
from isaaclab_arena.embodiments.g1_brainco.g1_brainco import (
    BRAINCO_HEAD_LINK_PATH,
    G1BraincoSceneCfg,
    brainco_joint_pos_43dof,
    brainco_joint_vel_43dof,
)
from isaaclab_arena.embodiments.g1_brainco.hand_retarget import BRAINCO_ACTIVE_ORDER, DEX3_GROOT_ORDER, retarget_batch
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_pink_action import (
    G1DecoupledWBCPinkAction,
    _identity_if_zero_norm_xyzw,
)
from isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_pink_action_cfg import G1DecoupledWBCPinkActionCfg
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.action_constants import (
    LEFT_HAND_STATE_IDX,
    LEFT_WRIST_LINK_NAME,
    LEFT_WRIST_POS_END_IDX,
    LEFT_WRIST_POS_START_IDX,
    LEFT_WRIST_QUAT_END_IDX,
    LEFT_WRIST_QUAT_START_IDX,
    RIGHT_HAND_STATE_IDX,
    RIGHT_WRIST_LINK_NAME,
    RIGHT_WRIST_POS_END_IDX,
    RIGHT_WRIST_POS_START_IDX,
    RIGHT_WRIST_QUAT_END_IDX,
    RIGHT_WRIST_QUAT_START_IDX,
)
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

_SIDES = ("left", "right")
_ORDER43 = _load_43dof_order()
_DEBUG = os.environ.get("G1_PINK_DEBUG") == "1"

# Dex3-space finger command the retargeted GR00T policy emits for a closed left hand, measured from the
# Revo2 source demos (processed_actions_dex3, identical in all 22 episodes). Order: DEX3_GROOT_ORDER.
# Through retarget_batch this is Revo2 [thumb_meta 1.27, thumb_prox 0.95, index/middle/ring/pinky 0.88],
# i.e. exactly the grasp recorded in the demos. Right hand is the sign mirror (thumb_0 is 0 anyway).
DEX3_CLOSED_LEFT = np.array([-0.79, -0.87, -0.79, -0.87, 0.0, 0.52, 0.87])
DEX3_CLOSED = {"left": DEX3_CLOSED_LEFT, "right": -DEX3_CLOSED_LEFT}

# Settled pelvis pose (env/world frame) of the robot in the Revo2 source demos (mean over all 22 episodes after
# step 20, std < 3 mm). Wrist targets in the source file are expressed relative to THIS pelvis pose. With
# ``anchor_targets=True`` the action term re-expresses them in the current pelvis frame each step, so the hand
# reaches the same world position even though the AGILE WBC settles/drifts differently under Pink IK than it did
# under the joint-space policy (measured: 6 cm backward drift over one replay, which made the Revo2 hand close
# short of the apple). It also makes Mimic's eef-vs-object math exact: eef poses reported by
# ``G1StaticAppleMimicEnv.get_robot_eef_pose`` are in the same fixed frame as the world object poses (up to this
# constant transform), instead of a frame that moves with the pelvis.
ANCHOR_ROOT_POS_W = (0.258, 0.076, -0.045)
ANCHOR_ROOT_QUAT_XYZW = (-0.0003, -0.0251, -0.0123, 0.9996)


def _pose_to_mat(pos, quat_xyzw) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R.from_quat(np.asarray(quat_xyzw, dtype=float)).as_matrix()
    T[:3, 3] = np.asarray(pos, dtype=float)
    return T


ANCHOR_T_W = _pose_to_mat(ANCHOR_ROOT_POS_W, ANCHOR_ROOT_QUAT_XYZW)


class G1BraincoWBCPinkAction(G1DecoupledWBCPinkAction):
    """Pink IK upper body + AGILE WBC lower body on the Revo2 articulation; binary hands via the retargeting layer."""

    def __init__(self, cfg: G1BraincoWBCPinkActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        assert not self.cfg.use_p_control, "navigation P-control is not supported on the Revo2 Pink embodiment"

        asset_names = list(self._asset.joint_names)
        self._body_names = [n for n in _ORDER43 if "_hand_" not in n]
        missing = [n for n in self._body_names if n not in asset_names]
        if missing:
            raise RuntimeError(f"Revo2 G1 articulation is missing expected body joints: {missing}")
        self._body_idx_asset = [asset_names.index(n) for n in self._body_names]
        self._dex3_idx43 = {side: [_ORDER43[f"{side}_{j}"] for j in DEX3_GROOT_ORDER] for side in _SIDES}
        self._revo2_idx_asset = {}
        for side in _SIDES:
            names = [f"{side}_{j}" for j in BRAINCO_ACTIVE_ORDER]
            absent = [n for n in names if n not in asset_names]
            if absent:
                raise RuntimeError(f"Revo2 G1 articulation is missing hand joints: {absent}")
            self._revo2_idx_asset[side] = [asset_names.index(n) for n in names]
        self._num_wbc_joints = len(self.wbc_g1_joints_order)
        # WBC solution index -> 43dof yaml index, body joints only (fingers are filled from the command).
        self._wbc_to_43 = [(self.wbc_g1_joints_order[n], _ORDER43[n]) for n in self._body_names]

        # Pre-computed hand targets for the two binary states.
        self._revo2_target = {
            side: {
                0: torch.as_tensor(retarget_batch(np.zeros((1, 7)), side)[0], dtype=torch.float32, device=self.device),
                1: torch.as_tensor(retarget_batch(DEX3_CLOSED[side][None], side)[0], dtype=torch.float32, device=self.device),
            }
            for side in _SIDES
        }
        self._dex3_target = {
            side: {
                0: torch.zeros(7, dtype=torch.float32, device=self.device),
                1: torch.as_tensor(DEX3_CLOSED[side], dtype=torch.float32, device=self.device),
            }
            for side in _SIDES
        }

        self._sim_targets = torch.zeros((self.num_envs, len(asset_names)), device=self.device)
        self._processed_actions_43 = torch.zeros((self.num_envs, len(_ORDER43)), device=self.device)
        self.anchor_T_w: np.ndarray | None = ANCHOR_T_W.copy() if self.cfg.anchor_targets else None
        # WBC solution index -> sim joint index for the joints the Revo2 articulation actually has
        # (Arena's postprocess_actions prints a warning per missing Dex3 finger joint on every step).
        self._wbc_to_sim = [
            (wbc_idx, asset_names.index(n)) for n, wbc_idx in self.wbc_g1_joints_order.items() if n in asset_names
        ]

    # Recorder-facing view: 43-DoF Dex3-space joint targets (43dof_joint_space.yaml order).
    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions_43

    @staticmethod
    def _hand_state(x: torch.Tensor) -> int:
        return 1 if float(x.reshape(-1)[0]) > 0.5 else 0

    def current_root_T_w(self) -> np.ndarray:
        """Current pelvis pose (4x4, world). This Isaac Lab build exposes ``root_link_quat_w`` as (x, y, z, w)
        (see ``G1BraincoWBCJointAction._prepare_observations``)."""
        data = self._asset.data
        p = wp.to_torch(data.root_link_pos_w)[0].cpu().numpy()
        q_xyzw = wp.to_torch(data.root_link_quat_w)[0].cpu().numpy()
        return _pose_to_mat(p, q_xyzw)

    def _anchored_to_current_pelvis(self, pos: torch.Tensor, quat_xyzw: torch.Tensor):
        """Re-express a target given in the anchor (source pelvis) frame in the current pelvis frame."""
        T_target_anchor = _pose_to_mat(pos.numpy(), quat_xyzw.numpy())
        T_now_inv = np.linalg.inv(self.current_root_T_w())
        T = T_now_inv @ self.anchor_T_w @ T_target_anchor
        return (
            torch.as_tensor(T[:3, 3], dtype=pos.dtype),
            torch.as_tensor(R.from_matrix(T[:3, :3]).as_quat(), dtype=quat_xyzw.dtype),
        )

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions[:, : self.action_dim]
        actions_clone = actions.clone()

        # --- upper body: Pink IK on the Dex3 robot model (arm kinematics are identical) ---------------
        left_arm_pos = actions_clone[:, LEFT_WRIST_POS_START_IDX:LEFT_WRIST_POS_END_IDX].squeeze(0).cpu()
        left_arm_quat = _identity_if_zero_norm_xyzw(
            actions_clone[:, LEFT_WRIST_QUAT_START_IDX:LEFT_WRIST_QUAT_END_IDX].squeeze(0).cpu()
        )
        right_arm_pos = actions_clone[:, RIGHT_WRIST_POS_START_IDX:RIGHT_WRIST_POS_END_IDX].squeeze(0).cpu()
        right_arm_quat = _identity_if_zero_norm_xyzw(
            actions_clone[:, RIGHT_WRIST_QUAT_START_IDX:RIGHT_WRIST_QUAT_END_IDX].squeeze(0).cpu()
        )
        raw_left_pos = left_arm_pos.clone()
        if self.anchor_T_w is not None:
            left_arm_pos, left_arm_quat = self._anchored_to_current_pelvis(left_arm_pos, left_arm_quat)
            right_arm_pos, right_arm_quat = self._anchored_to_current_pelvis(right_arm_pos, right_arm_quat)
        if _DEBUG and getattr(self, "_dbg_step", 0) < 12:
            print(f"[pink_brainco]   ik-in L pos {np.asarray(left_arm_pos).round(3)} quat {np.asarray(left_arm_quat).round(3)} | "
                  f"R pos {np.asarray(right_arm_pos).round(3)} quat {np.asarray(right_arm_quat).round(3)} | "
                  f"root_T_w pos {self.current_root_T_w()[:3, 3].round(3)} rpy {R.from_matrix(self.current_root_T_w()[:3, :3]).as_euler('xyz', degrees=True).round(1)}", flush=True)
        left_arm_pose = np.eye(4)
        left_arm_pose[:3, :3] = R.from_quat(left_arm_quat).as_matrix()
        left_arm_pose[:3, 3] = left_arm_pos
        right_arm_pose = np.eye(4)
        right_arm_pose[:3, :3] = R.from_quat(right_arm_quat).as_matrix()
        right_arm_pose[:3, 3] = right_arm_pos

        left_state = self._hand_state(actions_clone[:, LEFT_HAND_STATE_IDX])
        right_state = self._hand_state(actions_clone[:, RIGHT_HAND_STATE_IDX])

        body_data = {LEFT_WRIST_LINK_NAME: left_arm_pose, RIGHT_WRIST_LINK_NAME: right_arm_pose}
        target_robot_joints = self.compute_upperbody_joint_positions(
            body_data, torch.tensor(float(left_state)), torch.tensor(float(right_state))
        )
        target_upper_body_joints = target_robot_joints[self.robot_model.get_joint_group_indices("upper_body")]
        self._last_ik_target_robot_joints = target_robot_joints
        if _DEBUG and getattr(self, "_dbg_step", 0) < 12:
            names = list(self.robot_model.joint_names)
            arm = [n for n in names if n.startswith("left_") and ("shoulder" in n or "elbow" in n or "wrist" in n)]
            qa = wp.to_torch(self._asset.data.joint_pos)[0].cpu().numpy()
            sim_names = list(self._asset.joint_names)
            print("[pink_brainco]   ik-out L arm:", {n: (round(float(target_robot_joints[names.index(n)]), 2), round(float(qa[sim_names.index(n)]), 2)) for n in arm}, flush=True)

        # --- lower body: AGILE WBC on a synthetic 43-DoF Dex3 view of the Revo2 robot -----------------
        navigate_cmd = self.get_navigation_cmd_from_actions(actions_clone)
        base_height_cmd = self.get_base_height_cmd_from_actions(actions_clone)
        torso_orientation_rpy_cmd = self.get_torso_orientation_rpy_cmd_from_actions(actions_clone)
        self._navigate_cmd = navigate_cmd.clone()
        self.set_wbc_goal(navigate_cmd, base_height_cmd, torso_orientation_rpy_cmd)
        self.wbc_policy.set_goal(self._wbc_goal)

        q43, dq43, qd43 = G1BraincoWBCJointAction._synthetic_wbc_joint_state(self)
        self.wbc_policy.set_observation(G1BraincoWBCJointAction._prepare_observations(self, q43, dq43, qd43))
        wbc_action = self.wbc_policy.get_action(target_upper_body_joints)

        # Same as Arena's postprocess_actions minus the per-step warning for the absent Dex3 finger joints.
        wbc_q = torch.as_tensor(wbc_action["q"], dtype=torch.float32, device=self.device)
        sim_targets = torch.zeros((self.num_envs, len(self._asset.joint_names)), device=self.device)
        for wbc_idx, sim_idx in self._wbc_to_sim:
            sim_targets[:, sim_idx] = wbc_q[:, wbc_idx]
        if self._extra_active_joint_sim_indices:
            for sim_idx, full_idx in zip(self._extra_active_joint_sim_indices, self._extra_active_joint_full_indices):
                sim_targets[:, sim_idx] = float(target_robot_joints[full_idx])

        # --- hands: binary state -> Dex3 command -> Revo2 joints ------------------------------
        states = {"left": left_state, "right": right_state}
        for side in _SIDES:
            sim_targets[:, self._revo2_idx_asset[side]] = self._revo2_target[side][states[side]]
        self._sim_targets = sim_targets

        # --- 43-DoF Dex3-space record of what was commanded -------------------------------------------
        out = self._processed_actions_43
        for wbc_idx, idx43 in self._wbc_to_43:
            out[:, idx43] = wbc_q[:, wbc_idx]
        if self._extra_active_joint_sim_indices:
            sim_names = list(self._asset.joint_names)
            for sim_idx in self._extra_active_joint_sim_indices:
                out[:, _ORDER43[sim_names[sim_idx]]] = sim_targets[:, sim_idx]
        for side in _SIDES:
            out[:, self._dex3_idx43[side]] = self._dex3_target[side][states[side]]

        if _DEBUG:
            self._log_debug(raw_left_pos, left_state)

    def _log_debug(self, target_pos, left_state: int) -> None:
        """Every 10 steps: left wrist target vs measured (anchor frame if anchored, else pelvis), PD error, fingers, apple."""
        self._dbg_step = getattr(self, "_dbg_step", 0) + 1
        if self._dbg_step % 10 != 1:
            return
        data = self._asset.data
        body_names = list(data.body_names)
        joint_names = list(self._asset.joint_names)
        wl = wp.to_torch(data.body_link_pos_w)[0, body_names.index(LEFT_WRIST_LINK_NAME)]
        root_p = wp.to_torch(data.root_link_pos_w)[0]
        if self.anchor_T_w is not None:
            meas = torch.as_tensor((np.linalg.inv(self.anchor_T_w) @ np.append(wl.cpu().numpy(), 1.0))[:3], device=wl.device, dtype=wl.dtype)
        else:
            T_inv = np.linalg.inv(self.current_root_T_w())
            meas = torch.as_tensor((T_inv @ np.append(wl.cpu().numpy(), 1.0))[:3], device=wl.device, dtype=wl.dtype)
        tgt = torch.as_tensor(np.asarray(target_pos), device=meas.device, dtype=meas.dtype)
        q = wp.to_torch(data.joint_pos)[0]
        fingers = q[self._revo2_idx_asset["left"]]
        arm = ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
               "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"]
        arm_idx = [joint_names.index(n) for n in arm]
        arm_err = (self._sim_targets[0, arm_idx] - q[arm_idx]).cpu().numpy()
        waist = [joint_names.index(n) for n in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint") if n in joint_names]
        try:
            apple = self._env.scene["apple_01_objaverse_robolab"]
            ap_w = apple.data.root_pos_w
            ap = (ap_w if isinstance(ap_w, torch.Tensor) else wp.to_torch(ap_w))[0]
            ap_rel = float(torch.linalg.norm(ap - wl))
            ap_z = float(ap[2] - root_p[2])
        except Exception:  # noqa: BLE001
            ap_rel, ap_z = float("nan"), float("nan")
        print(
            f"[pink_brainco] step {self._dbg_step:4d} hand={left_state} L tgt {tgt.cpu().numpy().round(3)} "
            f"meas {meas.cpu().numpy().round(3)} d(tgt-meas) {(tgt - meas).cpu().numpy().round(3)} | "
            f"arm PD err(rad) {arm_err.round(2)} | waist q {q[waist].cpu().numpy().round(2)} | "
            f"fingers {fingers.cpu().numpy().round(2)} | apple-wrist {ap_rel:.3f} m, apple z(rel root) {ap_z:+.3f} | "
            f"root {root_p.cpu().numpy().round(3)} nav {self._navigate_cmd[0].cpu().numpy().round(2)}",
            flush=True,
        )

    def apply_actions(self):
        self._asset.set_joint_position_target(self._sim_targets, self._joint_ids)


@configclass
class G1BraincoWBCPinkActionCfg(G1DecoupledWBCPinkActionCfg):
    class_type: type = G1BraincoWBCPinkAction
    anchor_targets: bool = True
    """Interpret wrist targets in the fixed source-pelvis frame (ANCHOR_T_W) instead of the live pelvis frame."""


@configclass
class G1BraincoPinkActionCfg:
    """Same IK active set as ``G1WBCAgilePinkActionCfg`` (arms + waist), Revo2 action term."""

    g1_action: G1BraincoWBCPinkActionCfg = G1BraincoWBCPinkActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        wbc_version="agile",
        upperbody_active_joint_groups=["arms"],
        upperbody_extra_active_joints=["waist_roll_joint", "waist_yaw_joint", "waist_pitch_joint"],
    )


@configclass
class G1BraincoPinkObservationsCfg(G1WBCPinkObservationsCfg):
    """Pink observations with the joint state reported in the 43-DoF Dex3 space (the convention the policy expects)."""

    @configclass
    class PolicyCfg(G1WBCPinkObservationsCfg.PolicyCfg):
        robot_joint_pos = ObsTerm(func=brainco_joint_pos_43dof, params={"asset_cfg": SceneEntityCfg("robot")})

    @configclass
    class WBCObsCfg(G1WBCPinkObservationsCfg.WBCObsCfg):
        robot_joint_pos = ObsTerm(func=brainco_joint_pos_43dof, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_joint_vel = ObsTerm(func=brainco_joint_vel_43dof, params={"asset_cfg": SceneEntityCfg("robot")})

    policy: PolicyCfg = PolicyCfg()
    wbc: WBCObsCfg = WBCObsCfg()


@register_asset
class G1WBCAgilePinkBraincoEmbodiment(G1WBCAgilePinkEmbodiment):
    """AGILE WBC + Pink IK upper body on the G1 with BrainCo Revo2 Touch hands (Mimic source/generation)."""

    name = "g1_wbc_agile_pink_brainco"

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        camera_offset: Pose | None = _DEFAULT_G1_CAMERA_OFFSET,
        use_tiled_camera: bool = False,
        lock_waist: bool = False,
    ):
        super().__init__(
            enable_cameras=enable_cameras,
            initial_pose=initial_pose,
            camera_offset=camera_offset,
            use_tiled_camera=use_tiled_camera,
            lock_waist=lock_waist,
        )
        cam = getattr(self.camera_config, "robot_head_cam", None)
        if hasattr(cam, "prim_path"):
            cam.prim_path = f"{BRAINCO_HEAD_LINK_PATH}/RobotHeadCam"

        self.scene_config = G1BraincoSceneCfg()
        self.action_config = G1BraincoPinkActionCfg()
        if lock_waist:
            _remove_waist_from_pink_ik_action_config(self.action_config)
        self.observation_config = G1BraincoPinkObservationsCfg()
        self.observation_config.policy.concatenate_terms = self.concatenate_observation_terms
        self.observation_config.wbc.concatenate_terms = self.concatenate_observation_terms
        self.observation_config.action.concatenate_terms = self.concatenate_observation_terms

    def set_finger_contact_friction(self, material_path, static_friction, dynamic_friction, prim_name_markers):
        markers = tuple(prim_name_markers) + ("ring", "pinky")
        return super().set_finger_contact_friction(
            material_path=material_path,
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            prim_name_markers=markers,
        )
