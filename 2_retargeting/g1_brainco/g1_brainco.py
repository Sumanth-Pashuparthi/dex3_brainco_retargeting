"""Embodiment: Unitree G1 29-DoF wearing BrainCo Revo2 Touch hands.

Identical to ``g1_wbc_agile_joint`` except for the hands:

    kinematics / collision / inertia   USD converted from g1_29dof_with_brainco_hand.urdf,
                                       6 revolute DoF per hand instead of 7, five fingers instead
                                       of three, distal phalanges rigid
    actuator limits                    Revo2 effort/velocity limits replace the Dex3 "hands" group,
                                       whose ``.*_hand_.*`` pattern matches nothing on this robot
    controller interface               G1BraincoWBCJointAction retargets the frozen policy's Dex3
                                       commands and reports pseudo-Dex3 state back

Body actuator gains, whole-body controller, cameras, scene and task are inherited unchanged so a
baseline comparison isolates the hand swap.
"""

from __future__ import annotations

import torch

import isaaclab.sim.schemas.schemas_cfg as sim_schemas
import warp as wp
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.g1.g1 import (
    _DEFAULT_G1_CAMERA_OFFSET,
    G1_AGILE_CFG,
    G1AgileSceneCfg,
    G1WBCAgileJointEmbodiment,
    G1WBCJointObservationsCfg,
)
from isaaclab_arena.utils.pose import Pose

from .brainco_action import G1BraincoWBCJointActionCfg, _load_43dof_order
from .hand_retarget import BRAINCO_ACTIVE_ORDER, DEX3_GROOT_ORDER, inverse_batch

_SIDES = ("left", "right")
_ORDER43 = _load_43dof_order()
_BODY_NAMES = [n for n in _ORDER43 if "_hand_" not in n]

BRAINCO_G1_USD = (
    "/workspaces/isaaclab_arena/assets/brainco/g1_29dof_with_brainco_hand/"
    "g1_29dof_with_brainco_hand.usda"
)

# The stock G1 asset is authored flat, so every link is a direct child of the robot root and the
# default camera path `{ENV_REGEX_NS}/Robot/head_link/RobotHeadCam` resolves. The URDF converter
# instead nests links by kinematic tree under a `Geometry` scope, so that path matches nothing and
# USD quietly creates an empty Xform at the robot origin: the camera then sits at the pelvis
# staring at the robot's own arms instead of riding the head.
BRAINCO_HEAD_LINK_PATH = (
    "{ENV_REGEX_NS}/Robot/Geometry/pelvis/waist_yaw_link/waist_roll_link/torso_link/head_link"
)


def _joint_state_43dof(env, asset_cfg: SceneEntityCfg, velocity: bool) -> torch.Tensor:
    """Report the Revo2 robot in the 43-DoF Dex3 joint space the frozen policy consumes.

    Body joints pass through; each hand's 6 measured Revo2 angles are inverted to the 7 pseudo-Dex3
    angles producing the same finger closure. Ordering matches ``43dof_joint_space.yaml``, which is
    what the policy client uses to remap sim state to policy state, so no client change is needed.
    """
    asset = env.scene[asset_cfg.name]
    names = list(asset.joint_names)
    raw = asset.data.joint_vel if velocity else asset.data.joint_pos
    # ArticulationData exposes warp arrays here, not torch tensors.
    q = raw if isinstance(raw, torch.Tensor) else wp.to_torch(raw)
    out = torch.zeros((q.shape[0], len(_ORDER43)), dtype=q.dtype, device=q.device)

    for name in _BODY_NAMES:
        out[:, _ORDER43[name]] = q[:, names.index(name)]

    if velocity:
        # Finger velocities are not part of the policy's effective state; leave them zero rather
        # than inventing a chain rule through the closure map.
        return out

    q_np = q.detach().cpu().numpy()
    for side in _SIDES:
        idx = [names.index(f"{side}_{j}") for j in BRAINCO_ACTIVE_ORDER]
        pseudo = inverse_batch(q_np[:, idx], side)
        for j, jname in enumerate(DEX3_GROOT_ORDER):
            out[:, _ORDER43[f"{side}_{jname}"]] = torch.as_tensor(
                pseudo[:, j], dtype=q.dtype, device=q.device
            )
    return out


def brainco_joint_pos_43dof(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return _joint_state_43dof(env, asset_cfg, velocity=False)


def brainco_joint_vel_43dof(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return _joint_state_43dof(env, asset_cfg, velocity=True)


@configclass
class G1BraincoObservationsCfg(G1WBCJointObservationsCfg):
    """Baseline observations, with the joint state expressed in Dex3 space."""

    @configclass
    class PolicyCfg(G1WBCJointObservationsCfg.PolicyCfg):
        robot_joint_pos = ObsTerm(
            func=brainco_joint_pos_43dof, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        robot_joint_vel = ObsTerm(
            func=brainco_joint_vel_43dof, params={"asset_cfg": SceneEntityCfg("robot")}
        )

    @configclass
    class WBCObsCfg(G1WBCJointObservationsCfg.WBCObsCfg):
        robot_joint_pos = ObsTerm(
            func=brainco_joint_pos_43dof, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        robot_joint_vel = ObsTerm(
            func=brainco_joint_vel_43dof, params={"asset_cfg": SceneEntityCfg("robot")}
        )

    policy: PolicyCfg = PolicyCfg()
    wbc: WBCObsCfg = WBCObsCfg()


def _build_brainco_articulation() -> ArticulationCfg:
    cfg = G1_AGILE_CFG.copy()

    # The URDF converter authors a `PhysicsFixedJoint "root_joint"` welding the world to the pelvis
    # even when invoked with fix_base=False, which pins root_link_pos_w.z to 0.0 so the G1 never
    # stands. That joint is deactivated in the asset itself (see 2_retargeting/patch_converted_usd.py),
    # because `fix_root_link` only governs the fixed joint Isaac Lab manages and leaves a
    # converter-authored one composed in. This flag keeps anything from re-adding one.
    props = cfg.spawn.articulation_props
    props = (
        props.replace(fix_root_link=False)
        if props is not None
        else sim_schemas.ArticulationRootPropertiesCfg(fix_root_link=False)
    )
    cfg.spawn = cfg.spawn.replace(usd_path=BRAINCO_G1_USD, articulation_props=props)

    actuators = dict(cfg.actuators)
    # The Dex3 "hands" group keys off `.*_hand_.*`, which matches no joint on the Revo2 robot.
    actuators.pop("hands", None)

    # PhysX joint velocity caps. The stock G1 USD carries `maxJointVelocity` on every joint (hips
    # 32, knees 20, ankles/arms 37, wrists 22 rad/s) and Isaac Lab keeps the USD value when
    # `velocity_limit_sim` is None, so those caps are silently part of the baseline plant. The URDF
    # converter authors none, so PhysX loads 5.9e36 (unlimited) on all 29 body joints. The
    # `velocity_limit` values in the AGILE config are exactly the stock USD values.
    for name, act in list(actuators.items()):
        if act.velocity_limit is not None and act.velocity_limit_sim is None:
            actuators[name] = act.replace(velocity_limit_sim=act.velocity_limit)

    # effort_limit matches the baseline Dex3 "hands" group (5.0 N.m) rather than the Revo2 URDF's
    # per-joint 0.5-2.0 N.m. Those URDF numbers are the outlier: BrainCo rate the Revo2 at 50 N
    # grasp / 15 N pinch, comparable to the Dex3, and capping the fingers at 2 N.m would give this
    # hand less holding authority than the baseline had, confounding the comparison. Velocity
    # limits stay at the datasheet values.
    actuators["revo2_thumb"] = IdealPDActuatorCfg(
        joint_names_expr=[".*_thumb_metacarpal_joint", ".*_thumb_proximal_joint"],
        effort_limit=5.0,
        velocity_limit={".*_thumb_metacarpal_joint": 2.6175, ".*_thumb_proximal_joint": 2.5303},
        velocity_limit_sim={".*_thumb_metacarpal_joint": 2.6175, ".*_thumb_proximal_joint": 2.5303},
        stiffness=4.0,
        damping=0.5,
        armature=0.03,
        friction=0.03,
    )
    actuators["revo2_fingers"] = IdealPDActuatorCfg(
        joint_names_expr=[
            ".*_index_proximal_joint",
            ".*_middle_proximal_joint",
            ".*_ring_proximal_joint",
            ".*_pinky_proximal_joint",
        ],
        effort_limit=5.0,
        velocity_limit=2.2685,
        velocity_limit_sim=2.2685,
        stiffness=4.0,
        damping=0.5,
        armature=0.03,
        friction=0.03,
    )
    cfg.actuators = actuators
    return cfg


G1_BRAINCO_AGILE_CFG = _build_brainco_articulation()


@configclass
class G1BraincoSceneCfg(G1AgileSceneCfg):
    robot: ArticulationCfg = G1_BRAINCO_AGILE_CFG


@configclass
class G1BraincoActionCfg:
    g1_action: G1BraincoWBCJointActionCfg = G1BraincoWBCJointActionCfg(
        asset_name="robot", joint_names=[".*"], wbc_version="agile"
    )


@register_asset
class G1WBCAgileJointBraincoEmbodiment(G1WBCAgileJointEmbodiment):
    """The stock G1 / AGILE stack with Dex3 hands swapped for BrainCo Revo2 Touch."""

    name = "g1_wbc_agile_joint_brainco"

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        camera_offset: Pose | None = _DEFAULT_G1_CAMERA_OFFSET,
        use_tiled_camera: bool = True,
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
        self.action_config = G1BraincoActionCfg()
        self.observation_config = G1BraincoObservationsCfg()
        self.observation_config.policy.concatenate_terms = self.concatenate_observation_terms
        self.observation_config.wbc.concatenate_terms = self.concatenate_observation_terms

    def set_finger_contact_friction(
        self, material_path, static_friction, dynamic_friction, prim_name_markers
    ):
        # The stock marker list stops at "middle" because the Dex3 has three fingers; the Revo2
        # also has ring and pinky, which carry load in a power grasp.
        return super().set_finger_contact_friction(
            material_path=material_path,
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            prim_name_markers=tuple(prim_name_markers) + ("ring", "pinky"),
        )
