"""Pick-and-place task variant that carries a G1-compatible Mimic config.

Why a subclass: ``PickAndPlaceTask.get_mimic_env_cfg`` only knows the single-arm / GR1 left-right
layouts and reads ``destination_object.name`` (``None`` in the static apple env). ``G1MimicEnv``
instead expects three end-effectors -- ``left``, ``right`` and ``body`` (the 7 lower-body command
dims) -- exactly like ``G1LocomanipPickPlaceMimicEnvCfg``. This file supplies that graph for the
static apple task and adds a ``subtask_terms`` observation group so annotation can run ``--auto``.
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.tasks.common.mimic_default_params import MIMIC_DATAGEN_CONFIG_DEFAULTS
from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

from g1_apple_mimic import mdp as apple_mdp

WRIST_LINK_BY_ARM = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}


class G1StaticApplePickPlaceTask(PickAndPlaceTask):
    """``PickAndPlaceTask`` with a G1 (left/right/body) Mimic subtask graph and an auto grasp signal."""

    def __init__(
        self,
        pick_up_object: Asset,
        destination_location: Asset,
        background_scene: Asset,
        mimic_arm: str = "left",
        object_spawn_z_env: float = 0.0,
        lift_height_m: float = 0.03,
        max_hand_dist_m: float = 0.18,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        force_threshold: float = 1.0,
        velocity_threshold: float = 0.1,
    ):
        super().__init__(
            pick_up_object=pick_up_object,
            destination_location=destination_location,
            background_scene=background_scene,
            destination_object=None,
            episode_length_s=episode_length_s,
            task_description=task_description,
            force_threshold=force_threshold,
            velocity_threshold=velocity_threshold,
        )
        if mimic_arm not in WRIST_LINK_BY_ARM:
            raise ValueError(f"mimic_arm must be one of {sorted(WRIST_LINK_BY_ARM)}, got {mimic_arm!r}")
        self.mimic_arm = mimic_arm
        self.grasp_signal_name = f"grasp_{mimic_arm}"
        self.observation_cfg = G1StaticAppleSubtaskObservationsCfg(
            signal_name=self.grasp_signal_name,
            object_name=pick_up_object.name,
            wrist_link_name=WRIST_LINK_BY_ARM[mimic_arm],
            spawn_z_env=object_spawn_z_env,
            lift_height_m=lift_height_m,
            max_hand_dist_m=max_hand_dist_m,
        )

    def get_observation_cfg(self):
        return self.observation_cfg

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        # arm_mode is G1's DUAL_ARM; the graph below is fixed by ``mimic_arm`` instead.
        del arm_mode
        return G1StaticAppleMimicEnvCfg(
            active_arm=self.mimic_arm,
            grasp_signal_name=self.grasp_signal_name,
            pick_up_object_name=self.pick_up_object.name,
            destination_location_name=self.destination_location.name,
        )


@configclass
class _LeftGraspSubtaskTermsCfg(ObsGroup):
    grasp_left: ObsTerm = ObsTerm(func=apple_mdp.object_grasped_by_hand, params={})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class _RightGraspSubtaskTermsCfg(ObsGroup):
    grasp_right: ObsTerm = ObsTerm(func=apple_mdp.object_grasped_by_hand, params={})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class G1StaticAppleSubtaskObservationsCfg:
    """Adds the ``subtask_terms`` group read by ``G1StaticAppleMimicEnv.get_subtask_term_signals``."""

    subtask_terms: ObsGroup = MISSING

    def __init__(
        self,
        signal_name: str,
        object_name: str,
        wrist_link_name: str,
        spawn_z_env: float,
        lift_height_m: float,
        max_hand_dist_m: float,
    ):
        group_cls = {"grasp_left": _LeftGraspSubtaskTermsCfg, "grasp_right": _RightGraspSubtaskTermsCfg}[signal_name]
        self.subtask_terms = group_cls()
        getattr(self.subtask_terms, signal_name).params = {
            "object_cfg": SceneEntityCfg(object_name),
            "wrist_link_name": wrist_link_name,
            "spawn_z_env": spawn_z_env,
            "lift_height_m": lift_height_m,
            "max_hand_dist_m": max_hand_dist_m,
        }


@configclass
class G1StaticAppleMimicEnvCfg(MimicEnvCfg):
    """Mimic subtask graph for G1 static apple-to-plate: one active arm, one idle arm, static body."""

    active_arm: str = "left"
    grasp_signal_name: str = "grasp_left"
    pick_up_object_name: str = "apple_01_objaverse_robolab"
    destination_location_name: str = "clay_plates_hot3d_robolab"

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = "g1_static_apple_pick_place_D0"
        for key, value in MIMIC_DATAGEN_CONFIG_DEFAULTS.items():
            setattr(self.datagen_config, key, value)

        common = dict(
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": 3},
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )

        # Active arm: reach+grasp the apple (relative to the apple), then carry+place (relative to the plate).
        self.subtask_configs[self.active_arm] = [
            SubTaskConfig(
                object_ref=self.pick_up_object_name,
                subtask_term_signal=self.grasp_signal_name,
                subtask_term_offset_range=(10, 20),
                action_noise=0.005,
                num_interpolation_steps=5,
                **common,
            ),
            SubTaskConfig(
                object_ref=self.destination_location_name,
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                action_noise=0.005,
                num_interpolation_steps=5,
                **common,
            ),
        ]

        # Idle arm: single segment, replayed as-is (no noise) so it stays parked out of the camera view.
        idle_arm = "right" if self.active_arm == "left" else "left"
        self.subtask_configs[idle_arm] = [
            SubTaskConfig(
                object_ref=self.pick_up_object_name,
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                action_noise=0.0,
                num_interpolation_steps=0,
                **common,
            )
        ]

        # Body: required by G1MimicEnv (nav / base-height / torso commands). Static task -> replay as-is.
        self.subtask_configs["body"] = [
            SubTaskConfig(
                object_ref=self.pick_up_object_name,
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                action_noise=0.0,
                num_interpolation_steps=0,
                **common,
            )
        ]
