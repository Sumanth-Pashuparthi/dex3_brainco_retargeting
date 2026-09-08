"""``galileo_g1_static_apple_mimic``: the static apple env, Mimic-enabled.

Same scene / embodiment / spawn poses / success term as ``galileo_g1_static_pick_and_place`` (it is
built by calling that env's ``get_env`` and then swapping the task object), plus:

* a ``G1StaticApplePickPlaceTask`` carrying the left/right/body Mimic subtask graph and the
  ``grasp_<arm>`` auto-annotation signal,
* ``G1StaticAppleMimicEnv`` as the embodiment's mimic env class,
* optional apple XY spawn jitter (``--apple_xy_range_m``) so generated demos are spatially diverse.
  The upstream env ships with ``APPLE_SPAWN_XY_RANGE_M = 0.0``; Mimic on a fixed apple pose would
  only add action noise.

Usage (any Arena imitation_learning script)::

    --external_environment_class_path g1_apple_mimic.environment:GalileoG1StaticAppleMimicEnvironment \
    --mimic galileo_g1_static_apple_mimic --mimic_arm left [--apple_xy_range_m 0.02] \
    --embodiment g1_wbc_agile_pink_brainco   # Revo2 hands (g1_apple_mimic.brainco_pink); default is Dex3
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from isaaclab_arena_environments.galileo_g1_static_pick_and_place_environment import (
    PICK_UP_OBJECT_SPAWN_XY,
    GalileoG1StaticPickAndPlaceEnvironment,
    _shelf_spawn_z,
)

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


class GalileoG1StaticAppleMimicEnvironment(GalileoG1StaticPickAndPlaceEnvironment):

    name: str = "galileo_g1_static_apple_mimic"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        from isaaclab_arena.utils.pose import PoseRange

        import g1_apple_mimic.brainco_pink  # noqa: F401  registers g1_wbc_agile_pink_brainco before --embodiment lookup
        from g1_apple_mimic.mimic_env import G1StaticAppleMimicEnv
        from g1_apple_mimic.task import G1StaticApplePickPlaceTask

        arena_env = super().get_env(args_cli)
        base_task = arena_env.task
        arena_env.name = self.name

        spawn_z = _shelf_spawn_z(args_cli.object)

        # Per-episode apple XY jitter for generation diversity. Annotation replays the recorded
        # initial_state via reset_to(), so it is unaffected by this randomization.
        if args_cli.apple_xy_range_m > 0.0:
            x0, y0 = PICK_UP_OBJECT_SPAWN_XY
            r = args_cli.apple_xy_range_m
            base_task.pick_up_object.set_initial_pose(
                PoseRange(
                    position_xyz_min=(x0 - r, y0 - r, spawn_z),
                    position_xyz_max=(x0 + r, y0 + r, spawn_z),
                    rpy_min=(0.0, 0.0, 0.0),
                    rpy_max=(0.0, 0.0, 0.0),
                )
            )

        arena_env.task = G1StaticApplePickPlaceTask(
            pick_up_object=base_task.pick_up_object,
            destination_location=base_task.destination_location,
            background_scene=base_task.background_scene,
            mimic_arm=args_cli.mimic_arm,
            object_spawn_z_env=spawn_z,
            lift_height_m=args_cli.lift_height_m,
            max_hand_dist_m=args_cli.hand_dist_m,
            episode_length_s=base_task.get_episode_length_s(),
            task_description=base_task.get_task_description(),
            force_threshold=base_task.force_threshold,
            velocity_threshold=base_task.velocity_threshold,
        )
        arena_env.embodiment.mimic_env = G1StaticAppleMimicEnv
        return arena_env

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        GalileoG1StaticPickAndPlaceEnvironment.add_cli_args(parser)
        parser.add_argument(
            "--mimic_arm",
            type=str,
            default="left",
            choices=("left", "right"),
            help="Arm that performs the pick in the source demos (the other arm is treated as idle).",
        )
        parser.add_argument(
            "--apple_xy_range_m",
            type=float,
            default=0.02,
            help="Half-range of per-episode apple XY spawn jitter used during generation. 0 disables.",
        )
        parser.add_argument(
            "--lift_height_m",
            type=float,
            default=0.03,
            help="grasp_<arm> signal: apple must be this far above its shelf spawn height.",
        )
        parser.add_argument(
            "--hand_dist_m",
            type=float,
            default=0.18,
            help="grasp_<arm> signal: apple origin must be within this distance of the wrist link.",
        )
