#
# ``isaaclab_arena/scripts/imitation_learning/generate_dataset.py`` with two additions needed to
# run one generation process per GPU and merge afterwards:
#
#   --datagen_seed N       overrides ``datagen_config.seed`` (default 1 in every Arena task) so the
#                          three GPU workers sample different subtask boundaries / noise / apple XY.
#   --max_num_failures N   overrides ``datagen_config.max_num_failures``.
#   --status_file PATH     live JSON heartbeat {success, attempts, failures, target, elapsed_s},
#                          read from isaaclab_mimic.datagen.generation's module counters.
#
# Everything else (env setup, recorders, async env_loop) is unchanged from upstream.

from __future__ import annotations

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

parser = get_isaaclab_arena_cli_parser()
parser.add_argument("--generation_num_trials", type=int, required=True, help="Number of demos to be generated.")
parser.add_argument("--input_file", type=str, required=True, help="Annotated source dataset.")
parser.add_argument("--output_file", type=str, default="./datasets/output_dataset.hdf5")
parser.add_argument("--pause_subtask", action="store_true", help="Pause after every subtask (debug, needs render).")
parser.add_argument("--datagen_seed", type=int, default=None, help="Override datagen_config.seed.")
parser.add_argument("--max_num_failures", type=int, default=None, help="Override datagen_config.max_num_failures.")
parser.add_argument("--status_file", type=str, default=None, help="Write a live JSON status here.")
parser.add_argument("--status_interval_s", type=float, default=20.0)
add_example_environments_cli_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import asyncio
import gymnasium as gym
import inspect
import json
import logging
import numpy as np
import os
import random
import threading
import time
import torch

import isaaclab_mimic.envs  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode
from isaaclab_mimic.datagen import generation as mimic_generation
from isaaclab_mimic.datagen.generation import env_loop, setup_async_generation
from isaaclab_mimic.datagen.utils import setup_output_paths

from isaaclab_arena.utils.isaaclab_utils.recorders import ArenaEnvRecorderManagerCfg

logger = logging.getLogger(__name__)


def setup_env_config(output_dir: str, output_file_name: str):
    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()

    env_cfg.datagen_config.generation_num_trials = args_cli.generation_num_trials
    if args_cli.datagen_seed is not None:
        env_cfg.datagen_config.seed = args_cli.datagen_seed
    if args_cli.max_num_failures is not None:
        env_cfg.datagen_config.max_num_failures = args_cli.max_num_failures
    env_cfg.env_name = env_name

    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        raise NotImplementedError("No success termination term was found in the environment.")

    env_cfg.terminations = None
    env_cfg.observations.policy.concatenate_terms = False

    if args_cli.enable_cameras:
        env_cfg.recorders = ArenaEnvRecorderManagerCfg()
    else:
        env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    if env_cfg.datagen_config.generation_keep_failed:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES
    else:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    return env_cfg, env_name, success_term


class _StatusWriter(threading.Thread):
    """Periodically dump isaaclab_mimic's global counters to ``--status_file``."""

    def __init__(self, path: str, target: int, interval_s: float):
        super().__init__(daemon=True)
        self.path, self.target, self.interval_s = path, target, interval_s
        self.t0 = time.time()
        self._stop = threading.Event()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def write(self, state: str) -> None:
        payload = {
            "state": state,
            "success": int(mimic_generation.num_success),
            "attempts": int(mimic_generation.num_attempts),
            "failures": int(mimic_generation.num_failures),
            "target": self.target,
            "elapsed_s": round(time.time() - self.t0, 1),
            "seed": args_cli.datagen_seed,
            "num_envs": args_cli.num_envs,
            "output_file": args_cli.output_file,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, self.path)

    def run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.write("running")
            except Exception as e:  # noqa: BLE001
                print(f"[status] write failed: {e}")

    def stop(self, state: str) -> None:
        self._stop.set()
        try:
            self.write(state)
        except Exception:  # noqa: BLE001
            pass


def main():
    output_dir, output_file_name = setup_output_paths(args_cli.output_file)
    env_cfg, env_name, success_term = setup_env_config(output_dir, output_file_name)

    env = gym.make(env_name, cfg=env_cfg)
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import reapply_viewer_cfg

    reapply_viewer_cfg(env)
    env = env.unwrapped
    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")
    if "action_noise_dict" not in inspect.signature(env.target_eef_pose_to_action).parameters:
        logger.warning("target_eef_pose_to_action uses the deprecated 'noise' parameter.")

    seed = env.cfg.datagen_config.seed
    print(f"[generate] datagen seed={seed} num_envs={args_cli.num_envs} target={args_cli.generation_num_trials}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env.reset()

    status = None
    if args_cli.status_file:
        status = _StatusWriter(args_cli.status_file, args_cli.generation_num_trials, args_cli.status_interval_s)
        status.write("starting")
        status.start()

    final_state = "finished"
    try:
        async_components = setup_async_generation(
            env=env,
            num_envs=args_cli.num_envs,
            input_file=args_cli.input_file,
            success_term=success_term,
            pause_subtask=args_cli.pause_subtask,
            motion_planners=None,
        )
        try:
            data_gen_tasks = asyncio.ensure_future(asyncio.gather(*async_components["tasks"]))
            env_loop(
                env,
                async_components["reset_queue"],
                async_components["action_queue"],
                async_components["info_pool"],
                async_components["event_loop"],
            )
        except asyncio.CancelledError:
            print("Tasks were cancelled.")
        finally:
            data_gen_tasks.cancel()
            try:
                async_components["event_loop"].run_until_complete(data_gen_tasks)
            except asyncio.CancelledError:
                print("Remaining async tasks cancelled and cleaned up.")
            except Exception as e:  # noqa: BLE001
                print(f"Error cancelling remaining async tasks: {e}")
    except BaseException:
        final_state = "error"
        raise
    finally:
        if status is not None:
            status.stop(final_state)
        env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    except Exception as e:  # noqa: BLE001
        print(f"\nError occurred: {e}")
        import traceback

        traceback.print_exc()
        os._exit(1)
    simulation_app.close()
