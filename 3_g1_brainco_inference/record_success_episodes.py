"""Run many episodes and save each successful one as its own labelled clip.

Recording every episode into one file is fine at the baseline's 65% success but useless at a few
percent: a 20-episode run is 6000 frames of near-identical failures and, in three attempts,
contained no success at all.

This buffers frames per episode and writes a clip only when the episode ends in success, plus the
first few failures for the failure-mode record. A long run then yields short labelled clips of the
task being completed rather than a haystack.

The head camera is used rather than the Kit viewport because `env.render()` comes back essentially
black when running headless, and because the head camera is what the policy is conditioned on.

Copy into the IsaacLab-Arena checkout root and run with /isaac-sim/python.sh; see README.md.
"""

import os

import numpy as np
import torch
import tqdm

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.evaluation.policy_runner import get_policy_cls
from isaaclab_arena.evaluation.policy_runner_cli import add_policy_runner_arguments
from isaaclab_arena.metrics.metrics_logger import metrics_to_plain_python_types
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena_environments.cli import (
    get_arena_builder_from_cli,
    get_isaaclab_arena_environments_cli_parser,
)
from isaaclab_arena_gr00t.utils.groot_path import ensure_groot_deps_in_path


def to_uint8_rgb(frame) -> np.ndarray | None:
    """Normalise a torch or numpy camera frame of unknown dtype and layout to HxWx3 uint8."""
    if frame is None:
        return None
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:  # (num_envs, H, W, C)
        frame = frame[0]
    if frame.ndim != 3:
        return None
    if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (3, 4):  # CHW -> HWC
        frame = np.transpose(frame, (1, 2, 0))
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        # Camera observations may be float in [0,1] or [0,255] depending on `normalize`.
        peak = float(frame.max()) if frame.size else 0.0
        frame = frame * 255.0 if peak <= 1.0 else frame
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def find_ego(obs):
    """Pull the head-camera RGB out of the observation dict, wherever it is nested."""
    if not isinstance(obs, dict):
        return None
    for key in ("camera_obs", "policy", "obs"):
        sub = obs.get(key)
        if isinstance(sub, dict):
            for name, val in sub.items():
                if "cam" in name and "rgb" in name:
                    return val
    for name, val in obs.items():
        if "cam" in name and "rgb" in name:
            return val
    return None


def write_mp4(path: str, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        print(f"  no frames for {path}, skipping")
        return
    import cv2

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    mean = float(np.mean([f.mean() for f in frames]))
    print(f"  wrote {path}  ({len(frames)} frames, {w}x{h}, mean pixel {mean:.1f})")


def main() -> None:
    args_parser = get_isaaclab_arena_cli_parser()
    args_parser.add_argument("--out_dir", type=str, default="/eval/success_episodes")
    args_parser.add_argument("--record_episodes", type=int, default=100)
    args_parser.add_argument(
        "--keep_failures",
        type=int,
        default=3,
        help="also save this many failure clips, for the failure-mode record",
    )
    args_cli, _ = args_parser.parse_known_args()

    with SimulationAppContext(args_cli):
        add_policy_runner_arguments(args_parser)
        args_cli, _ = args_parser.parse_known_args()
        policy_cls = get_policy_cls(args_cli.policy_type)
        args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
        args_parser = policy_cls.add_args_to_parser(args_parser)
        args_cli = args_parser.parse_args()

        arena_builder = get_arena_builder_from_cli(args_cli)
        env, _ = arena_builder.make_registered_and_return_cfg(render_mode="rgb_array")
        policy = policy_cls.from_args(args_cli)
        os.makedirs(args_cli.out_dir, exist_ok=True)

        obs, _ = env.reset()
        policy.reset()
        task_description = (
            args_cli.language_instruction
            or env.unwrapped.cfg.isaaclab_arena_env.task.get_task_description()
        )
        policy.set_task_description(task_description)
        print(f"task description: {task_description!r}", flush=True)

        fps = 25
        ep_frames: list[np.ndarray] = []
        episodes_done = 0
        failures_kept = 0
        results: list[bool] = []

        pbar = tqdm.tqdm(total=args_cli.record_episodes, desc="Episodes", unit="ep")
        while episodes_done < args_cli.record_episodes:
            with torch.inference_mode():
                actions = policy.get_action(env, obs)
                obs, _, terminated, truncated, _ = env.step(actions)

                ego = to_uint8_rgb(find_ego(obs))
                if ego is not None:
                    ep_frames.append(ego)

                if terminated.any() or truncated.any():
                    # `terminated` is any termination, timeout included, so it cannot be used as
                    # the success flag. The task's own "success" term is what the metric records.
                    succeeded = bool(
                        env.unwrapped.termination_manager.get_term("success")[0].item()
                    )
                    results.append(succeeded)
                    episodes_done += 1

                    if succeeded:
                        write_mp4(
                            os.path.join(args_cli.out_dir, f"SUCCESS_ep{episodes_done:03d}.mp4"),
                            ep_frames,
                            fps,
                        )
                        print(f"SUCCESS_EPISODE {episodes_done}", flush=True)
                    elif failures_kept < args_cli.keep_failures:
                        failures_kept += 1
                        write_mp4(
                            os.path.join(args_cli.out_dir, f"fail_ep{episodes_done:03d}.mp4"),
                            ep_frames,
                            fps,
                        )

                    ep_frames = []
                    pbar.update(1)
                    pbar.set_postfix_str(f"{sum(results)} success / {len(results)}")
                    env_ids = (terminated | truncated).nonzero().flatten()
                    policy.reset(env_ids=env_ids)
        pbar.close()

        if getattr(env.unwrapped.cfg, "metrics", None) is not None:
            from isaaclab_arena.metrics.metrics import compute_metrics

            print(f"FINAL METRICS: {metrics_to_plain_python_types(compute_metrics(env.unwrapped))}")
        print(f"PER-EPISODE: {results}  ({sum(results)}/{len(results)} success)", flush=True)

        if policy.is_remote:
            policy.shutdown_remote(kill_server=False)
        env.close()


if __name__ == "__main__":
    ensure_groot_deps_in_path()
    main()
