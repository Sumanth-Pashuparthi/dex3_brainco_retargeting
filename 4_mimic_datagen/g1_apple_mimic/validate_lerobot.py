"""Check a converted GR00T-LeRobot dataset the way ``launch_finetune.py`` will consume it.

Run inside the Isaac-GR00T (N1.7) venv, NOT the Arena container::

    .venv/bin/python /path/to/g1_apple_mimic/validate_lerobot.py \
        --dataset_path /path/to/g1_apple_mimic_generated/lerobot \
        --modality_config_path /path/to/isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_gr00t_n_1_7_config.py

Checks:
  * meta/{info.json, episodes.jsonl, tasks.jsonl, modality.json, stats.json, relative_stats.json}
  * every episode has a parquet + an mp4; parquet rows == episodes.jsonl length
  * observation.state / action are 43-D and finite; modality.json slices cover the WBC layout
  * teleop.* and *.eef_pose columns present (needed by the Arena modality config)
  * GR00T's own LeRobotEpisodeLoader (with the registered NEW_EMBODIMENT config) loads a few
    episodes end-to-end, including video decode, and reports shapes
Exit 0 = ready for finetuning, 1 = problems.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_META = ["info.json", "episodes.jsonl", "tasks.jsonl", "modality.json", "stats.json"]
REQUIRED_COLUMNS = [
    "observation.state",
    "action",
    "observation.eef_pose",
    "action.eef_pose",
    "teleop.base_height_command",
    "teleop.navigate_command",
    "teleop.torso_orientation_rpy_command",
    "annotation.human.task_description",
    "timestamp",
    "episode_index",
    "frame_index",
    "index",
    "task_index",
    "next.done",
]
STATE_DIM = ACTION_DIM = 43


def read_jsonl(p: Path):
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--modality_config_path", required=True)
    ap.add_argument("--embodiment_tag", default="new_embodiment")
    ap.add_argument("--num_probe_episodes", type=int, default=3)
    ap.add_argument("--min_episodes", type=int, default=1)
    args = ap.parse_args()

    root = Path(args.dataset_path)
    meta = root / "meta"
    problems: list[str] = []

    print(f"dataset: {root}")
    for name in REQUIRED_META:
        if not (meta / name).exists():
            problems.append(f"missing meta/{name}")
    if not (meta / "relative_stats.json").exists():
        print("note: meta/relative_stats.json missing (finetune generates it on rank 0; convert_to_lerobot.sh runs stats.py)")
    if problems:
        for p in problems:
            print("  -", p)
        return 1

    info = json.load(open(meta / "info.json"))
    episodes = read_jsonl(meta / "episodes.jsonl")
    tasks = read_jsonl(meta / "tasks.jsonl")
    modality = json.load(open(meta / "modality.json"))
    print(f"robot_type={info.get('robot_type')} fps={info.get('fps')} episodes={len(episodes)} "
          f"frames={info.get('total_frames')} tasks={[t['task'] for t in tasks]}")
    if len(episodes) < args.min_episodes:
        problems.append(f"only {len(episodes)} episodes (< {args.min_episodes})")
    if info.get("total_episodes") != len(episodes):
        problems.append(f"info.total_episodes={info.get('total_episodes')} != episodes.jsonl {len(episodes)}")

    # modality layout must be the WBC one the Arena config expects
    for group in ("left_arm", "right_arm", "left_hand", "right_hand", "waist"):
        for mod in ("state", "action"):
            if group not in modality[mod]:
                problems.append(f"modality.json {mod} lacks {group}")
    for key in ("base_height_command", "navigate_command"):
        if key not in modality["action"]:
            problems.append(f"modality.json action lacks {key}")
    if "ego_view" not in modality.get("video", {}):
        problems.append("modality.json video lacks ego_view")

    # per-episode files + parquet content
    lengths = []
    n_files_missing = 0
    total_rows = 0
    state_min = np.full(STATE_DIM, np.inf); state_max = np.full(STATE_DIM, -np.inf)
    act_min = np.full(ACTION_DIM, np.inf); act_max = np.full(ACTION_DIM, -np.inf)
    for ep in episodes:
        i = ep["episode_index"]
        chunk = i // info["chunks_size"]
        pq = root / info["data_path"].format(episode_chunk=chunk, episode_index=i)
        mp4 = root / info["video_path"].format(episode_chunk=chunk, video_key="observation.images.ego_view", episode_index=i)
        if not pq.exists() or not mp4.exists() or mp4.stat().st_size == 0:
            n_files_missing += 1
            continue
        df = pd.read_parquet(pq)
        if len(df) != ep["length"]:
            problems.append(f"episode {i}: parquet rows {len(df)} != episodes.jsonl length {ep['length']}")
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            problems.append(f"episode {i}: missing columns {missing_cols}")
            continue
        st = np.stack(df["observation.state"].to_numpy()); ac = np.stack(df["action"].to_numpy())
        if st.shape[1] != STATE_DIM or ac.shape[1] != ACTION_DIM:
            problems.append(f"episode {i}: state {st.shape} action {ac.shape} (expected 43-D)")
            continue
        if not (np.isfinite(st).all() and np.isfinite(ac).all()):
            problems.append(f"episode {i}: non-finite state/action values")
        if not bool(df["next.done"].iloc[-1]):
            problems.append(f"episode {i}: last row next.done is not True")
        state_min = np.minimum(state_min, st.min(0)); state_max = np.maximum(state_max, st.max(0))
        act_min = np.minimum(act_min, ac.min(0)); act_max = np.maximum(act_max, ac.max(0))
        lengths.append(len(df)); total_rows += len(df)
    if n_files_missing:
        problems.append(f"{n_files_missing} episodes missing parquet/mp4")
    if lengths:
        print(f"episode length: min {min(lengths)} max {max(lengths)} mean {np.mean(lengths):.0f} steps; total rows {total_rows}")
        print(f"state range per group (rad): legs [{state_min[:12].min():.2f},{state_max[:12].max():.2f}] "
              f"waist [{state_min[12:15].min():.2f},{state_max[12:15].max():.2f}] "
              f"left_arm [{state_min[15:22].min():.2f},{state_max[15:22].max():.2f}] "
              f"left_hand [{state_min[22:29].min():.2f},{state_max[22:29].max():.2f}] "
              f"right_arm [{state_min[29:36].min():.2f},{state_max[29:36].max():.2f}] "
              f"right_hand [{state_min[36:43].min():.2f},{state_max[36:43].max():.2f}]")
        moving = np.where((act_max - act_min) > 0.05)[0]
        print(f"action channels that move (> 0.05 rad range): {len(moving)}/43 -> idx {moving.tolist()}")
        if info.get("total_frames") not in (None, total_rows):
            problems.append(f"info.total_frames={info.get('total_frames')} != sum of parquet rows {total_rows}")

    # GR00T loader probe with the same modality config launch_finetune.py registers
    try:
        cfg_path = Path(args.modality_config_path)
        sys.path.append(str(cfg_path.parent))
        importlib.import_module(cfg_path.stem)
        from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
        from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
        from gr00t.data.embodiment_tags import EmbodimentTag

        tag = EmbodimentTag.resolve(args.embodiment_tag)
        modality_configs = MODALITY_CONFIGS[tag.value]
        loader = LeRobotEpisodeLoader(dataset_path=str(root), modality_configs=modality_configs)
        n = min(args.num_probe_episodes, len(loader))
        for k in range(n):
            ep_df = loader[k]
            cols = list(ep_df.columns)
            shapes = {}
            for c in cols:
                v = ep_df[c].iloc[0]
                shapes[c] = getattr(v, "shape", type(v).__name__)
            print(f"GR00T loader episode {k}: {len(ep_df)} rows; columns/shapes: {shapes}")
        print(f"GR00T LeRobotEpisodeLoader OK ({len(loader)} episodes, embodiment_tag={tag.value}, "
              f"action horizon={len(modality_configs['action'].delta_indices)})")
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        problems.append(f"GR00T loader failed: {e}")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1
    print("\nOK: dataset is ready for launch_finetune.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
