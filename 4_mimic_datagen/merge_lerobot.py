"""Combine several GR00T-LeRobot datasets into one, so a targeted round can be trained with round 2.

GR00T post-training takes a single ``dataset_path``, so a new generation round cannot simply be
pointed at alongside the old one. Merging the HDF5s instead and re-converting would work, but the
round-2 HDF5 is 210 GB and this box has ~56 GB free, whereas the LeRobot form of the same data is
1.8 GB. So the merge happens here, at the LeRobot level.

Episodes from the first input keep their indices; later inputs are shifted after it. Videos are
hardlinked when the output is on the same filesystem, so the merge costs almost no space. Parquet
files must be rewritten for shifted episodes because ``episode_index``, ``index`` and
``task_index`` are absolute.

``meta/stats.json`` is deliberately not merged: normalisation statistics have to be recomputed
over the combined data or training normalises against the wrong distribution. Re-run

    python gr00t/data/stats.py --dataset-path <out> --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path <cfg>

afterwards -- run_merge_lerobot.sh does this for you.

  python3 merge_lerobot.py -o <out>/lerobot <round2>/lerobot <round3>/lerobot
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import pyarrow as pa
import pyarrow.parquet as pq


def read_meta(root: str) -> dict:
    with open(os.path.join(root, "meta", "info.json")) as f:
        info = json.load(f)
    episodes = [json.loads(line) for line in open(os.path.join(root, "meta", "episodes.jsonl"))]
    tasks = [json.loads(line) for line in open(os.path.join(root, "meta", "tasks.jsonl"))]
    return {"info": info, "episodes": episodes, "tasks": tasks}


def link_or_copy(src: str, dst: str, allow_link: bool) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    if allow_link:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", help="LeRobot dataset roots, in the order to concatenate")
    p.add_argument("-o", "--out", required=True, help="output LeRobot root")
    p.add_argument("--copy", action="store_true", help="copy videos instead of hardlinking them")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    metas = [read_meta(r) for r in args.inputs]
    chunk_size = metas[0]["info"]["chunks_size"]
    video_keys = [k for k, v in metas[0]["info"]["features"].items() if v.get("dtype") == "video"]

    for root, meta in zip(args.inputs[1:], metas[1:]):
        if meta["info"]["fps"] != metas[0]["info"]["fps"]:
            raise SystemExit(f"fps mismatch: {root}")
        if set(meta["info"]["features"]) != set(metas[0]["info"]["features"]):
            raise SystemExit(f"feature set mismatch: {root}")

    # One task table for the merged set, keyed by task string so duplicates collapse.
    task_index: dict[str, int] = {}
    for meta in metas:
        for t in meta["tasks"]:
            task_index.setdefault(t["task"], len(task_index))

    total_ep = sum(len(m["episodes"]) for m in metas)
    total_frames = sum(sum(e["length"] for e in m["episodes"]) for m in metas)
    print(f"{len(args.inputs)} inputs -> {total_ep} episodes, {total_frames} frames")
    for root, meta in zip(args.inputs, metas):
        n = len(meta["episodes"])
        print(f"  {n:5d} episodes  {sum(e['length'] for e in meta['episodes']):8d} frames  {root}")
    if args.dry_run:
        print("\ndry run, nothing written")
        return

    os.makedirs(os.path.join(args.out, "meta"), exist_ok=True)
    allow_link = not args.copy

    out_episodes = []
    ep_offset = 0
    # Frames contributed by earlier inputs. Each source's `index` column is already a global
    # counter over its own episodes, so it shifts by this constant -- not by a running per-episode
    # total, which would count every preceding episode twice.
    frame_offset = 0
    for root, meta in zip(args.inputs, metas):
        local_tasks = {t["task_index"]: t["task"] for t in meta["tasks"]}
        passthrough = ep_offset == 0 and frame_offset == 0 and not _needs_task_remap(local_tasks, task_index)
        for ep in meta["episodes"]:
            src_idx = ep["episode_index"]
            dst_idx = src_idx + ep_offset
            src_pq = os.path.join(
                root, "data", f"chunk-{src_idx // chunk_size:03d}", f"episode_{src_idx:06d}.parquet"
            )
            dst_pq = os.path.join(
                args.out, "data", f"chunk-{dst_idx // chunk_size:03d}", f"episode_{dst_idx:06d}.parquet"
            )

            if passthrough:
                link_or_copy(src_pq, dst_pq, allow_link)
            else:
                table = pq.read_table(src_pq)
                cols = {name: table.column(name) for name in table.column_names}
                n = table.num_rows
                cols["episode_index"] = pa.array([dst_idx] * n, type=table.schema.field("episode_index").type)
                base = table.column("index").to_pylist()
                cols["index"] = pa.array(
                    [i + frame_offset for i in base], type=table.schema.field("index").type
                )
                if "task_index" in cols:
                    remap = [
                        task_index[local_tasks[t]] for t in table.column("task_index").to_pylist()
                    ]
                    cols["task_index"] = pa.array(remap, type=table.schema.field("task_index").type)
                os.makedirs(os.path.dirname(dst_pq), exist_ok=True)
                pq.write_table(pa.table(cols, schema=table.schema), dst_pq)

            for key in video_keys:
                src_v = os.path.join(
                    root, "videos", f"chunk-{src_idx // chunk_size:03d}", key, f"episode_{src_idx:06d}.mp4"
                )
                if not os.path.exists(src_v):
                    raise SystemExit(f"missing video: {src_v}")
                dst_v = os.path.join(
                    args.out, "videos", f"chunk-{dst_idx // chunk_size:03d}", key, f"episode_{dst_idx:06d}.mp4"
                )
                link_or_copy(src_v, dst_v, allow_link)

            out_episodes.append(
                {"episode_index": dst_idx, "tasks": ep["tasks"], "length": ep["length"]}
            )
        ep_offset += len(meta["episodes"])
        frame_offset += sum(e["length"] for e in meta["episodes"])
        print(f"  merged {root} -> episodes up to {ep_offset - 1}, frames up to {frame_offset - 1}")

    info = dict(metas[0]["info"])
    info["total_episodes"] = len(out_episodes)
    info["total_frames"] = sum(e["length"] for e in out_episodes)
    info["total_tasks"] = len(task_index)
    info["total_videos"] = len(out_episodes) * max(1, len(video_keys))
    info["total_chunks"] = (len(out_episodes) + chunk_size - 1) // chunk_size
    with open(os.path.join(args.out, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=4)
    with open(os.path.join(args.out, "meta", "episodes.jsonl"), "w") as f:
        for ep in out_episodes:
            f.write(json.dumps(ep) + "\n")
    with open(os.path.join(args.out, "meta", "tasks.jsonl"), "w") as f:
        for task, idx in sorted(task_index.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")
    shutil.copy2(
        os.path.join(args.inputs[0], "meta", "modality.json"),
        os.path.join(args.out, "meta", "modality.json"),
    )

    print(f"\nwrote {args.out}: {info['total_episodes']} episodes, {info['total_frames']} frames")
    print("meta/stats.json was NOT written -- recompute it with gr00t/data/stats.py before training")


def _needs_task_remap(local_tasks: dict[int, str], task_index: dict[str, int]) -> bool:
    return any(task_index[name] != idx for idx, name in local_tasks.items())


if __name__ == "__main__":
    main()
