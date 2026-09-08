#!/usr/bin/env python3
"""Report what is actually inside a dataset, for HDF5 shards or a converted LeRobot tree.

Used at every stage of the pipeline, so the same numbers are quoted in the README, printed after a
merge, and printed by status.sh while generation runs.

    python3 dataset_stats.py <file.hdf5> [more.hdf5 ...]
    python3 dataset_stats.py --brief <dir>/*.hdf5
    python3 dataset_stats.py <lerobot_dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

CONTROL_HZ = 50.0


def _fmt_mb(path: str) -> str:
    return f"{os.path.getsize(path) / 1e6:.1f} MB"


def hdf5_stats(path: str) -> dict:
    import h5py

    with h5py.File(path, "r") as f:
        if "data" not in f:
            return {"path": path, "error": "no /data group"}
        d = f["data"]
        names = sorted((k for k in d if k.startswith("demo_")), key=lambda s: int(s.split("_")[1]))
        lengths, successes = [], 0
        for n in names:
            g = d[n]
            lengths.append(int(g.attrs.get("num_samples", len(g["actions"]) if "actions" in g else 0)))
            successes += int(bool(g.attrs.get("success", False)))

        out = {
            "path": path,
            "size": _fmt_mb(path),
            "demos": len(names),
            "successes": successes,
            "steps": int(sum(lengths)),
        }
        if lengths:
            out.update(
                min_len=int(min(lengths)),
                max_len=int(max(lengths)),
                mean_len=round(float(np.mean(lengths)), 1),
                seconds=round(sum(lengths) / CONTROL_HZ, 1),
            )
        if names:
            g = d[names[0]]
            if "actions" in g:
                out["action_dim"] = int(g["actions"].shape[1])
            out["groups"] = sorted(g.keys())
            if "camera_obs" in g:
                out["cameras"] = {k: list(g["camera_obs"][k].shape[1:]) for k in g["camera_obs"]}
            sig = g.get("obs/datagen_info/subtask_term_signals")
            if sig is not None:
                out["subtask_signals"] = sorted(sig.keys())
        if "env_args" in d.attrs:
            try:
                ea = json.loads(d.attrs["env_args"])
                out["env_name"] = ea.get("env_name")
                if "target_embodiment" in ea:
                    out["embodiment"] = ea["target_embodiment"]
            except (ValueError, TypeError):
                pass
    return out


def lerobot_stats(path: str) -> dict:
    info = json.load(open(os.path.join(path, "meta", "info.json")))
    out = {
        "path": path,
        "episodes": info.get("total_episodes"),
        "frames": info.get("total_frames"),
        "fps": info.get("fps"),
        "robot_type": info.get("robot_type"),
    }
    if out["frames"] and out["fps"]:
        out["seconds"] = round(out["frames"] / out["fps"], 1)
    feats = info.get("features", {})
    out["features"] = {k: v.get("shape") for k, v in feats.items()}
    for extra in ("modality.json", "stats.json"):
        out[extra] = os.path.exists(os.path.join(path, "meta", extra))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--brief", action="store_true", help="one line per file")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    results = []
    for p in args.paths:
        try:
            results.append(lerobot_stats(p) if os.path.isdir(p) else hdf5_stats(p))
        except Exception as e:  # a shard being written to is expected to fail here
            results.append({"path": p, "error": f"{type(e).__name__}: {e}"})

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    total = 0
    for r in results:
        name = os.path.basename(r["path"].rstrip("/"))
        if "error" in r:
            print(f"{name:44s} {r['error']}")
            continue
        if args.brief:
            n = r.get("demos", r.get("episodes", 0))
            total += n or 0
            print(f"{name:44s} demos={n:<5} {r.get('size', '')}")
            continue
        print(f"\n{name}")
        for k, v in r.items():
            if k == "path":
                continue
            print(f"  {k:18s} {v}")
    if args.brief and len(results) > 1:
        print(f"{'TOTAL':44s} demos={total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
