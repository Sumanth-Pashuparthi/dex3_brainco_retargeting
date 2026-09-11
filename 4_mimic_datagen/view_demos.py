#!/usr/bin/env python3
"""Render the head-camera stream of generated demos to an MP4 grid and a contact sheet.

Works while generation is running: the shard is copied first (the workers hold it open for writing
and flush after each episode, so a copy is a consistent snapshot of the finished demos), then read
from the copy. Host only: h5py, numpy, opencv.

    python3 view_demos.py                                  # newest gen_w*.hdf5, last 6 demos
    python3 view_demos.py gen_w3.hdf5 --demos 0 5 17       # specific demos
    python3 view_demos.py --all-shards --n 9 --cols 3      # newest demo from each shard, 3 x 3

Each tile is stamped with shard / demo, step and time, and the apple's spawn XY so the
randomisation is visible. Output defaults to /tmp/g1_apple_mimic_demos.{mp4,png}.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import tempfile

import cv2
import h5py
import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

CAM = "camera_obs/robot_head_cam_rgb"
APPLE = "states/rigid_object/apple_01_objaverse_robolab/root_pose"
HZ = 50


def snapshot(path: str, tmpdir: str) -> str:
    dst = os.path.join(tmpdir, os.path.basename(path))
    shutil.copyfile(path, dst)
    return dst


def load(path: str, want: list[int] | None, n: int) -> list[dict]:
    out = []
    with h5py.File(path, "r") as f:
        keys = sorted(f["data"], key=lambda k: int(k.split("_")[1]))
        if want is not None:
            keys = [f"demo_{i}" for i in want if f"demo_{i}" in f["data"]]
        else:
            keys = keys[-n:]
        for k in keys:
            g = f["data"][k]
            if CAM not in g:
                continue
            out.append({
                "name": f"{os.path.basename(path).replace('.hdf5', '')}/{k}",
                "frames": g[CAM][:],
                "apple": g[APPLE][0, :2] if APPLE in g else None,
            })
    return out


def stamp(im: np.ndarray, lines: list[str]) -> np.ndarray:
    for i, s in enumerate(lines):
        cv2.putText(im, s, (6, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(im, s, (6, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return im


def tile(demo: dict, t: int, size: tuple[int, int]) -> np.ndarray:
    T = len(demo["frames"])
    i = min(t, T - 1)
    im = cv2.resize(cv2.cvtColor(demo["frames"][i], cv2.COLOR_RGB2BGR), size)
    lines = [demo["name"], f"step {i}/{T - 1}  {i / HZ:4.1f}s" + ("  (end)" if t >= T - 1 else "")]
    if demo["apple"] is not None:
        lines.append(f"apple xy {demo['apple'][0]:+.3f} {demo['apple'][1]:+.3f}")
    return stamp(im, lines)


def grid(tiles: list[np.ndarray], cols: int) -> np.ndarray:
    h, w = tiles[0].shape[:2]
    rows = -(-len(tiles) // cols)
    canvas = np.zeros((rows * h, cols * w, 3), np.uint8)
    for k, im in enumerate(tiles):
        r, c = divmod(k, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hdf5", nargs="?", help="shard or merged file (default: newest gen_w*.hdf5 in --data_dir)")
    ap.add_argument("--data_dir", default=os.path.expanduser("~/g1_baseline/datasets/g1_apple_mimic"))
    ap.add_argument("--all-shards", action="store_true", help="take demos from every gen_w*.hdf5, newest first")
    ap.add_argument("--demos", type=int, nargs="*", help="explicit demo indices (single file only)")
    ap.add_argument("--n", type=int, default=6, help="how many demos when --demos is not given (latest)")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--tile", type=int, nargs=2, default=(480, 360), metavar=("W", "H"))
    ap.add_argument("--fps", type=float, default=HZ, help="playback rate; 50 = real time")
    ap.add_argument("--stride", type=int, default=1, help="use every k-th frame")
    ap.add_argument("-o", "--out", default="/tmp/g1_apple_mimic_demos", help="output stem (.mp4 and .png)")
    a = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(a.data_dir, "gen_w*.hdf5")), key=os.path.getmtime, reverse=True)
    if a.hdf5:
        files = [a.hdf5]
    elif a.all_shards:
        files = shards
    elif shards:
        files = shards[:1]
    else:
        raise SystemExit(f"no gen_w*.hdf5 in {a.data_dir}")

    demos: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        for p in files:
            if os.path.getsize(p) < 10_000:
                continue  # worker created the file but has not written a demo yet
            try:
                per = a.n if not a.all_shards else max(1, -(-a.n // len(files)))
                demos += load(snapshot(p, tmp), a.demos, per)
            except OSError as e:
                print(f"skip {p}: {e}")
            if len(demos) >= a.n and a.demos is None:
                demos = demos[: a.n]
                break
    if not demos:
        raise SystemExit("no finished demos yet")

    size = tuple(a.tile)
    T = max(len(d["frames"]) for d in demos)
    cols = min(a.cols, len(demos))
    vw = cv2.VideoWriter(a.out + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), a.fps / a.stride,
                         (cols * size[0], (-(-len(demos) // cols)) * size[1]))
    for t in range(0, T + HZ, a.stride):  # hold the last frame for 1 s
        vw.write(grid([tile(d, t, size) for d in demos], cols))
    vw.release()

    # contact sheet: first / grasp-ish (40 %) / last frame of each demo
    rows = []
    for d in demos:
        n = len(d["frames"])
        rows.append(np.hstack([tile(d, i, size) for i in (0, int(0.4 * n), n - 1)]))
    cv2.imwrite(a.out + ".png", np.vstack(rows))

    for d in demos:
        print(f"{d['name']:<24} {len(d['frames']):4d} steps {len(d['frames']) / HZ:5.1f}s"
              + (f"  apple xy {d['apple'].round(3)}" if d["apple"] is not None else ""))
    print(f"wrote {a.out}.mp4 ({T / a.fps * a.stride:.0f}s) and {a.out}.png")


if __name__ == "__main__":
    main()
