#!/usr/bin/env python3
"""Plot one rollout beside the Mimic source demo built from it.

Shows what prepare_source.sh actually does to a trajectory: the wrist height the joint policy
produced, the binary hand state recovered from its Dex3 finger commands, and the scripted place and
settle padding that replace the policy's own release.

    python3 make_figure.py <rollouts.hdf5> <source_demos.hdf5> -o media/source_conversion.png
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

CONTROL_HZ = 50.0
DEX3_LEFT_HAND_IDX = [29, 30, 31, 35, 36, 37, 41]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rollouts")
    ap.add_argument("source")
    ap.add_argument("-o", "--out", default="media/source_conversion.png")
    ap.add_argument("--demo", type=int, default=0)
    ap.add_argument("--pad_steps", type=int, default=40, help="must match prepare_source.sh")
    args = ap.parse_args()

    # The converter drops demos it cannot use, so demo_i in the source is not demo_i in the
    # rollouts. It records where each one came from.
    with h5py.File(args.source, "r") as f:
        sg = f["data"][f"demo_{args.demo}"]
        a = sg["actions"][()]
        origin = sg.attrs.get("source_demo", f"demo_{args.demo}")
        origin = origin.decode() if isinstance(origin, bytes) else str(origin)
    src_z, src_hand = a[:, 4], a[:, 0]

    with h5py.File(args.rollouts, "r") as f:
        g = f["data"][origin]
        wrist_z = g["obs/left_wrist_pose_pelvis_frame"][()][:, 2, 3]
        dex3 = g["processed_actions_dex3"][()]
        closure = np.abs(dex3[:, DEX3_LEFT_HAND_IDX]).mean(axis=1)

    t_in = np.arange(len(wrist_z)) / CONTROL_HZ
    t_out = np.arange(len(a)) / CONTROL_HZ

    # Where the source stops following the rollout: the scripted place takes over shortly after the
    # grasp. Found by comparison rather than assumed, so the shading stays right if the knobs change.
    n = min(len(wrist_z), len(a))
    grasp = int(np.argmax(src_hand > 0.5))
    diverged = np.nonzero(np.abs(src_z[:n] - wrist_z[:n]) > 0.01)[0]
    diverged = diverged[diverged > grasp]
    cut = (diverged[0] if len(diverged) else n) / CONTROL_HZ
    # The settle padding is the trailing hold-pose steps appended after the release.
    pad_start = (len(a) - int(args.pad_steps)) / CONTROL_HZ

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1]})

    ax0.plot(t_in, wrist_z, color="#2a6f97", lw=1.8, label="rollout: measured left-wrist z")
    ax0.plot(t_out, src_z, color="#c1440e", lw=1.8, label="source demo: commanded left-wrist z")
    ax0.axvspan(cut, pad_start, color="#f2e3d5", zorder=0)
    ax0.axvspan(pad_start, t_out[-1], color="0.91", zorder=0)
    ax0.set_ylabel("wrist height above pelvis (m)")
    ax0.set_title("Rollout (50-D joint actions) converted to a Mimic source demo (23-D Pink actions)")
    ax0.legend(fontsize=8, loc="upper left")
    ax0.grid(alpha=0.3)

    ax1.plot(t_in, closure, color="#2a6f97", lw=1.4, label="rollout: mean |Dex3 finger command| (rad)")
    ax1.plot(t_out, src_hand, color="#c1440e", lw=1.8, label="source demo: binary hand state")
    ax1.axhline(0.15, color="0.4", ls=":", lw=1.2)
    ax1.text(0.15, 0.17, "close threshold 0.15 rad", fontsize=7.5, color="0.3")
    ax1.axvspan(cut, pad_start, color="#f2e3d5", zorder=0)
    ax1.axvspan(pad_start, t_out[-1], color="0.91", zorder=0)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("hand")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.3)

    top = ax0.get_ylim()[1]
    ax0.text((cut + pad_start) / 2, top, "scripted place", ha="center", va="top", fontsize=8, color="0.35")
    ax0.text((pad_start + t_out[-1]) / 2, top, "settle\npadding", ha="center", va="top", fontsize=8, color="0.35")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
