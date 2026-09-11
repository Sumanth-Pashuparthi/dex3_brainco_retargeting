#!/usr/bin/env python3
"""Count hand-close events per episode in a GR00T-LeRobot dataset.

This is the measurement that settled why the fine-tuned policy never retries a failed grasp. The
hypothesis under test was that retry behaviour was being destroyed at inference time -- by the
policy executing a whole 0.8 s action chunk before looking at the world again. Two inference-side
fixes were built and neither produced a retry, so the question became whether the behaviour was in
the training data to begin with.

It is not. A retry has to show up as the hand opening and closing more than once in an episode,
and in the merged dataset every single episode closes it exactly once. That is a property of how
the data was made rather than an accident: the demonstrations are success-filtered Mimic replays,
so a fumble-then-recover episode could not have entered the set.

Usage:
    python3 audit_regrasp.py /path/to/lerobot            # or a HF dataset id via --repo
    python3 audit_regrasp.py $DATA_DIR/g1_apple_mimic_r2r3/lerobot --json out.json

Requires pandas and pyarrow; no GPU and no Isaac Sim.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

# Index of the left-hand grasp command inside the 43-D action vector. From meta/modality.json:
# left_hand spans 22-29, and within it this is the one dimension the recorded demos actually
# drive -- the others stay at their neutral value, so averaging the block washes the signal out.
# An earlier version of this audit thresholded the block mean and reported zero closes in every
# episode, which was wrong in the opposite direction.
GRASP_DIM = 23

# The command is binary in this data: exactly 0.0 (open) or -0.87 rad (closed). The midpoint is a
# safe threshold and still works if a future dataset drives it continuously.
CLOSED_BELOW = -0.4


def episode_close_events(action: np.ndarray) -> int:
    """Number of times the hand transitions from open to closed in one episode."""
    closed = action[:, GRASP_DIM] < CLOSED_BELOW
    # Count rising edges rather than closed frames: a grasp held for 200 frames is one event.
    return int(np.sum(closed[1:] & ~closed[:-1]) + (1 if closed[0] else 0))


def audit(root: str) -> dict:
    files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"no parquet files under {root}/data -- is this a LeRobot tree?")

    hist: Counter[int] = Counter()
    offenders: list[dict] = []
    dim_range = [np.inf, -np.inf]

    for path in files:
        df = pd.read_parquet(path, columns=["action", "episode_index"])
        action = np.stack(df["action"].to_numpy())
        dim_range[0] = min(dim_range[0], float(action[:, GRASP_DIM].min()))
        dim_range[1] = max(dim_range[1], float(action[:, GRASP_DIM].max()))

        n = episode_close_events(action)
        hist[n] += 1
        if n != 1:
            offenders.append({
                "episode": int(df["episode_index"].iloc[0]),
                "close_events": n,
                "file": os.path.relpath(path, root),
            })

    total = sum(hist.values())
    regrasp = sum(c for n, c in hist.items() if n >= 2)
    return {
        "dataset": root,
        "episodes_scanned": total,
        "grasp_dim": GRASP_DIM,
        "grasp_dim_range": dim_range,
        "close_events_histogram": {str(k): hist[k] for k in sorted(hist)},
        "episodes_with_regrasp": regrasp,
        "episodes_never_closing": hist.get(0, 0),
        "examples": offenders[:20],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="path to a LeRobot tree (the directory holding data/ and meta/)")
    ap.add_argument("--json", help="also write the result here")
    args = ap.parse_args()

    res = audit(args.dataset)

    print(f"scanned {res['episodes_scanned']} episodes in {res['dataset']}")
    print(f"action dim {res['grasp_dim']} ranges "
          f"[{res['grasp_dim_range'][0]:.3f}, {res['grasp_dim_range'][1]:.3f}]")
    print("\nhand-close events per episode:")
    for k, v in res["close_events_histogram"].items():
        print(f"  {k} close(s): {v:>6} episodes")
    print(f"\nepisodes containing a re-grasp (>= 2 closes): {res['episodes_with_regrasp']}")
    print(f"episodes that never close the hand:           {res['episodes_never_closing']}")
    if res["episodes_with_regrasp"] == 0:
        print("\n=> no recovery behaviour exists in this dataset, so none can be cloned from it.")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
