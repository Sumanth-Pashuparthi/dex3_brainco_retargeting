"""Sanity-check a recorded HDF5 before feeding it to the G1 apple Mimic pipeline.

No simulator needed (h5py only). Checks that the file is what ``annotate_demos.py`` /
``generate_dataset.py`` on ``galileo_g1_static_apple_mimic`` expect:

* Arena ``record_demos.py`` layout: ``/data/demo_*`` with ``actions``, ``initial_state``, ``states``
* 23-D Pink IK actions (``g1_wbc_agile_pink`` / ``g1_wbc_pink``), NOT the 50-D joint embodiment
* robot + apple + plate present in ``initial_state``
* per-demo ``success`` attr, and (if present) which arm's hand-state channel closes -> mimic_arm hint

Exit code 0 = OK, 1 = problems found.

    python g1_apple_mimic/validate_source_demos.py /datasets/g1_apple_mimic/source_demos.hdf5
"""

from __future__ import annotations

import argparse
import json
import sys

import h5py
import numpy as np

EXPECTED_ACTION_DIM = 23
LEFT_HAND_IDX, RIGHT_HAND_IDX = 0, 1
APPLE = "apple_01_objaverse_robolab"
PLATE = "clay_plates_hot3d_robolab"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("hdf5")
    ap.add_argument("--object", default=APPLE)
    ap.add_argument("--destination", default=PLATE)
    args = ap.parse_args()

    problems: list[str] = []
    with h5py.File(args.hdf5, "r") as f:
        if "data" not in f:
            print("ERROR: no /data group; not an Arena record_demos file")
            return 1
        data = f["data"]
        env_args = {}
        if "env_args" in data.attrs:
            try:
                env_args = json.loads(data.attrs["env_args"])
            except Exception:  # noqa: BLE001
                pass
        demos = sorted([k for k in data.keys() if k.startswith("demo_")], key=lambda s: int(s.split("_")[1]))
        print(f"file: {args.hdf5}")
        print(f"format_version: {f.attrs.get('format_version', 'missing (legacy wxyz!)')}")
        print(f"env_name: {env_args.get('env_name', '')!r}  demos: {len(demos)}")
        if "format_version" not in f.attrs:
            problems.append("missing root format_version attr (legacy WXYZ quaternions)")
        if len(demos) == 0:
            problems.append("no demo_* groups")

        left_close = right_close = 0
        lengths = []
        for name in demos:
            d = data[name]
            if "actions" not in d:
                problems.append(f"{name}: no actions")
                continue
            a = d["actions"]
            lengths.append(a.shape[0])
            if a.shape[1] != EXPECTED_ACTION_DIM:
                problems.append(
                    f"{name}: actions dim {a.shape[1]} != {EXPECTED_ACTION_DIM} "
                    "(must be recorded with the Pink IK embodiment g1_wbc_agile_pink)"
                )
                continue
            arr = a[()]
            # hand-state channels: 0 open, 1 closed (Pink retargeter); detect which arm is used
            if np.any(arr[:, LEFT_HAND_IDX] > 0.5):
                left_close += 1
            if np.any(arr[:, RIGHT_HAND_IDX] > 0.5):
                right_close += 1
            if "success" in d.attrs and not bool(d.attrs["success"]):
                problems.append(f"{name}: success=False")
            init = d.get("initial_state")
            if init is None:
                problems.append(f"{name}: no initial_state")
            else:
                rigid = init.get("rigid_object", {})
                for obj in (args.object, args.destination):
                    if obj not in rigid:
                        problems.append(f"{name}: initial_state has no rigid_object/{obj}")
                if "articulation" not in init or "robot" not in init["articulation"]:
                    problems.append(f"{name}: initial_state has no articulation/robot")
            if "obs" in d and "datagen_info" in d["obs"]:
                print(f"note: {name} already contains obs/datagen_info (annotated file?)")

        if lengths:
            print(f"episode length: min {min(lengths)}  max {max(lengths)}  mean {np.mean(lengths):.0f} steps")
        print(f"hand closes: left in {left_close}/{len(demos)} demos, right in {right_close}/{len(demos)} demos")
        if left_close and not right_close:
            print("=> use --mimic_arm left")
        elif right_close and not left_close:
            print("=> use --mimic_arm right")
        elif left_close and right_close:
            print("WARNING: both hands close in the dataset; Mimic graph supports one active arm.")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1
    print("\nOK: source demos look compatible with galileo_g1_static_apple_mimic")
    return 0


if __name__ == "__main__":
    sys.exit(main())
