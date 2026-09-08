#!/usr/bin/env python3
"""Apply the corrections the URDF converter output needs before it matches the baseline asset.

Plain text editing of the .usda layers, so this needs no Isaac Sim and is reviewable in a diff.
Each patch is idempotent. Rationale for all three is in docs/report.md.

  1. root_joint deactivated   The converter authors a PhysicsFixedJoint welding the pelvis to the
                              world even when invoked with fix_base=False, which pins pelvis z to
                              exactly 0.0 so the G1 never stands. Isaac Lab's `fix_root_link` flag
                              only governs the joint Isaac Lab manages and leaves this one composed
                              in, so it has to be switched off in the asset.

  2. instanceable -> false    USD stage traversal does not descend into instanced prims. The
                              converter marks every link instanceable, so the task's high-friction
                              material never binds to the fingers and a camera cannot be authored
                              under head_link. Symptom in the log:
                              "no G1 hand/finger collision prims were found".

  3. pelvis contour collider  Only if the URDF lacked pelvis_contour_link. The baseline G1 has that
                              collider and the scene has an invisible support slab beneath the
                              table; the baseline robot spawns overlapping it and braces against
                              it, which is why it stands to the millimetre. Without the collider
                              the pelvis creeps ~5.5 cm backwards every episode. The URDF in urdf/
                              already restores the real link, so this patch is a fallback for a
                              conversion from an unmodified URDF.

Usage:
    python3 patch_converted_usd.py /path/to/g1_29dof_with_brainco_hand
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

# AABB of the baseline G1's pelvis_contour_link, in the pelvis frame.
CONTOUR_MIN = (-0.0580, -0.0680, -0.1520)
CONTOUR_MAX = (0.0610, 0.0680, -0.0280)

CONTOUR_BOX = """
            # Stand-in for the baseline G1's pelvis_contour_link collider, same extents.
            def Cube "pelvis_contour_box" (
                prepend apiSchemas = ["PhysicsCollisionAPI"]
            )
            {{
                float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
                uniform token purpose = "guide"
                double size = 1
                quatf xformOp:orient = (1, 0, 0, 0)
                float3 xformOp:scale = ({sx:.4f}, {sy:.4f}, {sz:.4f})
                double3 xformOp:translate = ({tx:.4f}, {ty:.4f}, {tz:.4f})
                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
            }}
"""


def read(path: str) -> str:
    with open(path) as f:
        return f.read()


def write(path: str, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)


def patch_root_joint(root: str) -> str:
    path = os.path.join(root, "payloads", "Physics", "physx.usda")
    if not os.path.exists(path):
        return f"skip: {path} not found"
    text = read(path)

    if re.search(r'over "root_joint"\s*\(\s*active = false', text):
        return "root_joint: already deactivated"

    # The converter writes `over "root_joint" ( prepend apiSchemas = [...] )`; replace the
    # parenthesised metadata with `active = false`, which composes over the joint definition.
    patched, n = re.subn(
        r'(over "root_joint"\s*\()[^)]*(\))',
        r"\1\n            active = false\n        \2",
        text,
        count=1,
    )
    if n == 0:
        # No `over` block exists yet; append one inside the outermost scope.
        idx = text.rstrip().rfind("}")
        if idx < 0:
            return "root_joint: FAILED, unexpected layer structure"
        patched = (
            text[:idx]
            + '\n        over "root_joint"\n        {\n            active = false\n        }\n'
            + text[idx:]
        )
    write(path, patched)
    return "root_joint: deactivated"


def patch_instanceable(root: str) -> str:
    targets = glob.glob(os.path.join(root, "payloads", "*.usda")) + glob.glob(
        os.path.join(root, "*.usda")
    )
    changed = 0
    for path in targets:
        text = read(path)
        patched = re.sub(r"instanceable\s*=\s*true", "instanceable = false", text)
        if patched != text:
            write(path, patched)
            changed += 1
    if changed == 0:
        return "instanceable: already false everywhere"
    return f"instanceable: set false in {changed} layer(s)"


def patch_pelvis_contour(root: str) -> str:
    path = os.path.join(root, "payloads", "base.usda")
    if not os.path.exists(path):
        return f"skip: {path} not found"
    text = read(path)

    if "pelvis_contour_link" in text:
        return "pelvis contour: real link present, no stand-in needed"
    if "pelvis_contour_box" in text:
        return "pelvis contour: stand-in already present"

    match = re.search(r'^(\s*)def Xform "pelvis"[^\n]*\n\1\{', text, flags=re.M)
    if match is None:
        return "pelvis contour: FAILED, could not locate the pelvis Xform"

    scale = tuple(hi - lo for lo, hi in zip(CONTOUR_MIN, CONTOUR_MAX))
    centre = tuple(0.5 * (lo + hi) for lo, hi in zip(CONTOUR_MIN, CONTOUR_MAX))
    box = CONTOUR_BOX.format(
        sx=scale[0], sy=scale[1], sz=scale[2], tx=centre[0], ty=centre[1], tz=centre[2]
    )
    insert = match.end()
    write(path, text[:insert] + box + text[insert:])
    return f"pelvis contour: stand-in box added (scale {scale}, centre {centre})"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("asset_dir", help="directory holding the converted .usda and payloads/")
    args = ap.parse_args()

    root = os.path.abspath(args.asset_dir)
    if not os.path.isdir(root):
        print(f"not a directory: {root}", file=sys.stderr)
        return 1

    failed = False
    for result in (
        patch_root_joint(root),
        patch_instanceable(root),
        patch_pelvis_contour(root),
    ):
        print(f"  {result}")
        failed |= "FAILED" in result

    print("\nThe remaining two corrections live in code, not in the asset:")
    print("  head camera prim path      2_retargeting/g1_brainco/g1_brainco.py, "
          "BRAINCO_HEAD_LINK_PATH")
    print("  PhysX joint velocity caps  2_retargeting/g1_brainco/g1_brainco.py, "
          "velocity_limit_sim")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
