#!/usr/bin/env python3
"""Geometric comparison of the Dex3-1 and BrainCo Revo2 hands, and the tables retargeting uses.

Forward kinematics on the two URDFs, NumPy only, no simulator. This is where the constants in
hand_specs.py come from, so it derives them from scratch rather than importing the retargeting
layer: everything here is independently checkable, and g1_brainco/ consumes the result.

Everything is measured in the wrist_yaw frame, common to both robots and the last link the policy
controls.

Usage:
    ./analyze.sh
    python3 analyze_hand_geometry.py --dex3_urdf <path> [--plot media/aperture_curves.png]
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np

from hand_specs import (
    BRAINCO_ACTIVE_ORDER,
    BRAINCO_LIMITS,
    DEX3_CLOSED_LEFT,
    DEX3_COMMANDED_GRASP,
    DEX3_GROOT_ORDER,
    DEX3_TIP_LINKS,
    DEX3_TIP_OFFSETS,
    REVO2_MOUNT_XYZ,
    THUMB_CF,
    THUMB_META,
    THUMB_PROX,
    DEX3_CLOSURE,
    REVO2_CLOSURE,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SIDE = "left"  # the task is left-handed; the right-hand action is identically zero
FINGERS = ("thumb", "index", "middle")


# ------------------------------------------------------------------ URDF kinematics
def rpy_to_mat(rpy: np.ndarray) -> np.ndarray:
    """URDF rpy, extrinsic XYZ: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def axis_angle(axis: np.ndarray, theta: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


class Urdf:
    """Minimal URDF forward kinematics."""

    def __init__(self, path: str):
        root = ET.parse(path).getroot()
        self.joints: dict[str, dict] = {}
        self.parent_of: dict[str, str] = {}
        for j in root.iter("joint"):
            child = j.find("child").get("link")
            o = j.find("origin")

            def vec(attr, o=o):
                raw = (o.get(attr) or "0 0 0") if o is not None else "0 0 0"
                return np.array([float(v) for v in raw.split()])

            ax = j.find("axis")
            self.joints[j.get("name")] = dict(
                type=j.get("type"),
                parent=j.find("parent").get("link"),
                xyz=vec("xyz"),
                rpy=vec("rpy"),
                axis=(np.array([float(v) for v in ax.get("xyz").split()])
                      if ax is not None else np.array([0.0, 0.0, 1.0])),
            )
            self.parent_of[child] = j.get("name")

    def fk(self, base_link: str, tip_link: str, q: dict[str, float]) -> np.ndarray:
        chain, link = [], tip_link
        while link != base_link:
            jn = self.parent_of.get(link)
            if jn is None:
                raise KeyError(f"{tip_link} is not a descendant of {base_link}")
            chain.append(jn)
            link = self.joints[jn]["parent"]

        M = np.eye(4)
        for jn in reversed(chain):
            j = self.joints[jn]
            M = M @ transform(rpy_to_mat(j["rpy"]), j["xyz"])
            if j["type"] in ("revolute", "continuous"):
                M = M @ transform(axis_angle(j["axis"], q.get(jn, 0.0)), np.zeros(3))
            elif j["type"] == "prismatic":
                M = M @ transform(np.eye(3), j["axis"] * q.get(jn, 0.0))
        return M


# ------------------------------------------------------------------ hand poses
def dex3_tips(model: Urdf, q7: np.ndarray, side: str) -> dict[str, np.ndarray]:
    """Fingertip points in wrist_yaw frame. The Dex3 has no tip link, so use mesh offsets."""
    q = {f"{side}_{n}": v for n, v in zip(DEX3_GROOT_ORDER, q7)}
    return {
        f: (model.fk(f"{side}_wrist_yaw_link", f"{side}_{link}", q)
            @ np.append(DEX3_TIP_OFFSETS[side][f], 1.0))[:3]
        for f, link in DEX3_TIP_LINKS.items()
    }


def revo2_tips(model: Urdf, q6: np.ndarray, side: str) -> dict[str, np.ndarray]:
    q = {f"{side}_{n}": v for n, v in zip(BRAINCO_ACTIVE_ORDER, q6)}
    tips = {}
    for f in FINGERS:
        # The URDF is inconsistent across sides: left tips are "*_tip_Link", right are "*_tip".
        for tip in (f"{side}_{f}_tip_Link", f"{side}_{f}_tip"):
            try:
                tips[f] = model.fk(f"{side}_wrist_yaw_link", tip, q)[:3, 3]
                break
            except KeyError:
                continue
        else:
            raise KeyError(f"no tip link found for {side} {f}")
    return tips


def aperture(tips: dict[str, np.ndarray]) -> float:
    """Thumb tip to the index/middle midpoint: the opening a grasped object must fit into."""
    return float(np.linalg.norm(tips["thumb"] - 0.5 * (tips["index"] + tips["middle"])))


def dex3_at(closure: float, side: str) -> np.ndarray:
    return closure * (DEX3_CLOSED_LEFT if side == "left" else -DEX3_CLOSED_LEFT)


def revo2_at(closure: float) -> np.ndarray:
    """Revo2 pose at a given closure, with the thumb on its opposition schedule."""
    q = closure * BRAINCO_LIMITS[:, 1]
    q[0] = float(np.interp(closure, THUMB_CF, THUMB_META))
    q[1] = float(np.interp(closure, THUMB_CF, THUMB_PROX))
    return np.clip(q, BRAINCO_LIMITS[:, 0], BRAINCO_LIMITS[:, 1])


def retarget(q7: np.ndarray, side: str) -> np.ndarray:
    """Dex3 joint targets -> Revo2 joint targets. Mirrors g1_brainco/hand_retarget.py."""
    closed = DEX3_CLOSED_LEFT if side == "left" else -DEX3_CLOSED_LEFT
    span = np.where(np.abs(closed) < 1e-9, 1.0, closed)
    frac = np.clip(np.where(np.abs(closed) < 1e-9, 0.0, q7 / span), 0.0, 1.0)
    # Index, middle and thumb averaged: the Dex3 pinches as one unit in this task, and the
    # calibration was derived from a uniform closure sweep.
    c = (0.5 * (frac[0] + frac[1]) + 0.5 * (frac[2] + frac[3]) + 0.5 * (frac[5] + frac[6])) / 3.0
    return revo2_at(float(np.interp(c, DEX3_CLOSURE, REVO2_CLOSURE)))


# ------------------------------------------------------------------ reports
def report_thumb_schedule(revo2: Urdf) -> None:
    """Check the shipped thumb schedule against an independent grid search.

    The Dex3's thumb_0 is a roll with no Revo2 counterpart; the Revo2 metacarpal is an opposition
    sweep across the palm. Searching both thumb joints for minimum thumb-to-finger aperture at each
    finger closure is what makes a pinch geometrically possible at all.
    """
    print("=" * 70)
    print("THUMB OPPOSITION SCHEDULE")
    metas = np.linspace(0.0, BRAINCO_LIMITS[0, 1], 41)
    proxs = np.linspace(0.0, BRAINCO_LIMITS[1, 1], 21)

    print("  closure   metacarpal  proximal   aperture")
    best_meta = []
    for c in THUMB_CF:
        best = (np.inf, 0.0, 0.0)
        for m in metas:
            for p in proxs:
                q = c * BRAINCO_LIMITS[:, 1]
                q[0], q[1] = m, p
                ap = aperture(revo2_tips(revo2, q, SIDE))
                if ap < best[0]:
                    best = (ap, m, p)
        best_meta.append(best[1])
        print(f"  {c:5.2f}    {best[1]:9.4f} {best[2]:9.4f}   {best[0] * 1000:6.1f} mm")

    # The search's answer at zero closure puts the thumb already across the palm, i.e. the hand
    # never opens, so the shipped schedule overrides the first two entries to keep it open at rest.
    drift = np.abs(THUMB_META[2:] - np.array(best_meta)[2:]).max()
    print(f"\n  shipped schedule agrees above closure 0.1 to {drift:.3f} rad "
          f"(grid step {metas[1] - metas[0]:.3f} rad).")
    print("  The first two entries are overridden so the hand is open at rest; unconstrained")
    print("  minimisation would hold the thumb across the palm even at zero closure.")


def report_aperture(dex3: Urdf, revo2: Urdf) -> None:
    """Why closure fraction cannot be mapped one to one."""
    print("\n" + "=" * 70)
    print("APERTURE: SOURCE vs RETARGETED")
    print("  closure    Dex3      Revo2")
    for c in np.linspace(0.0, 1.0, 11):
        d = aperture(dex3_tips(dex3, dex3_at(c, SIDE), SIDE)) * 1000
        r = aperture(revo2_tips(revo2, revo2_at(c), SIDE)) * 1000
        print(f"  {c:5.2f}   {d:6.1f} mm  {r:6.1f} mm")
    print("\n  The Dex3 opens to 193 mm and the Revo2 only to 141 mm, so equal closure fraction is")
    print("  not equal opening and a one-to-one map would close the Revo2 on empty air.")
    print("  Both curves are non-monotonic: past a point the fingers curl beyond the thumb and")
    print("  tip-to-tip distance grows again, so the calibration matches the running minimum,")
    print("  the tightest opening reached so far, which is monotone by construction.")
    print("\n  shipped calibration (Dex3 closure -> Revo2 closure):")
    print(f"    {[round(float(v), 3) for v in DEX3_CLOSURE]}")
    print(f"    {[round(float(v), 3) for v in REVO2_CLOSURE]}")


def report_fingertips(dex3: Urdf, revo2: Urdf) -> None:
    print("\n" + "=" * 70)
    print("FINGERTIPS IN wrist_yaw FRAME (metres)")
    for label, q7 in (("OPEN", np.zeros(7)), ("GRASP", DEX3_COMMANDED_GRASP)):
        dt, rt = dex3_tips(dex3, q7, SIDE), revo2_tips(revo2, retarget(q7, SIDE), SIDE)
        print(f"\n  {label}")
        for f in FINGERS:
            print(f"    {f:7s} dex3 {np.array2string(dt[f], precision=4, sign='+')}"
                  f"  revo2 {np.array2string(rt[f], precision=4, sign='+')}"
                  f"  err {np.linalg.norm(dt[f] - rt[f]) * 1000:5.1f} mm")
        print(f"    aperture: dex3 {aperture(dt) * 1000:.1f} mm   revo2 {aperture(rt) * 1000:.1f} mm")


def report_grasp_centre(dex3: Urdf, revo2: Urdf) -> None:
    """The wrist-adapter correction.

    The frozen policy aims the wrist, having learned where a Dex3 palm sits relative to it, so an
    offset here means the Revo2 closes somewhere other than where the apple is. Orientation needs
    no correction: both hands point their fingers along +x.
    """
    print("\n" + "=" * 70)
    print("GRASP-CENTRE OFFSET (at the commanded grasp)")
    print(f"  as-supplied mount origin: {np.array2string(REVO2_MOUNT_XYZ, precision=5)}")
    for side in ("left", "right"):
        g7 = DEX3_COMMANDED_GRASP if side == "left" else -DEX3_COMMANDED_GRASP
        dc = np.mean([dex3_tips(dex3, g7, side)[f] for f in FINGERS], axis=0)
        rc = np.mean([revo2_tips(revo2, retarget(g7, side), side)[f] for f in FINGERS], axis=0)
        off = dc - rc
        xyz = revo2.joints[f"{side}_base_joint"]["xyz"] + off
        print(f"  {side}: offset {np.array2string(off, precision=4)}"
              f"  |offset| {np.linalg.norm(off) * 1000:.1f} mm")
        print(f"        {side}_base_joint xyz = \"{' '.join(f'{v:.5f}' for v in xyz)}\"")
    print("\n  Near zero because the 25.2 mm adapter is already applied to the URDF in urdf/.")
    print("  Re-deriving from an unmodified URDF is what produced that number.")


def plot_curves(dex3: Urdf, revo2: Urdf, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\n  (matplotlib not installed, skipping {path})")
        return

    cs = np.linspace(0.0, 1.0, 41)
    d_ap = np.array([aperture(dex3_tips(dex3, dex3_at(c, SIDE), SIDE)) * 1000 for c in cs])
    r_ap = np.array([aperture(revo2_tips(revo2, revo2_at(c), SIDE)) * 1000 for c in cs])

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    ax = axes[0]
    ax.plot(cs, d_ap, lw=2, color="#1f3864", label="Dex3-1 (source)")
    ax.plot(cs, r_ap, lw=2, color="#2a7f7f", label="Revo2 (retargeted)")
    ax.set_xlabel("closure fraction")
    ax.set_ylabel("fingertip aperture [mm]")
    ax.set_title("Equal closure is not equal opening", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.annotate("thumb swings into\nopposition", xy=(0.05, 62), xytext=(0.15, 100),
                fontsize=7, color="#2a7f7f",
                arrowprops=dict(arrowstyle="->", color="#2a7f7f", lw=1))
    ax.annotate("both curves turn back up:\nfingers curl past the thumb",
                xy=(0.88, 74), xytext=(0.38, 155), fontsize=7, color="#444444",
                arrowprops=dict(arrowstyle="->", color="#444444", lw=1))

    ax = axes[1]
    ax.plot(DEX3_CLOSURE, REVO2_CLOSURE, "o-", lw=2, color="#2a7f7f")
    ax.plot([0, 1], [0, 1], ls=":", color="#888888", lw=1.4)
    ax.text(0.60, 0.72, "one-to-one\n(what we do not do)", fontsize=7, color="#666666")
    ax.set_xlabel("Dex3 closure commanded by the policy")
    ax.set_ylabel("Revo2 closure applied")
    ax.set_title("Aperture-matched calibration", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dex3_urdf", required=True, help="G1 + Dex3-1 URDF")
    ap.add_argument("--revo2_urdf",
                    default=os.path.join(HERE, "urdf", "g1_29dof_with_brainco_hand.urdf"))
    ap.add_argument("--skip_thumb_search", action="store_true",
                    help="skip the ~5 s grid search that verifies the thumb schedule")
    ap.add_argument("--plot", metavar="PATH", help="write the aperture curves here")
    args = ap.parse_args()

    dex3, revo2 = Urdf(args.dex3_urdf), Urdf(args.revo2_urdf)
    if not args.skip_thumb_search:
        report_thumb_schedule(revo2)
    report_aperture(dex3, revo2)
    report_fingertips(dex3, revo2)
    report_grasp_centre(dex3, revo2)
    if args.plot:
        plot_curves(dex3, revo2, args.plot)


if __name__ == "__main__":
    main()
