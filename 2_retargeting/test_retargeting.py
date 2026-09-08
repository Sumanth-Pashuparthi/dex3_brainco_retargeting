#!/usr/bin/env python3
"""Checks on the retargeting layer that need no simulator, only NumPy.

Run from this directory:  python3 test_retargeting.py
"""

from __future__ import annotations

import sys

import numpy as np

import hand_specs
from g1_brainco import hand_specs as installed_specs
from g1_brainco.hand_retarget import (
    BRAINCO_ACTIVE_ORDER,
    BRAINCO_LIMITS,
    DEX3_GROOT_ORDER,
    grasp_closure_batch,
    inverse_batch,
    retarget_batch,
)

# The grasp the frozen policy actually commands, read off a trace, in DEX3_GROOT_ORDER.
DEX3_GRASP = np.array([-0.598, -1.200, -0.600, -1.200, 0.000, 0.700, 0.700])

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        _failures.append(name)


def dex3_at_closure(c: float, side: str) -> np.ndarray:
    closed = hand_specs.DEX3_CLOSED_LEFT if side == "left" else -hand_specs.DEX3_CLOSED_LEFT
    return c * closed


def main() -> int:
    print("installed tables match the derived ones")
    # g1_brainco/hand_specs.py is a copy, so it can drift from the one the analysis writes.
    for name in ("THUMB_CF", "THUMB_META", "THUMB_PROX", "DEX3_CLOSURE", "REVO2_CLOSURE"):
        a = getattr(hand_specs, name)
        b = getattr(installed_specs, name)
        check(f"{name} in sync", bool(np.allclose(a, b)))

    print("\nshapes and ordering")
    check("7 Dex3 joints", len(DEX3_GROOT_ORDER) == 7)
    check("6 Revo2 joints", len(BRAINCO_ACTIVE_ORDER) == 6)
    for side in ("left", "right"):
        q = retarget_batch(np.zeros((4, 7)), side)
        check(f"{side}: (N,7) -> (N,6)", q.shape == (4, 6))

    print("\nlimits")
    for side in ("left", "right"):
        cs = np.linspace(0.0, 1.0, 21)
        q = retarget_batch(np.stack([dex3_at_closure(c, side) for c in cs]), side)
        lo_ok = bool((q >= BRAINCO_LIMITS[:, 0] - 1e-9).all())
        hi_ok = bool((q <= BRAINCO_LIMITS[:, 1] + 1e-9).all())
        check(f"{side}: every target within joint limits", lo_ok and hi_ok)

    print("\nopen hand maps to open hand")
    for side in ("left", "right"):
        q = retarget_batch(np.zeros((1, 7)), side)[0]
        # The metacarpal rests at 0.35 rad by design so the palm presents a wide aperture; the
        # flexors must be at zero.
        check(f"{side}: flexors open", bool(np.allclose(q[1:], 0.0, atol=1e-9)),
              f"thumb_meta={q[0]:.3f}")

    print("\nmonotonicity: more Dex3 closure never means less Revo2 closure")
    for side in ("left", "right"):
        cs = np.linspace(0.0, 1.0, 51)
        q = retarget_batch(np.stack([dex3_at_closure(c, side) for c in cs]), side)
        flex = q[:, 2]  # index proximal drives the grasp
        check(f"{side}: index flexion non-decreasing",
              bool((np.diff(flex) >= -1e-9).all()))

    print("\nthumb leads the fingers into opposition")
    for side in ("left", "right"):
        early = retarget_batch(dex3_at_closure(0.10, side)[None, :], side)[0]
        # At 10% source closure the thumb should already be swung across the palm while the
        # fingers have barely moved: this is what makes a pinch geometrically possible.
        check(f"{side}: thumb ahead of fingers at 10% closure",
              early[0] > 0.9 and early[2] < 0.2 * BRAINCO_LIMITS[2, 1],
              f"meta={early[0]:.3f} index={early[2]:.3f}")

    print("\nround trip: command -> Revo2 -> pseudo-Dex3 state")
    for side in ("left", "right"):
        errs = []
        for c in np.linspace(0.0, 0.5, 11):
            cmd = dex3_at_closure(c, side)
            q6 = retarget_batch(cmd[None, :], side)
            back = inverse_batch(q6, side)
            errs.append(abs(grasp_closure_batch(back, side)[0] - c))
        worst = max(errs)
        # A dead zone here means the policy commands closure, nothing moves, and its next
        # observation reports an open hand. On the closing branch the map must be invertible.
        check(f"{side}: closing branch round trip exact", worst < 1e-6, f"max err {worst:.2e}")

    print("\nring and pinky follow the driven fingers (no source signal)")
    for side in ("left", "right"):
        q = retarget_batch(DEX3_GRASP[None, :] * (1 if side == "left" else -1), side)[0]
        check(f"{side}: ring/pinky track index/middle",
              bool(np.allclose(q[4:], q[2:4].mean(), atol=1e-9)),
              f"index={q[2]:.3f} ring={q[4]:.3f} pinky={q[5]:.3f}")

    print("\npolicy's commanded grasp")
    for side in ("left", "right"):
        cmd = DEX3_GRASP if side == "left" else -DEX3_GRASP
        c_src = grasp_closure_batch(cmd[None, :], side)[0]
        q = retarget_batch(cmd[None, :], side)[0]
        print(f"  {side}: Dex3 closure {c_src:.3f} -> Revo2 "
              f"{np.array2string(q, precision=3, suppress_small=True)}")
        check(f"{side}: grasp actually closes the fingers", q[2] > 0.5)

    print()
    if _failures:
        print(f"{len(_failures)} check(s) failed: {', '.join(_failures)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
