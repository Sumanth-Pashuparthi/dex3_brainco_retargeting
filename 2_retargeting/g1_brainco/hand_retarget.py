"""Dex3-1 (7 DoF/hand) to BrainCo Revo2 Touch (6 active DoF/hand) action retargeting.

Two directions:

    retarget_batch   policy Dex3 joint targets  ->  Revo2 joint targets
    inverse_batch    measured Revo2 joints      ->  pseudo-Dex3 state for the policy

NumPy only and vectorised over environments, so it runs inside the Isaac Sim container without
extra dependencies. Every table it uses is measured in hand_specs.py, which analyze_hand_geometry.py
derives and install.sh copies in alongside this file.
"""

from __future__ import annotations

import os

import numpy as np

from .hand_specs import (
    BRAINCO_ACTIVE_ORDER,
    BRAINCO_LIMITS,
    DEX3_CLOSED_LEFT,
    DEX3_CLOSURE,
    DEX3_GROOT_ORDER,
    REVO2_CLOSURE,
    THUMB_CF,
    THUMB_META,
    THUMB_PROX,
)

__all__ = [
    "BRAINCO_ACTIVE_ORDER",
    "DEX3_GROOT_ORDER",
    "aperture_match",
    "aperture_unmatch",
    "dex3_closure_batch",
    "grasp_closure_batch",
    "inverse_batch",
    "retarget_batch",
]

# Closed-direction angle per Dex3 joint. Open is zero for all of them, and the right hand is the
# sign mirror of the left.
_DEX3_CLOSED = {"left": DEX3_CLOSED_LEFT, "right": -DEX3_CLOSED_LEFT}

# Ring and pinky have no Dex3 counterpart; they follow the mean closure of index and middle.
SYNERGY_GAIN = 1.0

# Grasp depth beyond the aperture-matched value, for A/B testing without editing the tables.
_DEPTH = float(os.environ.get("BRAINCO_GRASP_DEPTH", "1.0"))

# The calibration saturates at closure 0.6, so only its rising part can be inverted. That branch is
# the whole usable range in the observation direction, since the hand never closes past it.
_RISING = np.flatnonzero(np.diff(REVO2_CLOSURE) > 0).max() + 2
_INV_REVO2 = REVO2_CLOSURE[:_RISING]
_INV_DEX3 = DEX3_CLOSURE[:_RISING]


def aperture_match(closure: np.ndarray) -> np.ndarray:
    """Dex3 closure fraction -> Revo2 closure fraction giving the same fingertip aperture."""
    return np.clip(_DEPTH * np.interp(closure, DEX3_CLOSURE, REVO2_CLOSURE), 0.0, 1.0)


def aperture_unmatch(closure: np.ndarray) -> np.ndarray:
    """Inverse of :func:`aperture_match`."""
    return np.interp(closure, _INV_REVO2, _INV_DEX3)


def dex3_closure_batch(q_dex3: np.ndarray, side: str) -> np.ndarray:
    """(N,7) Dex3 angles -> (N,7) closure fractions in [0,1]."""
    span = _DEX3_CLOSED[side]
    # thumb_0 is a rotation with no closing sense, so its span is zero and its fraction is forced
    # to zero rather than dividing by it.
    degenerate = np.abs(span) < 1e-9
    frac = np.asarray(q_dex3, dtype=float) / np.where(degenerate, 1.0, span)
    return np.clip(np.where(degenerate, 0.0, frac), 0.0, 1.0)


def grasp_closure_batch(q_dex3: np.ndarray, side: str) -> np.ndarray:
    """(N,7) Dex3 angles -> (N,) single grasp closure.

    The Dex3 pinches as one unit in this task and the aperture calibration was derived from a
    uniform closure sweep, so per-finger closures are averaged rather than driven independently.
    """
    c = dex3_closure_batch(q_dex3, side)
    c_index = 0.5 * (c[:, 0] + c[:, 1])
    c_middle = 0.5 * (c[:, 2] + c[:, 3])
    c_thumb = 0.5 * (c[:, 5] + c[:, 6])
    return (c_index + c_middle + c_thumb) / 3.0


def retarget_batch(q_dex3: np.ndarray, side: str) -> np.ndarray:
    """(N,7) Dex3 joint targets -> (N,6) Revo2 joint targets, in BRAINCO_ACTIVE_ORDER."""
    cf = aperture_match(grasp_closure_batch(q_dex3, side))
    cf_syn = np.clip(SYNERGY_GAIN * cf, 0.0, 1.0)

    q = np.zeros((len(cf), 6))
    q[:, 0] = np.interp(cf, THUMB_CF, THUMB_META)
    q[:, 1] = np.interp(cf, THUMB_CF, THUMB_PROX)
    q[:, 2] = cf * BRAINCO_LIMITS[2, 1]
    q[:, 3] = cf * BRAINCO_LIMITS[3, 1]
    q[:, 4] = cf_syn * BRAINCO_LIMITS[4, 1]
    q[:, 5] = cf_syn * BRAINCO_LIMITS[5, 1]
    return np.clip(q, BRAINCO_LIMITS[:, 0], BRAINCO_LIMITS[:, 1])


def inverse_batch(q_brainco: np.ndarray, side: str) -> np.ndarray:
    """(N,6) measured Revo2 angles -> (N,7) pseudo-Dex3 state the frozen policy expects.

    A finger's Dex3 chain is reconstructed by giving both of its joints the Revo2 finger's closure
    fraction, which is consistent with how :func:`retarget_batch` collapses them.
    """
    q = np.asarray(q_brainco, dtype=float)
    # Closure is read off the index and middle flexors. The thumb follows the opposition schedule
    # rather than a plain closure ramp, so its angle is not a usable measure of hand closure.
    cf = 0.5 * (q[:, 2] / BRAINCO_LIMITS[2, 1] + q[:, 3] / BRAINCO_LIMITS[3, 1])
    c = aperture_unmatch(np.clip(cf, 0.0, 1.0))

    frac = np.stack([c, c, c, c, np.zeros_like(c), c, c], axis=1)
    frac[:, 4] = 0.0  # thumb_0 is a rotation with no Revo2 counterpart
    return frac * _DEX3_CLOSED[side]
