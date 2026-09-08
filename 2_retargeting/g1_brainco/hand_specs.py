"""Measured constants for the two hands, used by analyze_hand_geometry.py and g1_brainco/.

Sources:
  [S1] Unitree Dex3-1 URDF shipped with the whole-body-control model data
  [S2] GR00T G1 embodiment modality config (joint order of the 7-dim hand slice)
  [S3] Unitree G1 29-DoF URDF (hand mount origins on *_wrist_yaw_link)
  [S4] Unitree G1 + BrainCo Revo2 URDF (urdf/)
  [S5] BrainCo Revo2 Touch product datasheet
"""

from __future__ import annotations

import numpy as np

# 7-dim Dex3 hand slice used by the GR00T G1 embodiment, per side. [S2]
DEX3_GROOT_ORDER = [
    "hand_index_0_joint",
    "hand_index_1_joint",
    "hand_middle_0_joint",
    "hand_middle_1_joint",
    "hand_thumb_0_joint",
    "hand_thumb_1_joint",
    "hand_thumb_2_joint",
]

# Closed-direction limit per Dex3 joint, in DEX3_GROOT_ORDER, left hand. [S1]
# thumb_0 is a rotation with no closing sense, hence 0. The right hand is the sign mirror.
DEX3_CLOSED_LEFT = np.array([-1.5708, -1.74533, -1.5708, -1.74533, 0.0, 1.0472, 1.74533])

# The Dex3 has no tip link, so fingertip points are taken as offsets in the last phalanx frame,
# measured from the collision meshes. [S1]
DEX3_TIP_OFFSETS = {
    "right": {
        "thumb": np.array([0.0, 0.052, 0.0]),
        "index": np.array([0.052, 0.0, 0.0]),
        "middle": np.array([0.052, 0.0, 0.0]),
    },
    "left": {
        "thumb": np.array([0.0, -0.052, 0.0]),
        "index": np.array([0.052, 0.0, 0.0]),
        "middle": np.array([0.052, 0.0, 0.0]),
    },
}
DEX3_TIP_LINKS = {
    "thumb": "hand_thumb_2_link",
    "index": "hand_index_1_link",
    "middle": "hand_middle_1_link",
}

# Revo2 actuated joints per side, and their limits. Zero is open. [S4]
BRAINCO_ACTIVE_ORDER = [
    "thumb_metacarpal_joint",
    "thumb_proximal_joint",
    "index_proximal_joint",
    "middle_proximal_joint",
    "ring_proximal_joint",
    "pinky_proximal_joint",
]
BRAINCO_LIMITS = np.array(
    [
        [0.0, 1.5184],
        [0.0, 1.0472],
        [0.0, 1.4661],
        [0.0, 1.4661],
        [0.0, 1.4661],
        [0.0, 1.4661],
    ]
)

# Thumb opposition schedule. Above index 1 this is the grid-search result from
# analyze_hand_geometry.py, to within the grid step. The first two entries are overridden so the
# hand is open at rest: unconstrained aperture minimisation would hold the thumb across the palm
# even at zero closure, and the hand would never open.
THUMB_CF = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
THUMB_META = np.array(
    [0.3500, 0.9000, 1.0437, 1.1066, 1.1467, 1.1724, 1.1910, 1.2753, 1.3383, 1.3969, 1.4483, 1.4968]
)
THUMB_PROX = np.array(
    [0.0000, 1.0000, 1.0472, 1.0472, 1.0472, 1.0472, 1.0472, 0.9524, 0.8592, 0.7590, 0.6570, 0.5496]
)

# Aperture calibration: which Revo2 closure reproduces the Dex3's fingertip opening.
#
# Strict aperture matching gives [0, 0, 0, 0.05, 0.3, 0.6, ...] — three entries at zero, because
# below Dex3 closure ~0.35 the Revo2 cannot open wide enough to match. A flat segment is a
# closed-loop hazard: the policy commands closure, nothing moves, and its next observation reports
# a fully open hand. The first four entries are therefore a ramp to 0.18, which costs little
# opening at closures the grasp never dwells at. From 0.4 upward these are the matched values.
DEX3_CLOSURE = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
REVO2_CLOSURE = np.array([0.0, 0.06, 0.12, 0.18, 0.30, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60])

# Mechanical coupling of the distal phalanges. [S4]
BRAINCO_DISTAL_RATIO = {"thumb": 1.0, "index": 1.155, "middle": 1.155, "ring": 1.155, "pinky": 1.155}

# Mount origins on *_wrist_yaw_link.
DEX3_MOUNT_XYZ = {"right": np.array([0.0415, -0.003, 0.0]),   # [S3]
                  "left": np.array([0.0415, 0.003, 0.0])}
REVO2_MOUNT_XYZ = np.array([0.0591, 0.0, 0.0])                # [S4], as supplied

# The grasp the frozen policy actually commands, read off a runtime trace, in DEX3_GROOT_ORDER,
# left hand.
DEX3_COMMANDED_GRASP = np.array([-0.598, -1.200, -0.600, -1.200, 0.000, 0.700, 0.700])
