"""Convert G1 *joint-embodiment* rollout demos (Revo2/BrainCo, 50-D actions) into a Mimic source file
for the Pink IK embodiment (``g1_wbc_agile_pink``, 23-D actions, Dex3 hands).

Isaac Lab Mimic for G1 stitches wrist-pose trajectories through ``G1MimicEnv`` which only exists for
the Pink embodiment. The joint rollouts already contain everything Mimic needs, just not in that shape:

    23-D Pink action[t] = [left_hand_state, right_hand_state,
                           left_wrist_pos(3), left_wrist_quat_xyzw(4),      <- obs/left_wrist_pose_pelvis_frame[t+1]
                           right_wrist_pos(3), right_wrist_quat_xyzw(4),    <- obs/right_wrist_pose_pelvis_frame[t+1]
                           navigate_cmd(3)=0, base_height_cmd(1), torso_rpy_cmd(3)]  <- action/* groups

    hand_state[t] = 1 if the commanded Dex3 finger targets (processed_actions_dex3[t], 43-DoF yaml order)
                    are closed (abs-mean over that hand's 7 joints > --close_thresh), else 0
    initial_state = same robot root pose + apple/plate poses; robot joints = the 29 shared body joints
                    (identical articulation order: the 43-DoF yaml order == Isaac Lab breadth-first order
                    whose first 29 entries match the Revo2 robot's joint_names) + 14 Dex3 hand joints at 0 (open)

Only ``initial_state`` + ``actions`` are consumed by ``annotate_demos.py`` (it re-simulates and records
fresh obs/states), so ``states``/``obs``/camera data are not copied. Output gets ``format_version=1``
(xyzw) and an ``env_args`` header naming the Mimic env.

    python g1_apple_mimic/convert_joint_demos_to_pink_source.py \
        /datasets/rollouts/revo2_demos_all.hdf5 /datasets/g1_apple_mimic/source_demos.hdf5
"""

from __future__ import annotations

import argparse
import json
import sys

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

NUM_BODY_JOINTS = 29
NUM_DEX3_JOINTS = 43
# indices of each hand's 7 joints in isaaclab_arena_gr00t/embodiments/g1/43dof_joint_space.yaml
DEX3_LEFT_HAND_IDX = [29, 30, 31, 35, 36, 37, 41]
DEX3_RIGHT_HAND_IDX = [32, 33, 34, 38, 39, 40, 42]
BODY_JOINT_NAMES_29 = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint",
    "right_hip_roll_joint", "waist_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint",
    "waist_pitch_joint", "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_elbow_joint",
    "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint", "left_wrist_pitch_joint",
    "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]  # fmt: skip


def pose_mat_to_pos_quat_xyzw(mats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = mats[:, :3, 3]
    quat = R.from_matrix(mats[:, :3, :3]).as_quat()  # xyzw
    # keep a continuous sign (avoid q -> -q flips between steps)
    for t in range(1, len(quat)):
        if np.dot(quat[t], quat[t - 1]) < 0:
            quat[t] = -quat[t]
    return pos.astype(np.float32), quat.astype(np.float32)


def hand_state_from_dex3(dex3: np.ndarray, idx: list[int], thresh: float, min_hold: int) -> np.ndarray:
    closed = np.abs(dex3[:, idx]).mean(axis=1) > thresh
    # debounce: drop closures shorter than min_hold steps (policy chatter)
    out = closed.copy()
    t = 0
    while t < len(out):
        if out[t]:
            s = t
            while t < len(out) and out[t]:
                t += 1
            if t - s < min_hold:
                out[s:t] = False
        else:
            t += 1
    return out.astype(np.float32)


def _smoothstep(n: int) -> np.ndarray:
    s = np.linspace(0.0, 1.0, n + 1)[1:]
    return s * s * (3.0 - 2.0 * s)


def _segment(p0: np.ndarray, p1: np.ndarray, n: int) -> np.ndarray:
    return p0[None] + _smoothstep(n)[:, None] * (p1 - p0)[None]


def scripted_place(actions: np.ndarray, src: h5py.Group, L: np.ndarray, left_hand: np.ndarray, args) -> np.ndarray:
    """Replace everything after the grasp with a clean lift -> carry over the plate centre -> lower -> open -> retreat.

    The retargeted policy's own carry/release is sloppy (it opens while swinging the arm back, releasing the apple
    on the plate rim, 5-9 cm left of centre) and does not survive open-loop replay through Pink IK. Mimic only
    needs the *grasp* segment from the human/policy demo; the place segment is object-relative to the plate,
    which is not randomised, so a scripted one is both cleaner training data and far more repeatable.
    All poses are in the source pelvis frame (== the anchor frame used by the Revo2 Pink action term).
    """
    t_c = int(np.argmax(left_hand))
    t_cut = min(t_c + args.place_hold_src, len(actions) - 1)
    root = src["states/articulation/robot/root_pose"][()]  # (T, 7) env-local, quat xyzw
    apple = src["states/rigid_object/apple_01_objaverse_robolab/root_pose"][()][:, :3]
    plate = src["states/rigid_object/clay_plates_hot3d_robolab/root_pose"][()][:, :3]

    def to_pelvis(p_w: np.ndarray, t: int) -> np.ndarray:
        return R.from_quat(root[t, 3:7]).inv().apply(p_w - root[t, :3])

    apple_p = to_pelvis(apple[t_cut], t_cut)
    plate_p = to_pelvis(plate[0], 0)
    wrist_p = L[t_cut, :3, 3]
    in_hand = apple_p - wrist_p  # apple offset from the wrist while held (pelvis frame; orientation is held)

    p0 = actions[t_cut, 2:5].copy()
    q0 = actions[t_cut, 5:9].copy()
    z_carry = p0[2] + args.place_lift
    p_over = np.array([plate_p[0] - in_hand[0], plate_p[1] - in_hand[1], z_carry], np.float32)
    p_release = p_over.copy()
    p_release[2] = p0[2] + args.place_release_dz
    p_home = actions[0, 2:5].copy()
    p_home[2] = max(p_home[2], p_release[2]) + 0.05

    pos = np.concatenate([
        _segment(p0, np.array([p0[0], p0[1], z_carry], np.float32), 15),
        _segment(np.array([p0[0], p0[1], z_carry], np.float32), p_over, 25),
        np.repeat(p_over[None], 5, axis=0),
        _segment(p_over, p_release, 12),
        np.repeat(p_release[None], 5, axis=0),           # settle, still closed
    ])
    n_closed = len(pos)
    pos = np.concatenate([
        pos,
        np.repeat(p_release[None], 12, axis=0),          # open, let the apple drop
        _segment(p_release, p_release + np.array([0, 0, 0.08], np.float32), 10),
        _segment(p_release + np.array([0, 0, 0.08], np.float32), p_home, 25),
    ])
    n = len(pos)
    tail = np.repeat(actions[t_cut][None], n, axis=0)
    tail[:, 0] = 0.0
    tail[:n_closed, 0] = 1.0
    tail[:, 1] = 0.0
    tail[:, 2:5] = pos
    tail[:, 5:9] = q0
    tail[:, 16:19] = 0.0  # nav
    out = np.concatenate([actions[: t_cut + 1], tail], axis=0).astype(np.float32)
    print(f"    scripted place: cut@{t_cut} (grasp@{t_c}), in-hand offset {in_hand.round(3)}, plate(pelvis) {plate_p.round(3)}, "
          f"release wrist {p_release.round(3)}, T {len(actions)} -> {len(out)}")
    return out


def convert_demo(src: h5py.Group, args, body_joint_names_src: list[str]) -> dict:
    T = src["actions"].shape[0]
    L = src["obs/left_wrist_pose_pelvis_frame"][()]
    Rw = src["obs/right_wrist_pose_pelvis_frame"][()]
    dex3 = src["processed_actions_dex3"][()]
    assert L.shape == (T, 4, 4) and dex3.shape == (T, NUM_DEX3_JOINTS), (L.shape, dex3.shape)

    # target at t = pose reached at t+1 (what the joint policy's action produced); last step holds.
    nxt = np.concatenate([np.arange(1, T), [T - 1]]) + 0
    if args.lookahead != 1:
        nxt = np.clip(np.arange(T) + args.lookahead, 0, T - 1)
    lpos, lquat = pose_mat_to_pos_quat_xyzw(L[nxt])
    rpos, rquat = pose_mat_to_pos_quat_xyzw(Rw[nxt])

    left_hand = hand_state_from_dex3(dex3, DEX3_LEFT_HAND_IDX, args.close_thresh, args.min_hold)
    right_hand = hand_state_from_dex3(dex3, DEX3_RIGHT_HAND_IDX, args.close_thresh, args.min_hold)
    if args.idle_arm == "right":
        right_hand[:] = 0.0
    elif args.idle_arm == "left":
        left_hand[:] = 0.0

    nav = np.zeros((T, 3), np.float32)
    base_h = src["action/base_height_cmd"][()].astype(np.float32).reshape(T, 1)
    torso = src["action/torso_orientation_rpy_cmd"][()].astype(np.float32).reshape(T, 3)

    actions = np.concatenate(
        [left_hand[:, None], right_hand[:, None], lpos, lquat, rpos, rquat, nav, base_h, torso], axis=1
    ).astype(np.float32)
    assert actions.shape == (T, 23), actions.shape

    if args.scripted_place and left_hand.any():
        actions = scripted_place(actions, src, L, left_hand, args)
        T = actions.shape[0]

    if args.z_offset:
        actions[:, 2 + 2] += args.z_offset  # left wrist z (cols 2:5 = left pos)

    # Settle padding: the source episodes end the moment success fires; the Pink IK replay lags, so the
    # apple is often still in the hand / moving at the last step. Hold the final wrist pose with both
    # hands OPEN for a few more steps so the apple can land on the plate and come to rest.
    if args.pad_steps > 0:
        tail = np.repeat(actions[-1:], args.pad_steps, axis=0)
        tail[:, 0:2] = 0.0
        actions = np.concatenate([actions, tail], axis=0)
        T = actions.shape[0]

    q_src = src["initial_state/articulation/robot/joint_position"][()]  # (1, 41)
    if args.target == "revo2":
        # Same articulation (g1_wbc_agile_pink_brainco shares the Revo2 USD): copy the joint state verbatim.
        q_dst = q_src.astype(np.float32)
        dq_dst = src["initial_state/articulation/robot/joint_velocity"][()].astype(np.float32)
    else:
        # Dex3 target: robot root + 29 shared body joints (by name) + 14 open Dex3 hand joints
        name_to_val = {n: q_src[0, i] for i, n in enumerate(body_joint_names_src)}
        q_dst = np.zeros((1, NUM_DEX3_JOINTS), np.float32)
        for j, n in enumerate(BODY_JOINT_NAMES_29):
            q_dst[0, j] = name_to_val[n]
        dq_dst = np.zeros((1, NUM_DEX3_JOINTS), np.float32)

    init = {
        "articulation/robot/root_pose": src["initial_state/articulation/robot/root_pose"][()],
        "articulation/robot/root_velocity": src["initial_state/articulation/robot/root_velocity"][()],
        "articulation/robot/joint_position": q_dst,
        "articulation/robot/joint_velocity": dq_dst,
    }
    for obj in src["initial_state/rigid_object"].keys():
        for k in ("root_pose", "root_velocity"):
            init[f"rigid_object/{obj}/{k}"] = src[f"initial_state/rigid_object/{obj}/{k}"][()]

    stats = {
        "T": T,
        "left_closed_steps": int(left_hand.sum()),
        "right_closed_steps": int(right_hand.sum()),
        "first_left_close": int(np.argmax(left_hand)) if left_hand.any() else -1,
    }
    return {"actions": actions, "initial_state": init, "stats": stats}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--meta", default=None, help="<input>.meta.json with joint_names (default: next to input)")
    ap.add_argument("--close_thresh", type=float, default=0.15, help="abs-mean rad of Dex3 hand targets => closed")
    ap.add_argument("--min_hold", type=int, default=3, help="ignore closures shorter than this many steps")
    ap.add_argument("--lookahead", type=int, default=1, help="target pose = measured pose at t+lookahead")
    ap.add_argument("--pad_steps", type=int, default=40, help="append this many hold-pose/open-hand steps")
    ap.add_argument("--z_offset", type=float, default=0.0,
                    help="add to every left-wrist z target (m); compensates the Pink-IK/PD gravity sag (~+0.02)")
    ap.add_argument("--scripted_place", action="store_true",
                    help="replace the post-grasp carry/release with a scripted lift -> plate centre -> lower -> open")
    ap.add_argument("--place_hold_src", type=int, default=12, help="source steps kept after the close command")
    ap.add_argument("--place_lift", type=float, default=0.10, help="carry height above the grasp pose (m)")
    ap.add_argument("--place_release_dz", type=float, default=0.04, help="release height above the grasp pose (m)")
    ap.add_argument("--idle_arm", choices=("none", "left", "right"), default="none",
                    help="force this arm's hand_state to 0 (open) everywhere")
    ap.add_argument("--env_name", default="galileo_g1_static_apple_mimic")
    ap.add_argument("--only_success", action="store_true", default=True)
    ap.add_argument("--target", choices=("dex3", "revo2"), default="revo2",
                    help="target Pink embodiment: revo2 = g1_wbc_agile_pink_brainco (initial_state copied "
                         "verbatim), dex3 = g1_wbc_agile_pink (hand joints rebuilt as open Dex3)")
    ap.add_argument("--demos", default=None,
                    help="comma-separated source demo indices to keep (e.g. 0,6,7); default all")
    args = ap.parse_args()
    keep = None if args.demos is None else {int(x) for x in args.demos.split(",") if x.strip()}

    meta_path = args.meta or args.input.replace(".hdf5", ".meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    joint_names_src = meta["joint_names"]
    assert joint_names_src[:NUM_BODY_JOINTS] == BODY_JOINT_NAMES_29, "unexpected source body joint order"

    with h5py.File(args.input, "r") as fin, h5py.File(args.output, "w") as fout:
        din = fin["data"]
        env_args = json.loads(din.attrs.get("env_args", "{}")) if "env_args" in din.attrs else {}
        env_args["env_name"] = args.env_name
        env_args["type"] = 2
        sim_args = dict(env_args.get("sim_args", {}))
        sim_args["num_envs"] = 1
        env_args["sim_args"] = sim_args
        env_args["converted_from"] = {"file": args.input, "embodiment": meta.get("embodiment"), "action_dim": 50}
        env_args["target_embodiment"] = {"revo2": "g1_wbc_agile_pink_brainco", "dex3": "g1_wbc_agile_pink"}[args.target]

        fout.attrs["format_version"] = 1
        dout = fout.create_group("data")
        dout.attrs["env_args"] = json.dumps(env_args)

        names = sorted([k for k in din if k.startswith("demo_")], key=lambda s: int(s.split("_")[1]))
        n_out = total = 0
        for name in names:
            g = din[name]
            if keep is not None and int(name.split("_")[1]) not in keep:
                print(f"{name}: skipped (not in --demos)")
                continue
            if args.only_success and "success" in g.attrs and not bool(g.attrs["success"]):
                print(f"{name}: skipped (success=False)")
                continue
            ep = convert_demo(g, args, joint_names_src)
            s = ep["stats"]
            if s["left_closed_steps"] == 0 and s["right_closed_steps"] == 0:
                print(f"{name}: skipped (no hand closure detected)")
                continue
            og = dout.create_group(f"demo_{n_out}")
            og.attrs["num_samples"] = s["T"]
            og.attrs["success"] = True
            og.attrs["source_demo"] = name
            og.create_dataset("actions", data=ep["actions"])
            for k, v in ep["initial_state"].items():
                og.create_dataset(f"initial_state/{k}", data=v)
            print(
                f"{name} -> demo_{n_out}: T={s['T']}  left closed {s['left_closed_steps']} steps "
                f"(first at {s['first_left_close']}), right closed {s['right_closed_steps']} steps"
            )
            total += s["T"]
            n_out += 1
        dout.attrs["total"] = total
        print(f"\nwrote {n_out} demos ({total} steps) -> {args.output}")
    return 0 if n_out > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
