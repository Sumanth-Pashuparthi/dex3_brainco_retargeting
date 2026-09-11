"""Build Mimic *source* demos from an existing Mimic *generated* dataset, optionally time-stretched.

Why: the harvested rollouts (``revo2_demos.hdf5``) and the first ``source_demos.hdf5`` are gone, but
every generated episode is itself a successful 23-D Pink demo in the Revo2 embodiment, with the exact
``initial_state`` it was replayed from. ``annotate_demos.py`` only consumes ``initial_state`` +
``actions`` (it re-simulates and records fresh obs/states), so a subset of the generated set is a
valid source set for another generation round -- with no policy server, no 6 % harvest rate.

Time-stretching: Mimic executes one source step per environment step, so the generated demos move as
fast as their sources. The first round's sources came from a GR00T rollout (reach in 1.5 s) followed
by a scripted place (apple airborne < 1 s, wrist ~0.6 m/s with 2 m/s spikes), all squeezed under the
static task's 6 s ``episode_length_s``. ``--time_scale 2`` resamples the action sequence to twice the
steps (positions linear, quaternions slerp, binary hand states nearest, body commands linear), which
halves every velocity. Trailing steps where the action no longer changes (the settle padding) are
capped at ``--pad_steps`` so the stretch does not double the dead time at the end.

Source selection: ``--num_sources K`` picks K demos by farthest-point sampling on the initial apple
XY, so Mimic's ``nearest_neighbor_object`` selection has sources spread over the spawn box rather
than K near-duplicates.

    python3 g1_apple_mimic/make_source_from_generated.py \
        /datasets/g1_apple_mimic/g1_apple_mimic_generated.hdf5 /datasets/g1_apple_mimic/source_demos.hdf5 \
        --num_sources 24 --time_scale 2.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as R, Slerp  # noqa: E402

APPLE = "apple_01_objaverse_robolab"
# 23-D Pink action layout (see convert_joint_demos_to_pink_source.py)
HAND = [0, 1]
L_POS, L_QUAT = slice(2, 5), slice(5, 9)
R_POS, R_QUAT = slice(9, 12), slice(12, 16)
LINEAR = [slice(2, 5), slice(9, 12), slice(16, 19), slice(19, 20), slice(20, 23)]


def _continuous_sign(q: np.ndarray) -> np.ndarray:
    q = q.copy()
    for t in range(1, len(q)):
        if np.dot(q[t], q[t - 1]) < 0:
            q[t] = -q[t]
    return q


def _slerp(q_xyzw: np.ndarray, t_src: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    q = _continuous_sign(q_xyzw.astype(np.float64))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    out = Slerp(t_src, R.from_quat(q))(t_new).as_quat()
    return _continuous_sign(out).astype(np.float32)


def trailing_static_steps(a: np.ndarray, tol: float = 0.02) -> int:
    """Trailing steps with both hands open and the active wrist within ``tol`` m of its final position.

    Generated demos carry Mimic's per-step action noise (5 mm), so the settle padding is never
    bit-identical; a position tolerance finds it anyway.
    """
    final_l, final_r = a[-1, L_POS], a[-1, R_POS]
    n = 0
    for t in range(len(a) - 1, -1, -1):
        near = np.linalg.norm(a[t, L_POS] - final_l) < tol and np.linalg.norm(a[t, R_POS] - final_r) < tol
        if not near or a[t, 0] > 0.5 or a[t, 1] > 0.5:
            break
        n += 1
    return n


def retime(actions: np.ndarray, scale: float, pad_steps: int) -> np.ndarray:
    T = len(actions)
    static = trailing_static_steps(actions)
    body = actions[: T - static] if static > 0 else actions
    Tb = len(body)
    Tn = max(2, int(round(Tb * scale)))
    t_src = np.arange(Tb, dtype=np.float64)
    t_new = np.linspace(0.0, Tb - 1, Tn)
    out = np.zeros((Tn, actions.shape[1]), np.float32)
    for sl in LINEAR:
        for c in range(sl.start, sl.stop):
            out[:, c] = np.interp(t_new, t_src, body[:, c])
    out[:, L_QUAT] = _slerp(body[:, L_QUAT], t_src, t_new)
    out[:, R_QUAT] = _slerp(body[:, R_QUAT], t_src, t_new)
    nearest = np.clip(np.rint(t_new).astype(int), 0, Tb - 1)
    for c in HAND:
        out[:, c] = (body[nearest, c] > 0.5).astype(np.float32)
    # re-append a fixed settle tail: hold the final pose, hands as they ended (open)
    tail = np.repeat(out[-1:], pad_steps, axis=0)
    return np.concatenate([out, tail], axis=0)


def farthest_point(xy: np.ndarray, k: int, seed: int = 0) -> list[int]:
    n = len(xy)
    if k >= n:
        return list(range(n))
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(n))]
    d = np.linalg.norm(xy - xy[chosen[0]], axis=1)
    while len(chosen) < k:
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(xy - xy[i], axis=1))
    return sorted(chosen)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="generated HDF5 (merged or one shard)")
    ap.add_argument("output", help="source HDF5 for annotate_demos.py")
    ap.add_argument("--num_sources", type=int, default=24)
    ap.add_argument("--demos", default=None, help="explicit comma-separated demo indices (overrides --num_sources)")
    ap.add_argument("--time_scale", type=float, default=2.0, help="1.0 keeps the original speed")
    ap.add_argument("--pad_steps", type=int, default=60, help="settle steps appended after the last motion")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env_name", default="galileo_g1_static_apple_mimic")
    ap.add_argument("--target_embodiment", default="g1_wbc_agile_pink_brainco")
    args = ap.parse_args()

    with h5py.File(args.input, "r") as fin, h5py.File(args.output, "w") as fout:
        din = fin["data"]
        names = sorted([k for k in din if k.startswith("demo_")], key=lambda s: int(s.split("_")[1]))
        ok = [n for n in names if bool(din[n].attrs.get("success", True))]
        xy = np.stack([din[n][f"initial_state/rigid_object/{APPLE}/root_pose"][0, :2] for n in ok])
        if args.demos:
            keep = [int(x) for x in args.demos.split(",") if x.strip()]
        else:
            keep = farthest_point(xy, args.num_sources, args.seed)
        print(f"{len(ok)} successful demos in {args.input}; using {len(keep)}: {keep}")
        print(f"apple xy of chosen: x[{xy[keep, 0].min():.3f},{xy[keep, 0].max():.3f}] "
              f"y[{xy[keep, 1].min():.3f},{xy[keep, 1].max():.3f}]")

        env_args = json.loads(din.attrs["env_args"]) if "env_args" in din.attrs else {}
        env_args["env_name"] = args.env_name
        env_args["type"] = 2
        sim_args = dict(env_args.get("sim_args", {}))
        sim_args["num_envs"] = 1
        env_args["sim_args"] = sim_args
        env_args["target_embodiment"] = args.target_embodiment
        env_args["retimed_from"] = {"file": os.path.basename(args.input), "time_scale": args.time_scale,
                                    "pad_steps": args.pad_steps, "demos": keep}
        fout.attrs["format_version"] = int(fin.attrs.get("format_version", 1))
        dout = fout.create_group("data")
        dout.attrs["env_args"] = json.dumps(env_args)

        total = 0
        for j, i in enumerate(keep):
            g = din[ok[i]]
            a = g["actions"][()].astype(np.float32)
            assert a.shape[1] == 23, a.shape
            assert set(np.unique(np.rint(a[:, HAND]))) <= {0.0, 1.0}, "hand columns are not binary"
            a_new = retime(a, args.time_scale, args.pad_steps) if args.time_scale != 1.0 else a
            og = dout.create_group(f"demo_{j}")
            og.attrs["num_samples"] = len(a_new)
            og.attrs["success"] = True
            og.attrs["source_demo"] = ok[i]
            og.create_dataset("actions", data=a_new)
            g.copy(g["initial_state"], og, name="initial_state")
            close = np.where(a_new[:, 0] > 0.5)[0]
            static = trailing_static_steps(a)
            print(f"  {ok[i]} -> demo_{j}: {len(a)} steps (tail {static} static) -> {len(a_new)} steps "
                  f"({len(a_new) / 50:.1f}s), "
                  + (f"left hand closed {close[0] / 50:.2f}s..{close[-1] / 50:.2f}s" if len(close)
                     else "WARNING no left-hand closure"))
            total += len(a_new)
        dout.attrs["total"] = total
        print(f"\nwrote {len(keep)} source demos, {total} steps -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
