"""Turn an eval run's outcomes into a spawn-region plan for a targeted generation round.

Round 2 sampled the apple's +/-5 cm spawn box uniformly. The 100-rollout eval of checkpoint-5000
showed success is far from uniform over that box: about 0.90 near the plate at moderate reach,
falling to 0.00 at the far-reach / far-from-plate corner. This splits the box into cells, reads
each cell's eval success rate, and allocates new demos in proportion to how badly it does, so the
next round is aimed rather than simply larger.

Cells hold only a handful of eval episodes each, so the weight is a Beta(1,1) posterior mean
failure rate rather than the raw rate -- otherwise a cell that happened to go 0/3 would swallow
the budget. Cells already succeeding above --max_success get nothing.

  python3 plan_targeted.py --total 1000 --out $EVAL_DIR/round3_plan.json

The apple positions come from an eval summary.json when it carries spawn_xy, else from the arrays
saved by the eval analysis (eval100_spawn_xy.npy / eval100_success.npy). Those were recovered from
the recorded video through a pixel-to-world homography and carry roughly +/-1 cm of noise, so
treat the cell boundaries as approximate.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

# Nominal apple spawn, from isaaclab_arena_environments.galileo_g1_static_pick_and_place_environment
NOMINAL_XY = (0.5785, 0.27)
# Same default as env.sh, so sourcing it (or exporting EVAL_DIR) is enough to relocate this.
EVAL_DIR = os.environ.get(
    "EVAL_DIR", os.path.expanduser("~/g1_baseline/eval/g1_apple_mimic")
)


def load_outcomes(args):
    """Per-episode (offset from the run's spawn centre in cm, success flag)."""
    if args.summary and os.path.exists(args.summary):
        eps = json.load(open(args.summary))["episodes"]
        tagged = [e for e in eps if e.get("spawn_xy")]
        if tagged:
            xy = np.array([e["spawn_xy"] for e in tagged], float)
            ok = np.array([bool(e["success"]) for e in tagged])
            return (xy - xy.mean(0)) * 100.0, ok
        print(f"{args.summary} carries no spawn_xy; using the pixel-derived positions instead")
    xy = np.load(args.spawn_npy)
    ok = np.load(args.outcome_npy)
    # Offsets from the run's own centre, which cancels the constant bias in the pixel->world fit.
    return (xy - xy.mean(0)) * 100.0, ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="", help="eval summary.json, used when it has spawn_xy")
    p.add_argument("--spawn_npy", default=f"{EVAL_DIR}/eval100_spawn_xy.npy")
    p.add_argument("--outcome_npy", default=f"{EVAL_DIR}/eval100_success.npy")
    p.add_argument("--total", type=int, default=1000, help="new demos to allocate")
    p.add_argument("--half_width_cm", type=float, default=5.0, help="the spawn box being covered")
    p.add_argument("--cells", type=int, default=3, help="grid resolution per axis")
    p.add_argument(
        "--max_success",
        type=float,
        default=0.70,
        help="cells already succeeding above this get no budget",
    )
    p.add_argument("--min_episodes", type=int, default=3, help="cells with fewer eval episodes are skipped")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    off, ok = load_outcomes(args)
    h = args.half_width_cm
    edges = np.linspace(-h, h, args.cells + 1)
    # Widen the outermost bounds so positions the pixel fit pushed just outside still land.
    bounds = list(edges)
    bounds[0], bounds[-1] = -99.0, 99.0

    regions = []
    for i in range(args.cells):
        for j in range(args.cells):
            sel = (
                (off[:, 0] >= bounds[i])
                & (off[:, 0] < bounds[i + 1])
                & (off[:, 1] >= bounds[j])
                & (off[:, 1] < bounds[j + 1])
            )
            n = int(sel.sum())
            wins = int(ok[sel].sum()) if n else 0
            rate = wins / n if n else None
            weight = (n - wins + 1) / (n + 2) if n >= args.min_episodes else 0.0
            if rate is not None and rate > args.max_success:
                weight = 0.0
            regions.append(
                {
                    "x_off_cm": [float(edges[i]), float(edges[i + 1])],
                    "y_off_cm": [float(edges[j]), float(edges[j + 1])],
                    "eval_episodes": n,
                    "eval_successes": wins,
                    "eval_rate": rate,
                    "weight": weight,
                }
            )

    total_weight = sum(r["weight"] for r in regions)
    if total_weight == 0:
        raise SystemExit("no cell qualified; loosen --max_success or --min_episodes")
    for r in regions:
        r["demos"] = int(round(args.total * r["weight"] / total_weight))
    active = sorted((r for r in regions if r["demos"] > 0), key=lambda r: -r["demos"])
    active[0]["demos"] += args.total - sum(r["demos"] for r in active)

    plan = {
        "nominal_xy": list(NOMINAL_XY),
        "total_demos": sum(r["demos"] for r in active),
        "source_eval": args.summary or args.spawn_npy,
        "regions": active,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(plan, f, indent=2)

    print(f"{plan['total_demos']} demos over {len(active)} regions (offsets in cm from nominal)")
    print("X is reach away from the robot, Y is away from the plate\n")
    print(f"{'X offset':>16} {'Y offset':>16} {'eval':>12} {'demos':>7}")
    for r in active:
        rate = (
            "-"
            if r["eval_rate"] is None
            else f"{r['eval_successes']}/{r['eval_episodes']}={r['eval_rate']:.2f}"
        )
        print(
            f"  [{r['x_off_cm'][0]:+5.1f},{r['x_off_cm'][1]:+5.1f}]"
            f"    [{r['y_off_cm'][0]:+5.1f},{r['y_off_cm'][1]:+5.1f}]"
            f" {rate:>12} {r['demos']:7d}"
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
