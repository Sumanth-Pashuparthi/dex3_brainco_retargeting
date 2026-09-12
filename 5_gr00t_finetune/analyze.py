#!/usr/bin/env python3
"""Regenerate every figure and derived number for the fine-tuning stage.

Reads only ``results/`` and writes only ``figures/``, so it needs no GPU, no Isaac Sim and no
checkpoints -- the measurements are recorded. Paths are resolved relative to this file, so the
repository can live anywhere.

    python3 analyze.py            # figures + printed summary
    python3 analyze.py --stats    # printed summary only
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

# One colour per stage of the work, held constant across every figure.
C_FROZEN = "#b0403a"    # the frozen-policy baseline
C_R1R2 = "#2f6fa8"      # first fine-tune
C_R2R3 = "#2e8b57"      # second fine-tune
C_REF = "#8a8a8a"       # source-embodiment reference
C_WARN = "#c8892b"      # partial / stopped-early runs
EDGE = "#2a2a2a"

# Diverging for rates (red = bad), reversed for errors (where high is bad), sequential for counts.
CMAP_RATE = "RdYlGn"
CMAP_ERROR = "RdYlGn_r"
CMAP_COUNT = "Blues"


# --------------------------------------------------------------------------- statistics


def wilson(successes: int, n: int) -> tuple[float, float, float]:
    """Point estimate and 95% Wilson score interval.

    Wilson rather than normal-approximation because several configurations here have n as low as
    10, where the normal interval runs outside [0, 1] and understates the uncertainty.
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    z = 1.96
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def two_proportion_p(s1: int, n1: int, s2: int, n2: int) -> float:
    """Two-sided p-value for two independent success rates (pooled z-test)."""
    if n1 == 0 or n2 == 0:
        return float("nan")
    p1, p2 = s1 / n1, s2 / n2
    p = (s1 + s2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (p1 - p2) / se
    return math.erfc(abs(z) / math.sqrt(2))


def load() -> dict:
    data = json.loads((RESULTS / "results.json").read_text())
    for tr in data["trainings"]:
        curve = RESULTS / tr["loss_curve"]
        tr["curve"] = json.loads(curve.read_text())["points"] if curve.exists() else []
    return data


def ev(data: dict, eid: str) -> dict:
    for e in data["evals"]:
        if e["id"] == eid:
            return e
    raise KeyError(eid)


# --------------------------------------------------------------------------- figures


def _bar_with_ci(ax, labels, pairs, colors, *, annotate_n=True, ymax=1.0):
    """Bars with Wilson error bars. `pairs` is a list of (successes, n)."""
    xs = range(len(pairs))
    ps, los, his = [], [], []
    for s, n in pairs:
        p, lo, hi = wilson(s, n)
        ps.append(p)
        los.append(p - lo)
        his.append(hi - p)
    ax.bar(xs, ps, color=colors, width=0.62, zorder=3, edgecolor=EDGE, linewidth=0.9)
    ax.errorbar(xs, ps, yerr=[los, his], fmt="none", ecolor=EDGE, capsize=5, lw=1.3, zorder=4)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("success rate")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    if annotate_n:
        for x, (p, (s, n)) in enumerate(zip(ps, pairs)):
            ax.text(x, p + his[x] + 0.03, f"{s}/{n}\n{p:.2f}", ha="center", fontsize=8.5)
    return ps


def fig_headline(data, plt):
    """What the transfer bought, under matched +/-5 cm scoring where possible.

    Bars: Dex3 fixed (as-shipped reference), Dex3 +/-5 cm (matched protocol), Revo2 retarget-only
    (fixed), Revo2 fine-tuned +/-5 cm. The headline claim is Dex3 +/-5 cm 0.30 -> fine-tuned 0.50.
    The fixed-spawn fine-tuned score is excluded (reportable: false on ft_r1r2_c5000_j0).
    """
    rows = [ev(data, i) for i in ("frozen_dex3", "frozen_dex3_j05", "frozen_revo2", "ft_r1r2_c5000")]
    labels = ["Dex3-1 frozen\n(fixed spawn)",
              "Dex3-1 frozen\n(+/-5 cm spawn)",
              "Revo2 frozen\nretarget only (fixed)",
              "Revo2 fine-tuned\n(+/-5 cm spawn)"]
    fig, ax = plt.subplots(figsize=(8.2, 3.1))
    _bar_with_ci(
        ax,
        labels,
        [(r["successes"], r["n"]) for r in rows],
        [C_REF, C_REF, C_FROZEN, C_R1R2],
        ymax=0.96,
    )
    matched = rows[1]["successes"] / rows[1]["n"]
    ax.axhline(matched, ls="--", lw=1, color=C_REF, zorder=2)
    ax.text(1.5, matched + 0.02, "matched Dex3 +/-5 cm reference", fontsize=8, color=C_REF,
            ha="center")
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_title("Matched protocol: Dex3 +/-5 cm 0.30 -> Revo2 fine-tuned 0.50\n"
                 "retargeting alone is 0.06; the recovery is Mimic harvest + fine-tune",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_headline.png", dpi=170)
    plt.close(fig)


def fig_datasets(data, plt):
    """Where the training data came from, and what round 3 added."""
    ds = data["datasets"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    labels = [d["label"] for d in ds]
    colors = [C_R1R2, C_WARN, C_R2R3]

    a1.bar(range(3), [d["episodes"] for d in ds], color=colors, width=0.6, zorder=3)
    for i, d in enumerate(ds):
        a1.text(i, d["episodes"] + 40, f"{d['episodes']:,}", ha="center", fontsize=9)
    a1.set_xticks(range(3))
    a1.set_xticklabels(labels, fontsize=9)
    a1.set_ylabel("episodes")
    a1.set_title("Demonstrations per round", fontsize=10)
    a1.grid(axis="y", alpha=0.3, zorder=0)

    a2.bar(range(3), [d["frames"] / 1e6 for d in ds], color=colors, width=0.6, zorder=3)
    for i, d in enumerate(ds):
        a2.text(i, d["frames"] / 1e6 + 0.02, f"{d['frames']/1e6:.2f}M", ha="center", fontsize=9)
    a2.set_xticks(range(3))
    a2.set_xticklabels(labels, fontsize=9)
    a2.set_ylabel("frames (millions, 50 Hz)")
    a2.set_title("Frames per round", fontsize=10)
    a2.grid(axis="y", alpha=0.3, zorder=0)

    fig.suptitle("Round 3 is targeted, not more of the same", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_datasets.png", dpi=170)
    plt.close(fig)


def fig_loss(data, plt):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for tr, col in zip(data["trainings"], (C_R1R2, C_R2R3)):
        if not tr["curve"]:
            continue
        xs = [p["step"] for p in tr["curve"]]
        ys = [p["loss"] for p in tr["curve"]]
        lab = f"{tr['label']} (batch {tr['global_batch']})"
        a1.plot(xs, ys, lw=1.2, color=col, label=lab)
        a2.plot(xs, ys, lw=1.2, color=col, label=lab)
    a1.set_yscale("log")
    a1.set_xlabel("training step")
    a1.set_ylabel("flow-matching loss (log)")
    a1.set_title("Full range", fontsize=10)
    a1.grid(alpha=0.3)
    a1.legend(fontsize=8)

    a2.set_ylim(0.008, 0.030)
    a2.set_xlabel("training step")
    a2.set_ylabel("loss")
    a2.set_title("Converged region (linear)", fontsize=10)
    a2.grid(alpha=0.3)
    a2.legend(fontsize=8)

    fig.suptitle("Both runs converge; the loss stops being informative after ~3k steps", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_loss.png", dpi=170)
    plt.close(fig)


def fig_checkpoints(data, plt):
    """Success against training step, with the intervals that make it un-interpretable."""
    r12 = [e for e in data["evals"] if e.get("training_step") and e["model"].startswith("ft_r1r2")]
    r23 = [e for e in data["evals"] if e.get("training_step") and e["model"].startswith("ft_r2r3")]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for rows, col, lab in ((r12, C_R1R2, "fine-tune on r1r2"), (r23, C_R2R3, "fine-tune on r2+r3")):
        rows = sorted(rows, key=lambda e: e["training_step"])
        xs = [e["training_step"] for e in rows]
        ps, lo, hi = [], [], []
        for e in rows:
            p, l, h = wilson(e["successes"], e["n"])
            ps.append(p)
            lo.append(p - l)
            hi.append(h - p)
        ax.errorbar(xs, ps, yerr=[lo, hi], marker="o", ms=5, lw=1.5, capsize=4,
                    color=col, label=lab)
        for x, p, e in zip(xs, ps, rows):
            ax.annotate(f"{e['successes']}/{e['n']}", (x, p), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=7.5, color=col)
    ax.set_xlabel("training step")
    ax.set_ylabel("success rate (spawn jitter +/-5 cm)")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Success rate is flat across checkpoints and across datasets\n"
                 "every 95% interval overlaps every other", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_checkpoints.png", dpi=170)
    plt.close(fig)


def _load_spawn(data):
    """Bin the recorded spawn positions onto the analysis grid.

    Returns the per-cell success grid (with counts), the per-cell demo grid, and the raw points,
    all in the axis convention of spawn_analysis: rows are x bands (reach away from the robot),
    columns are y bands (away from the plate).
    """
    sa = data["spawn_analysis"]
    edges = sa["cell_edges_cm"]
    g = sa["grid"]

    ev = np.genfromtxt(RESULTS / "spawn_eval.csv", delimiter=",", names=True)
    de = np.genfromtxt(RESULTS / "spawn_demos.csv", delimiter=",", names=True)
    ev_xy = np.column_stack([ev["x_off_cm"], ev["y_off_cm"]])
    ev_ok = ev["success"].astype(bool)
    de_xy = np.column_stack([de["x_off_cm"], de["y_off_cm"]])

    # Tight bounds: episodes outside the +/-5 cm generation box are excluded rather than folded
    # into the edge cells, so every cell describes the region it is drawn over. (The round-3
    # allocation in plan_targeted.py widens them instead, which is why its per-cell episode counts
    # differ slightly from the ones shown here.)
    b = list(edges)

    rate = np.full((g, g), np.nan)
    counts = np.empty((g, g), dtype=object)
    demos = np.zeros((g, g))
    for i in range(g):
        for j in range(g):
            sel = ((ev_xy[:, 0] >= b[i]) & (ev_xy[:, 0] < b[i + 1])
                   & (ev_xy[:, 1] >= b[j]) & (ev_xy[:, 1] < b[j + 1]))
            n, w = int(sel.sum()), int(ev_ok[sel].sum())
            # Three episodes is too few to plot a rate from; leave the cell blank instead.
            if n >= 3:
                rate[i, j] = w / n
            counts[i, j] = f"{w}/{n}" if n else ""
            demos[i, j] = int(((de_xy[:, 0] >= b[i]) & (de_xy[:, 0] < b[i + 1])
                               & (de_xy[:, 1] >= b[j]) & (de_xy[:, 1] < b[j + 1])).sum())
    return sa, edges, g, rate, counts, demos, ev_xy, ev_ok, de_xy


def fig_spawn_outcome(data, plt):
    """Where the apple spawned and whether the episode succeeded.

    The left panel is every evaluation episode; the right collapses it onto the axis that carries
    almost all of the structure, distance from the plate.
    """
    sa, edges, g, _, _, _, ev_xy, ev_ok, de_xy = _load_spawn(data)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.9, 3.35))

    # Training-demo cloud underneath, so coverage and outcome are visible in one view.
    a1.scatter(de_xy[:, 1], de_xy[:, 0], s=3, c="#c9d9e8", alpha=0.45,
               label=f"training demos (n={len(de_xy)})", zorder=1)
    a1.scatter(ev_xy[~ev_ok, 1], ev_xy[~ev_ok, 0], s=46, marker="x", c=C_FROZEN, lw=1.7,
               label=f"fail ({int((~ev_ok).sum())})", zorder=3)
    a1.scatter(ev_xy[ev_ok, 1], ev_xy[ev_ok, 0], s=40, marker="o", c=C_R2R3,
               edgecolors="white", lw=0.6, label=f"success ({int(ev_ok.sum())})", zorder=4)
    a1.add_patch(plt.Rectangle((edges[0], edges[0]), edges[-1] - edges[0], edges[-1] - edges[0],
                               fill=False, ls="--", lw=1.2, ec="#333", zorder=5))
    a1.text(edges[-1] + 0.3, edges[0] - 1.1, "+/-5 cm generation box", fontsize=7,
            ha="right", va="top", color="#333",
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))
    a1.set_xlabel("y offset (cm)  ->  away from the plate")
    a1.set_ylabel("x offset (cm)  ->  longer reach")
    a1.set_title("Apple spawn vs outcome, 99 rollouts", fontsize=10)
    a1.legend(fontsize=7.5, loc="upper left", framealpha=0.92)
    a1.grid(alpha=0.3)
    a1.set_aspect("equal")

    # Marginal along y, in 2.5 cm bands, restricted to the generation box for the same reason the
    # grid is: outside it there is no training data, so a rate there measures extrapolation.
    inbox = (np.abs(ev_xy[:, 0]) <= edges[-1]) & (np.abs(ev_xy[:, 1]) <= edges[-1])
    bins = np.linspace(edges[0], edges[-1], 5)
    idx = np.digitize(ev_xy[inbox, 1], bins) - 1
    ok_in = ev_ok[inbox]
    centres, rates, ns, los, his = [], [], [], [], []
    for k in range(len(bins) - 1):
        sel = idx == k
        n = int(sel.sum())
        if n == 0:
            continue
        w = int(ok_in[sel].sum())
        p, lo, hi = wilson(w, n)
        centres.append((bins[k] + bins[k + 1]) / 2)
        rates.append(p)
        ns.append(n)
        los.append(p - lo)
        his.append(hi - p)
    a2.bar(centres, rates, width=2.1, color=C_R1R2, edgecolor=EDGE, lw=0.8, zorder=3)
    a2.errorbar(centres, rates, yerr=[los, his], fmt="none", ecolor=EDGE, capsize=4, lw=1.1,
                zorder=4)
    for c, r, n, h in zip(centres, rates, ns, his):
        a2.text(c, r + h + 0.03, f"n={n}", ha="center", fontsize=7.5)
    mean = sa["eval_successes"] / sa["eval_episodes"]
    a2.axhline(mean, ls="--", lw=1, color="#333", zorder=2)
    a2.text(bins[-1] - 0.15, mean + 0.02, f"run mean {mean:.2f}", fontsize=7.5, ha="right",
            color="#333")
    a2.set_xticks(bins)
    a2.set_xticklabels([f"{v:+.1f}" for v in bins])
    a2.set_xlabel("y offset (cm)  ->  away from the plate")
    a2.set_ylabel("success rate")
    a2.set_ylim(0, 1.08)
    a2.set_title("Success falls off with distance from the plate", fontsize=10)
    a2.grid(axis="y", alpha=0.3, zorder=0)

    fig.suptitle("The fine-tuned policy's failures are positional, not random", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGURES / "fig9_spawn_outcome.png", dpi=170)
    plt.close(fig)


def fig_spawn_maps(data, plt):
    """Three quantities over the same spawn box: success, the model's own offline error, coverage.

    The comparison is the whole point. Success varies six-fold; the offline action error the model
    is actually trained to minimise is flat; and the demonstration coverage is flat too.
    """
    sa, edges, g, rate, counts, demos, *_ = _load_spawn(data)
    mse = np.array([[np.nan if v is None else v for v in row]
                    for row in sa["offline_action_mse"]["grid"]], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.15))
    panels = [
        (rate, "sim SUCCESS RATE\n(99 rollouts -- what we care about)", CMAP_RATE, "{:.2f}", counts),
        (mse, "offline ACTION ERROR (MSE x1000)\n(400 episodes, exact spawns)", CMAP_ERROR,
         "{:.2f}", None),
        (demos, f"TRAINING DEMOS per cell\n({sa['training_demos_total']} episodes)", CMAP_COUNT,
         "{:.0f}", None),
    ]
    for ax, (arr, title, cmap, fmt, extra) in zip(axes, panels):
        im = ax.imshow(arr, cmap=cmap, origin="lower",
                       extent=[edges[0], edges[-1], edges[0], edges[-1]])
        for i in range(g):
            for j in range(g):
                if np.isnan(arr[i, j]):
                    continue
                cx = (edges[j] + edges[j + 1]) / 2
                cy = (edges[i] + edges[i + 1]) / 2
                ax.text(cx, cy + (0.45 if extra is not None else 0.0), fmt.format(arr[i, j]),
                        ha="center", va="center", fontsize=11, weight="bold", color="#1a1a1a")
                if extra is not None and extra[i, j]:
                    ax.text(cx, cy - 0.95, extra[i, j], ha="center", va="center",
                            fontsize=8, color="#333")
        ax.set_xticks(edges)
        ax.set_yticks(edges)
        ax.set_xticklabels([f"{v:+.1f}" for v in edges], fontsize=8)
        ax.set_yticklabels([f"{v:+.1f}" for v in edges], fontsize=8)
        ax.set_xlabel("y offset (cm)  ->  away from plate", fontsize=8.5)
        ax.set_ylabel("x offset (cm)  ->  longer reach", fontsize=8.5)
        ax.set_title(title, fontsize=9.5)
        ax.grid(color="#666", lw=0.6)
        fig.colorbar(im, ax=ax, fraction=0.046)

    oa = sa["offline_action_mse"]
    spread_s = np.nanmax(rate) / max(np.nanmin(rate), 0.01)
    fig.suptitle(
        f"Success varies {spread_s:.0f}x across the box, while the model's own offline error varies "
        f"{oa['spread_ratio']}x and demo coverage {demos.max()/demos.min():.1f}x", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURES / "fig10_spawn_maps.png", dpi=170)
    plt.close(fig)


def fig_generalisation(data, plt):
    """The limit that round 3 was meant to fix, and why we cannot yet say whether it did."""
    # The fixed-spawn (0 cm) run is deliberately absent: that pose is the one every training
    # demonstration was generated around, so it measures memorisation rather than generalisation.
    rows = [
        ("+/-5 cm\n(the training box)", ev(data, "ft_r1r2_c5000")),
        ("+/-10 cm\n(outside it)", ev(data, "ft_r1r2_c5000_j10")),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    _bar_with_ci(ax, [r[0] for r in rows],
                 [(r[1]["successes"], r[1]["n"]) for r in rows],
                 [C_R1R2, C_FROZEN])
    ax.set_xlabel("apple spawn jitter at evaluation")
    ax.set_title("The r1r2 policy does not generalise past its training spawn box\n"
                 "(fine-tuned checkpoint-5000)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_generalisation.png", dpi=170)
    plt.close(fig)


def fig_inference(data, plt):
    """Execution-strategy ablation: none of it moves the number or produces a retry."""
    var = data["inference_ablation"]["variants"]
    fig, ax = plt.subplots(figsize=(9.2, 4.5))
    cols = [C_R2R3 if not v.get("partial") else C_WARN for v in var]
    ps = _bar_with_ci(ax, [v["label"] for v in var],
                      [(v["successes"], v["n"]) for v in var], cols)
    base_p = var[0]["successes"] / var[0]["n"]
    ax.axhline(base_p, ls="--", lw=1, color="#333", zorder=2)
    ax.text(len(var) - 0.45, base_p + 0.02, "stock", fontsize=8, ha="right")
    for x, v in enumerate(var):
        ax.text(x, 0.04, "no retry\nobserved", ha="center", fontsize=7.5, color="white",
                weight="bold")
    ax.set_title("Changing how the action chunk is executed does not create recovery behaviour\n"
                 "(same checkpoint-8000; amber bars were stopped early)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig6_inference_ablation.png", dpi=170)
    plt.close(fig)


def fig_failure(data, plt):
    """What failure looks like: the episode either finishes early or runs out the clock."""
    fa = data["failure_analysis"]
    eps = fa["episodes"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.8, 4.1))

    xs = range(1, len(eps) + 1)
    secs = [e["frames"] / 50.0 for e in eps]
    cols = [C_R2R3 if e["success"] else C_FROZEN for e in eps]
    a1.bar(xs, secs, color=cols, width=0.66, zorder=3)
    a1.axhline(14.0, ls="--", lw=1, color="#333", zorder=4)
    a1.text(len(eps) + 0.2, 14.0, " timeout", fontsize=8, va="center")
    a1.set_xlabel("episode")
    a1.set_ylabel("duration (s)")
    a1.set_xticks(list(xs))
    a1.set_ylim(0, 16)
    a1.grid(axis="y", alpha=0.3, zorder=0)
    a1.set_title("Every failure runs the full clock", fontsize=10)

    n_to = sum(1 for e in eps if not e["success"] and e["frames"] >= 700)
    n_fail = sum(1 for e in eps if not e["success"])
    n_ok = len(eps) - n_fail
    a2.bar([0, 1, 2], [n_ok, n_to, n_fail - n_to],
           color=[C_R2R3, C_FROZEN, C_WARN], width=0.6, zorder=3)
    for x, v in zip([0, 1, 2], [n_ok, n_to, n_fail - n_to]):
        a2.text(x, v + 0.12, str(v), ha="center", fontsize=10)
    a2.set_xticks([0, 1, 2])
    a2.set_xticklabels(["success", "failure by\ntimeout", "failure by\nearly termination"],
                       fontsize=9)
    a2.set_ylabel("episodes")
    a2.grid(axis="y", alpha=0.3, zorder=0)
    a2.set_title("No episode ends early in failure", fontsize=10)

    fig.suptitle(f"Failure mode on the final checkpoint "
                 f"(object-moved rate {fa['object_moved_rate']:.2f})", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig7_failure_mode.png", dpi=170)
    plt.close(fig)


def fig_regrasp(data, plt):
    """The single most decisive measurement: the data contains no retries to imitate."""
    ra = data["regrasp_audit"]
    hist = {int(k): v for k, v in ra["close_events_histogram"].items()}
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    keys = [0, 1, 2, 3]
    vals = [hist.get(k, 0) for k in keys]
    ax.bar(keys, vals, color=[C_REF, C_R2R3, C_FROZEN, C_FROZEN], width=0.6, zorder=3)
    for k, v in zip(keys, vals):
        ax.text(k, v + 30, f"{v:,}", ha="center", fontsize=10)
    ax.set_xticks(keys)
    ax.set_xlabel("hand-close events in the episode")
    ax.set_ylabel("episodes")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_title(f"All {ra['episodes_scanned']:,} training episodes close the hand exactly once\n"
                 "a re-grasp would need two or more, so recovery is not in the data", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig8_regrasp_audit.png", dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------- summary


def summary(data) -> str:
    L: list[str] = []
    add = L.append
    add("=" * 78)
    add("GR00T N1.7 fine-tuning -- derived numbers")
    add("=" * 78)

    add("\nDatasets")
    for d in data["datasets"]:
        add(f"  {d['id']:6s} {d['episodes']:>5,} eps  {d['frames']:>9,} frames   "
            f"re-grasp episodes: {d['regrasp_episodes']}")

    add("\nTraining")
    for t in data["trainings"]:
        add(f"  {t['id']:9s} base={t['base_model'][:34]:34s} batch={t['global_batch']} "
            f"steps={t['max_steps']} ~{t['epochs']:.2f} epochs")
        if t["curve"]:
            add(f"             loss {t['curve'][0]['loss']:.4f} -> {t['curve'][-1]['loss']:.4f}")

    add("\nEvaluations (success rate, 95% Wilson interval)")
    for e in data["evals"]:
        p, lo, hi = wilson(e["successes"], e["n"])
        lab = e["label"].replace("\n", " ")
        add(f"  jitter {e['jitter_m']:.2f} m  {e['successes']:>3}/{e['n']:<4} = {p:.3f} "
            f"[{lo:.2f},{hi:.2f}]  {lab}")

    add("\nKey comparisons")
    fr = ev(data, "frozen_revo2")
    ft = ev(data, "ft_r1r2_c5000")
    dex = ev(data, "frozen_dex3")
    dex_j = ev(data, "frozen_dex3_j05")
    add("  HEADLINE (matched +/-5 cm protocol): Dex3 frozen vs Revo2 fine-tuned.")
    p = two_proportion_p(ft["successes"], ft["n"], dex_j["successes"], dex_j["n"])
    add(f"  fine-tuned (+/-5cm) vs Dex3 (+/-5cm):      "
        f"{ft['successes']}/{ft['n']} vs {dex_j['successes']}/{dex_j['n']}  "
        f"= {ft['successes']/ft['n']:.2f} vs {dex_j['successes']/dex_j['n']:.2f}  p={p:.3f}")
    add("  Retarget-only gap (fixed spawn) and as-shipped Dex3 reference:")
    p = two_proportion_p(ft["successes"], ft["n"], fr["successes"], fr["n"])
    add(f"  fine-tuned (+/-5cm) vs frozen Revo2 (fixed): "
        f"{ft['successes']}/{ft['n']} vs {fr['successes']}/{fr['n']}  p={p:.2e}")
    p = two_proportion_p(ft["successes"], ft["n"], dex["successes"], dex["n"])
    add(f"  fine-tuned (+/-5cm) vs frozen Dex3 (fixed):  "
        f"{ft['successes']}/{ft['n']} vs {dex['successes']}/{dex['n']}  p={p:.3f}")
    add("  (Fixed-spawn fine-tuned run is excluded -- that pose is its own training point.)")

    a = ev(data, "ft_r1r2_c5000")
    b = ev(data, "ft_r2r3_c10000")
    p = two_proportion_p(a["successes"], a["n"], b["successes"], b["n"])
    add(f"  r1r2 vs r2+r3 at jitter 5 cm:              "
        f"{a['successes']}/{a['n']} vs {b['successes']}/{b['n']}  p={p:.3f}  <- round 3 shows no gain here")

    hs = data["inference_ablation"]["hard_swap_early_test"]
    add(f"  stock vs shortened chunk (r1r2 c5000):     "
        f"{hs['stock']['successes']}/{hs['stock']['n']} vs "
        f"{hs['chunk10']['successes']}/{hs['chunk10']['n']}  p={hs['two_proportion_p']:.3f}")

    v = data["inference_ablation"]["variants"]
    p = two_proportion_p(v[0]["successes"], v[0]["n"], v[1]["successes"], v[1]["n"])
    add(f"  stock vs temporal ensemble (c8000):        "
        f"{v[0]['successes']}/{v[0]['n']} vs {v[1]['successes']}/{v[1]['n']}  p={p:.3f}")

    fa = data["failure_analysis"]
    nf = sum(1 for e in fa["episodes"] if not e["success"])
    nt = sum(1 for e in fa["episodes"] if not e["success"] and e["frames"] >= 700)
    ok = [e["frames"] / 50 for e in fa["episodes"] if e["success"]]
    sa, edges, g, rate, counts, demos, ev_xy, ev_ok, _ = _load_spawn(data)
    inbox = (np.abs(ev_xy[:, 0]) <= edges[-1]) & (np.abs(ev_xy[:, 1]) <= edges[-1])
    near = inbox & (ev_xy[:, 1] < 0)
    far = inbox & (ev_xy[:, 1] >= 0)
    nw, nn = int(ev_ok[near].sum()), int(near.sum())
    fw, fn = int(ev_ok[far].sum()), int(far.sum())
    add("\nSpawn position vs success (ft_r1r2 checkpoint-5000, inside the +/-5 cm box)")
    add(f"  near half of the box (y<0):  {nw}/{nn} = {nw/nn:.2f}")
    add(f"  far half of the box  (y>=0): {fw}/{fn} = {fw/fn:.2f}")
    add(f"  near vs far:                 p={two_proportion_p(nw, nn, fw, fn):.4f}")
    add(f"  per-cell success spread:     {np.nanmin(rate):.2f} to {np.nanmax(rate):.2f}  "
        f"({np.nanmax(rate)/max(np.nanmin(rate), 0.01):.1f}x)")
    oa = sa["offline_action_mse"]
    add(f"  per-cell offline MSE spread: {oa['spread_ratio']}x  "
        f"(cell-wise corr with success {oa['cellwise_corr_with_sim_success']:+.2f})")
    add(f"  per-cell demo spread:        {demos.min():.0f} to {demos.max():.0f}  "
        f"({demos.max()/demos.min():.2f}x)  -- coverage is uniform, success is not")

    add(f"\nFailure mode (final checkpoint): {nt}/{nf} failures are timeouts; "
        f"successes take {min(ok):.1f}-{max(ok):.1f} s")
    add("=" * 78)
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stats", action="store_true", help="print the summary without plotting")
    args = ap.parse_args()

    data = load()
    print(summary(data))

    if args.stats:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(exist_ok=True)
    for fn in (fig_headline, fig_datasets, fig_loss, fig_checkpoints,
               fig_spawn_outcome, fig_spawn_maps,
               fig_generalisation, fig_inference, fig_failure, fig_regrasp):
        fn(data, plt)
    made = sorted(p.name for p in FIGURES.glob("*.png"))
    print(f"\nwrote {len(made)} figures to {FIGURES.relative_to(HERE.parent)}/:")
    for m in made:
        print(f"  {m}")


if __name__ == "__main__":
    main()
