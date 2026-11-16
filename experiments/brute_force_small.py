"""Gap to optimal (roadmap goal 4): how far are the greedy methods from the best
seed set, now that rejection breaks submodularity?

On small graphs we enumerate every seed set of size k = 1..4 among all
non-rumor nodes, evaluate each once (saved and converted counts), and keep the
best value of three objectives under each outcome:
  total    total saved/converted                      (target of total greedy)
  maximin  (worst infected group's rate, total), lexicographic
                                                      (target of worst-first and soft-min greedy)
  parity   F_lambda = B/E - lambda * gap, lambda = 2  (target of DP greedy)
Each greedy method runs on the same worlds with the same candidates (every
non-rumor node), so the gap is purely optimization error, not sampling noise.

Graphs: Zachary's karate club (34 members, split by the club they joined;
"Officer" side is L) and 34-node two-group SBMs (24 H / 10 L, p_out/p_in = 0.2,
same mean degree as the club). Each draw is new rumor seeds, acceptance noise
and worlds (and a new SBM). p = 0.3 and 2 rumor seeds make the rumor reach
about 44% of nodes, close to the main network's 43%.

We also get the *true* price of fairness, best total over the total of the best
maximin set, to compare with the greedy-measured PoF used elsewhere.

Run:  python -m experiments.brute_force_small [--draws 10] [--processes 8]
      python -m experiments.brute_force_small --plot-only
"""

import argparse
import itertools
import json

import matplotlib.pyplot as plt
import numpy as np

from experiments.common import FIGURES, OUTCOMES, RESULTS, save_json, save_table, setup
from experiments.maximin_common import MAXIMIN_CONFIG
from experiments.maximin_plotting import STYLE, save
from experiments.plotting import line_ci
from experiments.runner import mean_ci, run_tasks
from fairblock import graphs, metrics, seeding

NAME = "brute_force_small"
KS = [1, 2, 3, 4]
MU_LOW = 0.3
GRAPHS = ("Karate club", "SBM (34 nodes)")
CFG = dict(MAXIMIN_CONFIG, p=0.3, n_rumor=2, R_train=300, R_test=2, per_group=100,
           n_high=24, n_low=10, sbm_ratio=0.2, sbm_degree=78 * 2 / 34)
OBJECTIVES = ("total", "maximin", "parity")
GREEDY = {  # method -> objective it targets
    "Total greedy": "total",
    "Worst-first greedy": "maximin",
    "Soft-min greedy": "maximin",
    "DP greedy": "parity",
}
IDX = {"saved": 0, "converted": 1}


def objective_values(counts, exposed, lam):
    """All three objectives for one seed set's counts under one outcome."""
    total = metrics.total_saved(counts)
    return dict(
        total=total,
        maximin=(metrics.min_rate(counts, exposed), total),
        parity=metrics.parity_objective(counts, exposed, lam),
        min_rate=metrics.min_rate(counts, exposed),
    )


def context(graph, draw):
    cfg = dict(CFG)
    if graph == "Karate club":
        return setup(MU_LOW, cfg, draw=draw, graph=graphs.karate_club())
    cfg["p_in"], cfg["p_out"] = graphs.sbm_probs(cfg["n_high"], cfg["n_low"], cfg["sbm_ratio"], cfg["sbm_degree"])
    return setup(MU_LOW, cfg, draw=draw)


def one_task(graph, draw):
    """Exhaustive optimum and every greedy method on one small instance."""
    ctx = context(graph, draw)
    ev, cands, exposed, lam = ctx["train"], ctx["candidates"], ctx["train"].exposed, CFG["lam"]

    # best[o][obj][k] = (value, seeds); also keep the plain min rate / total of each best set.
    best = {o: {obj: {} for obj in OBJECTIVES} for o in OUTCOMES}
    for k in KS:
        for S in itertools.combinations(cands, k):
            counts = ev.evaluate(list(S))
            for o in OUTCOMES:
                vals = objective_values(counts[IDX[o]], exposed, lam)
                for obj in OBJECTIVES:
                    cur = best[o][obj].get(k)
                    if cur is None or vals[obj] > cur["value"]:
                        best[o][obj][k] = dict(value=vals[obj], seeds=list(S), total=vals["total"], min_rate=vals["min_rate"])

    greedy = {o: {} for o in OUTCOMES}
    K = max(KS)
    for o in OUTCOMES:
        orders = {
            "Total greedy": seeding.greedy(ev, cands, K, lambda s, e: metrics.total_saved(s), o),
            "Worst-first greedy": seeding.worst_first_greedy(ev, cands, K, o, CFG["tol"], "total"),
            "Soft-min greedy": seeding.greedy(ev, cands, K, lambda s, e: metrics.soft_min_rate(s, e, CFG["tau"]), o),
            "DP greedy": seeding.greedy(ev, cands, K, lambda s, e: metrics.parity_objective(s, e, lam), o),
        }
        for m, order in orders.items():
            greedy[o][m] = {}
            for k in KS:
                vals = objective_values(ev.evaluate(order[:k])[IDX[o]], exposed, lam)
                greedy[o][m][k] = dict(value=vals[GREEDY[m]], seeds=order[:k], total=vals["total"], min_rate=vals["min_rate"])

    # JSON keys must be strings.
    fix = lambda d: {o: {a: {str(k): v for k, v in b.items()} for a, b in d[o].items()} for o in d}
    return dict(graph=graph, draw=draw, exposed=exposed, S_N=ctx["S_N"], best=fix(best), greedy=fix(greedy))


def gap_stats(rows, graph, o, m, k):
    """Per-instance optimization gap of method m at budget k, plus whether it was exactly optimal."""
    obj = GREEDY[m]
    gaps, hit = [], []
    for r in rows:
        if r["graph"] != graph:
            continue
        g, b = r["greedy"][o][m][str(k)], r["best"][o][obj][str(k)]
        if obj == "total":
            gaps.append(g["total"] / b["total"] if b["total"] > 0 else 1.0)
            hit.append(g["total"] >= b["total"] - 1e-9)
        elif obj == "maximin":
            gaps.append(g["min_rate"] / b["min_rate"] if b["min_rate"] > 0 else 1.0)
            hit.append(g["min_rate"] >= b["min_rate"] - 1e-9)
        else:  # parity objective can be negative, so report the shortfall OPT - greedy
            gaps.append(b["value"] - g["value"])
            hit.append(g["value"] >= b["value"] - 1e-9)
    return mean_ci(gaps), float(np.mean(hit))


def pof_stats(rows, graph, o, k):
    """True PoF (best total / total of the best maximin set) vs. the greedy-measured one."""
    true, measured = [], []
    for r in rows:
        if r["graph"] != graph:
            continue
        b = r["best"][o]
        true.append(b["total"][str(k)]["total"] / b["maximin"][str(k)]["total"])
        g = r["greedy"][o]
        measured.append(g["Total greedy"][str(k)]["total"] / g["Worst-first greedy"][str(k)]["total"])
    return mean_ci(true), mean_ci(measured)


PANELS = (("total", ("Total greedy",), "total / best"),
          ("maximin", ("Worst-first greedy", "Soft-min greedy"), "worst rate / best"),
          ("parity", ("DP greedy",), r"best $F_\lambda$ $-$ greedy $F_\lambda$"))
GRAPH_LINE = {"Karate club": "-", "SBM (34 nodes)": "--"}


def plot(rows):
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.6), sharex=True, gridspec_kw=dict(wspace=0.38))
    for row, o in enumerate(OUTCOMES):
        for col, (obj, methods, ylabel) in enumerate(PANELS):
            ax = axes[row, col]
            for graph in GRAPHS:
                for m in methods:
                    stats = [gap_stats(rows, graph, o, m, k)[0] for k in KS]
                    style = dict(STYLE[m], linestyle=GRAPH_LINE[graph])
                    if graph != GRAPHS[0]:
                        style["markerfacecolor"] = "white"
                    line_ci(ax, KS, *zip(*stats), style, f"{m}, {graph}")
            ax.axhline(0 if obj == "parity" else 1, color="#52514e", linewidth=0.8, linestyle=":")
            ax.set_ylabel(f"{ylabel} ({o})")
            ax.set_xticks(KS)
    for ax in axes[1]:
        ax.set_xlabel("budget k")
    axes[0, 0].set_title("Total blocking")
    axes[0, 1].set_title("Maximin")
    axes[0, 2].set_title(r"Parity ($\lambda$ = 2)")
    handles, labels = [], []
    for ax in axes[0]:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.09))
    n = len({r["draw"] for r in rows})
    fig.text(0.5, -0.01, f"Greedy vs. exhaustive optimum on the same worlds; {n} instances per graph; "
             "solid = karate club, dashed = SBM; bands = 95% CI", ha="center", fontsize=7, color="#52514e")
    save(fig, NAME)

    def fmt(c):
        mean, half = c
        return f"{mean:.3f} ± {half:.3f}" if np.isfinite(half) else f"{mean:.3f}"

    table = []
    for graph in GRAPHS:
        for o in OUTCOMES:
            for m in GREEDY:
                for k in KS:
                    g, hit = gap_stats(rows, graph, o, m, k)
                    table.append((graph, o, m, k, fmt(g), f"{hit:.2f}"))
    save_table(NAME, ["graph", "outcome", "method", "k", "gap (ratio; parity: shortfall)", "share optimal"], table)
    with open(FIGURES / f"{NAME}.md", "a") as f:
        f.write("\n| graph | outcome | k | true PoF (optimal sets) | PoF from greedy (total / worst-first) |\n|---|---|---|---|---|\n")
        for graph in GRAPHS:
            for o in OUTCOMES:
                for k in KS:
                    t, g = pof_stats(rows, graph, o, k)
                    f.write(f"| {graph} | {o} | {k} | {fmt(t)} | {fmt(g)} |\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if args.plot_only:
        rows = json.loads((RESULTS / f"{NAME}.json").read_text())["rows"]
    else:
        tasks = [(g, d) for g in GRAPHS for d in range(args.draws)]
        rows = run_tasks(one_task, tasks, processes=args.processes)
        save_json(NAME, dict(config=CFG, ks=KS, mu_low=MU_LOW, rows=rows))
    plot(rows)


if __name__ == "__main__":
    main()
