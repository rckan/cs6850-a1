"""Homophily sweep (roadmap goal 1): does saved-node fairness stop being free as
the two groups become more separated?

Shielding (truth seeds in H cutting rumor paths into L) needs edges between the
groups. We vary ratio = p_out / p_in while holding the expected mean degree at
the exploratory network's value, so only the split of edges between and within
groups changes, not the density. ratio = 0.2 is the exploratory SBM
(p_in = 0.05, p_out = 0.01). Each (ratio, draw) is an independent network,
rumor seeds, acceptance noise and set of worlds; results are means over draws
with 95% t confidence intervals.

The worst-group rate and gap here are over infected groups only: under strong
homophily the rumor sometimes never reaches L, and that group then has no rate.

Run:  python -m experiments.sweep_homophily [--draws 20] [--processes 8]
      python -m experiments.sweep_homophily --plot-only
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from experiments.common import FIGURES, OUTCOMES, RESULTS, save_json, save_table, setup
from experiments.maximin_common import MAXIMIN_CONFIG, measure, seed_sets
from experiments.maximin_plotting import STYLE, save
from experiments.plotting import line_ci
from experiments.runner import mean_ci, run_tasks
from fairblock import graphs, metrics

NAME = "sweep_homophily"
RATIOS = [0.02, 0.05, 0.1, 0.2, 0.4, 1.0]  # p_out / p_in; 1.0 = no community structure
MU_LOW = 0.3
K = 10
METHODS = ("Total greedy", "DP greedy", "Worst-first greedy", "Soft-min greedy")
FAIR = METHODS[1:]


def mean_degree(cfg):
    """Expected mean degree of the exploratory SBM in cfg."""
    nh, nl = cfg["n_high"], cfg["n_low"]
    within = nh * (nh - 1) / 2 + nl * (nl - 1) / 2
    return 2 * (cfg["p_in"] * within + cfg["p_out"] * nh * nl) / (nh + nl)


def config(ratio):
    cfg = dict(MAXIMIN_CONFIG)
    cfg["p_in"], cfg["p_out"] = graphs.sbm_probs(cfg["n_high"], cfg["n_low"], ratio, mean_degree(MAXIMIN_CONFIG))
    return cfg


def gap(rates, exposed):
    """Max - min rate over infected groups (0 if fewer than two are infected)."""
    r = [rates[g] for g in metrics.infected(exposed)]
    return max(r) - min(r) if r else 0.0


def one_task(ratio, draw):
    """Every method under both outcomes on one network. Runs in a worker process."""
    cfg = config(ratio)
    ctx = setup(MU_LOW, cfg, draw=draw)
    G, groups = ctx["G"], ctx["groups"]
    cross = sum(groups[u] != groups[v] for u, v in G.edges) / max(G.number_of_edges(), 1)
    out = dict(ratio=ratio, draw=draw, p_in=cfg["p_in"], p_out=cfg["p_out"], cross_edges=float(cross),
               exposed=ctx["test"].exposed, results={})
    for o in OUTCOMES:
        seeds = seed_sets(ctx, [K], o, METHODS, cfg)
        res = {}
        for m in METHODS:
            s = seeds[m][K]
            r = measure(ctx["test"], s, o)
            res[m] = dict(frac=r["frac"], min_rate=r["min_rate"], rates=r["rates"], n_infected=r["n_infected"],
                          gap=gap(r["rates"], ctx["test"].exposed), n_low=int((groups[s] == "L").sum()), seeds=s)
        out["results"][o] = res
    return out


def summarize(rows):
    """{outcome: {method: {stat: [(mean, half) per ratio]}}} plus per-ratio network stats."""
    summary = {o: {m: {} for m in METHODS} for o in OUTCOMES}
    for o in OUTCOMES:
        for m in METHODS:
            for stat in ("pof", "min_rate", "gap", "frac", "r_L", "n_low"):
                summary[o][m][stat] = []
                for ratio in RATIOS:
                    vals = []
                    for row in rows:
                        if row["ratio"] != ratio:
                            continue
                        r, best = row["results"][o][m], row["results"][o]["Total greedy"]["frac"]
                        if stat == "pof":
                            vals.append(best / r["frac"] if r["frac"] > 0 else np.nan)
                        elif stat == "r_L":
                            vals.append(r["rates"]["L"] if row["exposed"]["L"] > 0 else np.nan)
                        else:
                            vals.append(r[stat])
                    summary[o][m][stat].append(mean_ci(vals))
    net = []
    for ratio in RATIOS:
        sel = [r for r in rows if r["ratio"] == ratio]
        net.append(dict(
            ratio=ratio,
            cross_edges=float(np.mean([r["cross_edges"] for r in sel])),
            L_infected=float(np.mean([r["exposed"]["L"] > 0 for r in sel])),
            w_L=float(np.mean([r["exposed"]["L"] / sum(r["exposed"].values()) for r in sel])),
        ))
    return summary, net


def plot(rows):
    summary, net = summarize(rows)
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.6), sharex=True, gridspec_kw=dict(wspace=0.38))
    for row, o in enumerate(OUTCOMES):
        for m in METHODS:
            s = summary[o][m]
            if m in FAIR:
                line_ci(axes[row, 0], RATIOS, *zip(*s["pof"]), STYLE[m], m)
            line_ci(axes[row, 1], RATIOS, *zip(*s["min_rate"]), STYLE[m], m)
            line_ci(axes[row, 2], RATIOS, *zip(*s["n_low"]), STYLE[m], m)
        axes[row, 0].axhline(1, color="#52514e", linewidth=0.8, linestyle=":")
        axes[row, 0].set_ylabel(f"price of fairness ({o})")
        axes[row, 1].set_ylabel(f"worst group's {o} rate")
        axes[row, 2].set_ylabel("seeds in L")
    for ax in axes[1]:
        ax.set_xscale("log")
        ax.set_xticks(RATIOS, [f"{r:g}" for r in RATIOS])
        ax.minorticks_off()
        ax.set_xlabel(r"$p_{out}/p_{in}$ (left = more homophily)")
    n_draws = len({r["draw"] for r in rows})
    axes[0, 0].set_title("Price of fairness (vs. total greedy)")
    axes[0, 1].set_title("Worst-group rate")
    axes[0, 2].set_title(f"Seeds in L (of {K})")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(METHODS), frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.text(0.5, -0.01, rf"$\mu_L$={MU_LOW}, k={K}, mean degree fixed; {n_draws} draws, bands = 95% CI",
             ha="center", fontsize=7, color="#52514e")
    save(fig, NAME)

    def fmt(c):
        mean, half = c
        return f"{mean:.3f} ± {half:.3f}" if np.isfinite(half) else f"{mean:.3f}"

    table = [(o, m, ratio, *(fmt(summary[o][m][st][i]) for st in ("pof", "frac", "min_rate", "gap", "r_L", "n_low")))
             for o in OUTCOMES for m in METHODS for i, ratio in enumerate(RATIOS)]
    save_table(NAME, ["outcome", "method", "p_out/p_in", "PoF", "frac", "worst rate", "gap", "r_L", "seeds in L"], table)
    with open(FIGURES / f"{NAME}.md", "a") as f:
        f.write("\n| p_out/p_in | cross-edge share | draws with L infected | L exposure share |\n|---|---|---|---|\n")
        for n in net:
            f.write(f"| {n['ratio']:g} | {n['cross_edges']:.3f} | {n['L_infected']:.2f} | {n['w_L']:.3f} |\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if args.plot_only:
        rows = json.loads((RESULTS / f"{NAME}.json").read_text())["rows"]
    else:
        tasks = [(ratio, d) for ratio in RATIOS for d in range(args.draws)]
        rows = run_tasks(one_task, tasks, processes=args.processes)
        save_json(NAME, dict(config=MAXIMIN_CONFIG, ratios=RATIOS, mu_low=MU_LOW, k=K, methods=METHODS, rows=rows))
    plot(rows)


if __name__ == "__main__":
    main()
