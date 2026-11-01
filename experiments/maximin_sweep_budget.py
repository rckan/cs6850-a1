"""Maximin experiment B: sweep the budget k at fixed mu_L (tests Claim 2).

Ceiling-model prediction: lexicographic maximin's price of fairness is
non-increasing in k, while plain maximin (random tie-break) and DP can level
down. Left panels: the maximin value M. Right panels: PoF = total greedy's
total / the method's total.
"""

import json
import sys

import matplotlib.pyplot as plt

from experiments.common import OUTCOMES, RESULTS, save_json, save_table, setup
from experiments.maximin_common import MAXIMIN_CONFIG, measure, seed_sets
from experiments.maximin_plotting import MUTED, line, save

MU_LOW = 0.3
KS = list(range(1, 31))
METHODS = ("Total greedy", "DP greedy", "Worst-first greedy", "Worst-first, random ties", "Analytic E/a")


def main():
    ctx = setup(MU_LOW)
    results = {}
    for o in OUTCOMES:
        seeds = seed_sets(ctx, KS, o, METHODS)
        results[o] = {m: [dict(k=k, n_low=int((ctx["groups"][seeds[m][k]] == "L").sum()),
                               **measure(ctx["test"], seeds[m][k], o)) for k in KS] for m in METHODS}
        best = [r["frac"] for r in results[o]["Total greedy"]]
        for m in METHODS:
            for r, b in zip(results[o][m], best):
                r["pof"] = b / r["frac"] if r["frac"] > 0 else float("nan")
        print(f"{o} done", flush=True)
    save_json("maximin_sweep_budget", dict(config=MAXIMIN_CONFIG, mu_low=MU_LOW, a_mean=ctx["a_mean"], results=results))
    plot(results)


def plot(results):
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.4), sharex=True)
    for row, o in enumerate(OUTCOMES):
        for m in METHODS:
            line(axes[row, 0], KS, [r["min_rate"] for r in results[o][m]], m, markevery=3)
            if m != "Total greedy":
                line(axes[row, 1], KS, [r["pof"] for r in results[o][m]], m, markevery=3)
        axes[row, 1].axhline(1.0, color=MUTED, linewidth=0.8, linestyle=":")
        axes[row, 0].set_ylabel(f"worst group's {o} rate")
        axes[row, 1].set_ylabel(f"PoF ({o})")
    for ax in axes[1]:
        ax.set_xlabel("budget k")
    axes[0, 0].set_title(rf"Maximin value M, $\mu_L$={MU_LOW}")
    axes[0, 1].set_title("Price of fairness vs total greedy")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.03))
    save(fig, "maximin_sweep_budget")

    rows = [(o, m, r["k"], r["min_rate"], r["frac"], r["pof"], r["rates"]["H"], r["rates"]["L"], r["n_low"])
            for o in OUTCOMES for m in METHODS for r in results[o][m] if r["k"] % 5 == 0 or r["k"] == 1]
    save_table("maximin_sweep_budget", ["outcome", "method", "k", "M", "frac", "PoF", "r_H", "r_L", "seeds in L"], rows)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot(json.loads((RESULTS / "maximin_sweep_budget.json").read_text())["results"])
    else:
        main()
