"""Experiment A: sweep the low group's mean acceptance mu_L at a fixed budget k.

Plots total blocked fraction and DP gap against mu_L for each method, under both
outcome definitions, and compares the measured price of fairness with the
ceiling-model formula PoF = w_H + r w_L (r = realized mean a_H / mean a_L,
w_g = exposure share).
"""

import json
import sys

import matplotlib.pyplot as plt

from experiments.common import CONFIG, METHODS, OUTCOMES, RESULTS, measure, save_json, save_table, seed_sets, setup
from experiments.plotting import line, save

MU_LOW = [0.8, 0.6, 0.45, 0.3, 0.15]  # mu_H = 0.8: first point is equal means
K = 10


def main():
    results = {o: {m: [] for m in METHODS} for o in OUTCOMES}
    pof = {o: [] for o in OUTCOMES}
    for mu_low in MU_LOW:
        ctx = setup(mu_low)
        for o in OUTCOMES:
            seeds = seed_sets(ctx, [K], o)
            for m in METHODS:
                results[o][m].append(dict(mu_low=mu_low, a_mean=ctx["a_mean"], seeds=seeds[m][K], **measure(ctx["test"], seeds[m][K], o)))
            e = ctx["test"].exposed
            w_h, w_l = e["H"] / sum(e.values()), e["L"] / sum(e.values())
            best = results[o]["Total greedy"][-1]["frac"]
            pof[o].append(dict(
                mu_low=mu_low,
                formula=w_h + (ctx["a_mean"]["H"] / ctx["a_mean"]["L"]) * w_l,
                dp=best / results[o]["DP greedy"][-1]["frac"],
                analytic=best / results[o]["Analytic n/a"][-1]["frac"],
            ))
        print(f"mu_L={mu_low} done", flush=True)
    save_json("parity_sweep_acceptance", dict(config=CONFIG, k=K, results=results, pof=pof))
    plot(results, pof)


def plot(results, pof):
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.4), sharex=True)
    for row, o in enumerate(OUTCOMES):
        for m in METHODS:
            line(axes[row, 0], MU_LOW, [r["frac"] for r in results[o][m]], m)
            line(axes[row, 1], MU_LOW, [r["dp_gap"] for r in results[o][m]], m)
        axes[row, 0].set_ylabel(f"fraction of exposed {o}")
        axes[row, 1].set_ylabel(f"DP gap ({o} rate)")
    for ax in axes[1]:
        ax.set_xlabel(r"mean acceptance of low group $\mu_L$ ($\mu_H$ = 0.8)")
    axes[0, 0].set_title(f"Total blocked, k={K}")
    axes[0, 1].set_title("DP gap (max minus min group rate)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.03))
    save(fig, "parity_sweep_acceptance")

    rows = [(o, m, r["mu_low"], r["frac"], r["dp_gap"], r["rates"]["H"], r["rates"]["L"])
            for o in OUTCOMES for m in METHODS for r in results[o][m]]
    save_table("parity_sweep_acceptance", ["outcome", "method", "mu_L", "frac", "dp_gap", "r_H", "r_L"], rows)
    rows = [(o, p["mu_low"], p["formula"], p["dp"], p["analytic"]) for o in OUTCOMES for p in pof[o]]
    save_table("parity_pof", ["outcome", "mu_L", "PoF formula", "PoF DP", "PoF analytic"], rows)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        data = json.loads((RESULTS / "parity_sweep_acceptance.json").read_text())
        plot(data["results"], data["pof"])
    else:
        main()
