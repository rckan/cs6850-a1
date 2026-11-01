"""Maximin experiment A: sweep the low group's mean acceptance mu_L at a fixed budget k.

Plots the maximin value M (worst group's rate) and the total blocked fraction
against mu_L, under both outcomes, and compares the measured price of fairness
of maximin seeding with Claim 1's formula PoF = w_H + r w_L (r = realized mean
a_H / mean a_L, w_g = exposure share).
"""

import json
import sys

import matplotlib.pyplot as plt

from experiments.common import OUTCOMES, RESULTS, save_json, save_table, setup
from experiments.maximin_common import MAXIMIN_CONFIG, measure, seed_sets
from experiments.maximin_plotting import line, save

MU_LOW = [0.8, 0.6, 0.45, 0.3, 0.15]  # mu_H = 0.8: first point is equal means
K = 10
METHODS = ("Total greedy", "DP greedy", "Worst-first greedy", "Soft-min greedy", "Proportional", "Analytic E/a")


def main():
    results = {o: {m: [] for m in METHODS} for o in OUTCOMES}
    pof = {o: [] for o in OUTCOMES}
    for mu_low in MU_LOW:
        ctx = setup(mu_low)
        for o in OUTCOMES:
            seeds = seed_sets(ctx, [K], o, METHODS)
            for m in METHODS:
                s = seeds[m][K]
                n_low = int((ctx["groups"][s] == "L").sum())
                results[o][m].append(dict(mu_low=mu_low, seeds=s, n_low=n_low, **measure(ctx["test"], s, o)))
            e = ctx["test"].exposed
            w_h, w_l = e["H"] / sum(e.values()), e["L"] / sum(e.values())
            best = results[o]["Total greedy"][-1]["frac"]
            pof[o].append(dict(
                mu_low=mu_low,
                formula=w_h + (ctx["a_mean"]["H"] / ctx["a_mean"]["L"]) * w_l,
                **{m: best / results[o][m][-1]["frac"] for m in ("Worst-first greedy", "Soft-min greedy", "Analytic E/a", "DP greedy")},
            ))
        print(f"mu_L={mu_low} done", flush=True)
    save_json("maximin_sweep_acceptance", dict(config=MAXIMIN_CONFIG, k=K, results=results, pof=pof))
    plot(results, pof)


def plot(results, pof):
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.4), sharex=True)
    for row, o in enumerate(OUTCOMES):
        for m in METHODS:
            line(axes[row, 0], MU_LOW, [r["min_rate"] for r in results[o][m]], m)
            line(axes[row, 1], MU_LOW, [r["frac"] for r in results[o][m]], m)
        axes[row, 0].set_ylabel(f"worst group's {o} rate")
        axes[row, 1].set_ylabel(f"fraction of exposed {o}")
    for ax in axes[1]:
        ax.set_xlabel(r"mean acceptance of low group $\mu_L$ ($\mu_H$ = 0.8)")
    axes[0, 0].set_title(f"Maximin value M, k={K}")
    axes[0, 1].set_title("Total blocked")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.03))
    save(fig, "maximin_sweep_acceptance")

    rows = [(o, m, r["mu_low"], r["min_rate"], r["frac"], r["rates"]["H"], r["rates"]["L"], r["n_low"])
            for o in OUTCOMES for m in METHODS for r in results[o][m]]
    save_table("maximin_sweep_acceptance", ["outcome", "method", "mu_L", "M", "frac", "r_H", "r_L", "seeds in L"], rows)
    rows = [(o, p["mu_low"], p["formula"], p["Worst-first greedy"], p["Soft-min greedy"], p["Analytic E/a"], p["DP greedy"])
            for o in OUTCOMES for p in pof[o]]
    save_table("maximin_pof", ["outcome", "mu_L", "PoF formula", "PoF worst-first", "PoF soft-min", "PoF analytic E/a", "PoF DP"], rows)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        data = json.loads((RESULTS / "maximin_sweep_acceptance.json").read_text())
        plot(data["results"], data["pof"])
    else:
        main()
