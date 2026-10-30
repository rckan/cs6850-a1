"""Experiment B: sweep the budget k at fixed mu_L (tests Claim 2, leveling down).

Ceiling model prediction: once the low group's converted rate saturates near
its mean acceptance, DP greedy's total stops rising. The ceiling only holds for
the converted rate; the saved rate shows shielding. Right panels compare DP
greedy's per-group rates with total greedy's.
"""

import json
import sys

import matplotlib.pyplot as plt

from experiments.common import CONFIG, METHODS, OUTCOMES, RESULTS, measure, save_json, save_table, seed_sets, setup
from experiments.plotting import MUTED, line, save

MU_LOW = 0.3
KS = list(range(1, 31))


def main():
    ctx = setup(MU_LOW)
    results = {}
    for o in OUTCOMES:
        seeds = seed_sets(ctx, KS, o)
        results[o] = {m: [dict(k=k, **measure(ctx["test"], seeds[m][k], o)) for k in KS] for m in METHODS}
        print(f"{o} done", flush=True)
    save_json("parity_sweep_budget", dict(config=CONFIG, mu_low=MU_LOW, a_mean=ctx["a_mean"], results=results))
    plot(results, ctx["a_mean"]["L"])


def plot(results, mean_a_low):
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.4), sharex=True)
    for row, o in enumerate(OUTCOMES):
        for m in METHODS:
            line(axes[row, 0], KS, [r["frac"] for r in results[o][m]], m, markevery=3)
        for m in ("DP greedy", "Total greedy"):
            line(axes[row, 1], KS, [r["rates"]["H"] for r in results[o][m]], m, markevery=3)
            line(axes[row, 1], KS, [r["rates"]["L"] for r in results[o][m]], m, markevery=3,
                 linestyle="--", label=f"{m} (L)")
        axes[row, 0].set_ylabel(f"fraction of exposed {o}")
        axes[row, 1].set_ylabel(f"group {o} rate")
    # Converted-rate ceiling (plus seeds' own share); does not apply to saved.
    axes[1, 1].axhline(mean_a_low, color=MUTED, linewidth=0.8, linestyle=":")
    axes[1, 1].annotate(f"mean $a_v$ in L = {mean_a_low:.2f}", (KS[0], mean_a_low), xytext=(0, 3),
                        textcoords="offset points", ha="left", color=MUTED, fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("budget k")
    axes[0, 0].set_title(rf"Total blocked, $\mu_L$={MU_LOW}")
    axes[0, 1].set_title("Group rates, DP vs total greedy (solid H, dashed L)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.03))
    save(fig, "parity_sweep_budget")

    rows = [(o, m, r["k"], r["frac"], r["dp_gap"], r["rates"]["H"], r["rates"]["L"])
            for o in OUTCOMES for m in METHODS for r in results[o][m] if r["k"] % 5 == 0 or r["k"] == 1]
    save_table("parity_sweep_budget", ["outcome", "method", "k", "frac", "dp_gap", "r_H", "r_L"], rows)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        data = json.loads((RESULTS / "parity_sweep_budget.json").read_text())
        plot(data["results"], data["a_mean"]["L"])
    else:
        main()
