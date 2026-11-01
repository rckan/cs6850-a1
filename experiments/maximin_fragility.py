"""Maximin experiment C: more communities at a fixed budget (fragility).

Same graph, rumor seeds, biases and worlds as the other experiments; only the
partition changes. Louvain at increasing resolution splits the network into
more, smaller communities (as FWRRS uses Louvain clusters). Acceptance still
comes from the H/L bias model. With more infected communities than seeds, the
hard minimum M can in principle be 0 for every seed set; the soft minimum and
worst-first greedy are compared with plain greedy on M.

The candidate pool is the usual top-degree H/L pool plus each community's
highest-degree node, so no community is unseedable just because it has no
candidates.
"""

import json
import sys

import matplotlib.pyplot as plt

from experiments.common import OUTCOMES, RESULTS, save_json, save_table, setup
from experiments.maximin_common import MAXIMIN_CONFIG, louvain_labels, measure, relabel, seed_sets
from experiments.maximin_plotting import line, save
from fairblock import seeding

MU_LOW = 0.3
K = 10
RESOLUTIONS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0]
METHODS = ("Total greedy", "Worst-first greedy", "Greedy on M", "Soft-min greedy")


def main():
    base = setup(MU_LOW)
    results = {o: {m: [] for m in METHODS} for o in OUTCOMES}
    for res in RESOLUTIONS:
        labels = louvain_labels(base["G"], res)
        train, test = relabel(base["train"], labels), relabel(base["test"], labels)
        extra = [seeding.top_degree(base["G"], [v for v in range(len(labels)) if labels[v] == c], 1, base["S_N"])[0]
                 for c in sorted(set(labels)) if any(labels[v] == c and v not in base["S_N"] for v in range(len(labels)))]
        ctx = dict(base, candidates=list(dict.fromkeys(base["candidates"] + extra)))
        n_comm = len(set(labels))
        for o in OUTCOMES:
            seeds = seed_sets(ctx, [K], o, METHODS, train=train, groups=labels)
            for m in METHODS:
                results[o][m].append(dict(resolution=res, n_comm=n_comm, seeds=seeds[m][K], **measure(test, seeds[m][K], o)))
        print(f"resolution={res}: {n_comm} communities, {results['saved']['Total greedy'][-1]['n_infected']} infected", flush=True)
    save_json("maximin_fragility", dict(config=MAXIMIN_CONFIG, mu_low=MU_LOW, k=K, results=results))
    plot(results)


def plot(results):
    # A "communities left at 0" panel was all zeros (rates are means over worlds and the
    # truth reaches every community a little), so the count stays in the table only.
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.4), sharex=True, gridspec_kw=dict(wspace=0.32))
    for row, o in enumerate(OUTCOMES):
        for m in METHODS:
            rs = results[o][m]
            x = [r["n_infected"] for r in rs]
            line(axes[row, 0], x, [r["min_rate"] for r in rs], m)
            line(axes[row, 1], x, [r["frac"] for r in rs], m)
        axes[row, 0].set_ylabel(f"worst community's {o} rate")
        axes[row, 1].set_ylabel(f"fraction of exposed {o}")
    ticks = [r["n_infected"] for r in results[OUTCOMES[0]][METHODS[0]]]
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
    for ax in axes[1]:
        ax.set_xlabel("infected communities (Louvain, log scale)")
    axes[0, 0].set_title(rf"Maximin value M, k={K}, $\mu_L$={MU_LOW}")
    axes[0, 1].set_title("Total blocked")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    save(fig, "maximin_fragility")

    rows = [(o, m, r["n_comm"], r["n_infected"], r["min_rate"], r["n_zero"], r["frac"])
            for o in OUTCOMES for m in METHODS for r in results[o][m]]
    save_table("maximin_fragility", ["outcome", "method", "communities", "infected", "M", "infected at 0", "frac"], rows)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot(json.loads((RESULTS / "maximin_fragility.json").read_text())["results"])
    else:
        main()
