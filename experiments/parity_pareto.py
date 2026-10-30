"""Experiment C: sweep the parity weight lambda (total blocked vs DP gap front).

DP greedy maximizes B/E - lambda * dp_gap at fixed k; each lambda gives one
point. Points are not joined: sampling noise means they are not ordered by
lambda. Other methods are single reference points.
"""

import json
import sys

import matplotlib.pyplot as plt

from experiments.common import CONFIG, METHODS, OUTCOMES, RESULTS, measure, save_json, save_table, seed_sets, setup
from experiments.plotting import STYLE, save

LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
MU_LOW = 0.3
K = 10


def main():
    ctx = setup(MU_LOW)
    results = {}
    for o in OUTCOMES:
        front = []
        for lam in LAMBDAS:
            seeds = seed_sets(ctx, [K], o, methods=["DP greedy"], cfg={**CONFIG, "lam": lam})["DP greedy"][K]
            front.append(dict(lam=lam, seeds=seeds, **measure(ctx["test"], seeds, o)))
        refs = seed_sets(ctx, [K], o, methods=[m for m in METHODS if m != "DP greedy"])
        results[o] = dict(front=front, refs={m: measure(ctx["test"], s[K], o) for m, s in refs.items()})
        print(f"{o} done", flush=True)
    save_json("parity_pareto", dict(config=CONFIG, mu_low=MU_LOW, k=K, results=results))
    plot(results)


def merged_labels(front, tol=0.05):
    """One label per cluster of nearly identical points (within tol of each axis range)."""
    xs, ys = [r["dp_gap"] for r in front], [r["frac"] for r in front]
    dx, dy = (max(xs) - min(xs)) or 1, (max(ys) - min(ys)) or 1
    clusters = []
    for r in front:
        for c in clusters:
            if abs(r["dp_gap"] - c[0]["dp_gap"]) < tol * dx and abs(r["frac"] - c[0]["frac"]) < tol * dy:
                c.append(r)
                break
        else:
            clusters.append([r])
    return [(c[0]["dp_gap"], c[0]["frac"], "λ=" + ",".join(f"{r['lam']:g}" for r in c)) for c in clusters]


def plot(results):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), gridspec_kw=dict(wspace=0.35))
    for ax, o in zip(axes, OUTCOMES):
        front = results[o]["front"]
        ax.plot([r["dp_gap"] for r in front], [r["frac"] for r in front], linestyle="none",
                label="DP greedy (one point per λ)", **STYLE["DP greedy"])
        offsets = [(4, -8), (4, 3), (-24, 3)]
        for i, (x, y, text) in enumerate(merged_labels(front)):
            ax.annotate(text, (x, y), xytext=offsets[i % 3], textcoords="offset points", fontsize=6)
        for m, r in results[o]["refs"].items():
            ax.plot(r["dp_gap"], r["frac"], linestyle="none", label=m, **STYLE[m])
        ax.set_xlabel(f"DP gap ({o} rate)")
        ax.set_ylabel(f"fraction of exposed {o}")
        ax.set_title(rf"{o.capitalize()}: k={K}, $\mu_L$={MU_LOW}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.08))
    save(fig, "parity_pareto")

    rows = [(o, f"DP greedy λ={r['lam']:g}", r["frac"], r["dp_gap"]) for o in OUTCOMES for r in results[o]["front"]]
    rows += [(o, m, r["frac"], r["dp_gap"]) for o in OUTCOMES for m, r in results[o]["refs"].items()]
    save_table("parity_pareto", ["outcome", "method", "frac", "dp_gap"], rows)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot(json.loads((RESULTS / "parity_pareto.json").read_text())["results"])
    else:
        main()
