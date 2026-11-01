"""Shared setup for the maximin experiments, on top of experiments.common.

Same network, rumor seeds, biases and train/test worlds as the parity
experiments (common.setup), so the two sets of results are directly
comparable. Parity's METHODS list is left alone; the maximin methods live here.
"""

import networkx as nx
import numpy as np

from experiments.common import CONFIG, greedy_scores
from fairblock import metrics, seeding
from fairblock.model import BlockingEvaluator

MAXIMIN_CONFIG = dict(
    CONFIG,
    tol=0.005,      # worst-first tie tolerance on the target group's rate (rates are Monte Carlo means)
    tau=0.05,       # soft-min temperature: within tau*log(C) of the true min
    tie_seed=5,     # random tie-break for plain maximin
)

# Fixed order: also the legend order, chosen so adjacent colors pass the palette checks.
METHODS = (
    "Total greedy", "DP greedy",
    "Worst-first greedy", "Worst-first, random ties", "Greedy on M",
    "Soft-min greedy", "Proportional", "Analytic E/a",
)


def seed_sets(ctx, ks, outcome, methods, cfg=MAXIMIN_CONFIG, train=None, groups=None):
    """{method: {k: seeds}}. Greedy methods run once to max(ks) and are read off by prefix.

    train and groups default to the two-group H/L setup; pass a community-labelled
    evaluator and labels to run the same methods over finer communities.
    """
    train = train or ctx["train"]
    groups = np.asarray(ctx["groups"] if groups is None else groups)
    K = max(ks)
    out = {}
    for m in methods:
        if m == "Total greedy":
            order = seeding.greedy(train, ctx["candidates"], K, lambda s, e: metrics.total_saved(s), outcome)
        elif m == "DP greedy":
            order = seeding.greedy(train, ctx["candidates"], K, greedy_scores(cfg)["DP greedy"], outcome)
        elif m == "Greedy on M":
            order = seeding.greedy(train, ctx["candidates"], K, lambda s, e: metrics.min_rate(s, e), outcome)
        elif m == "Soft-min greedy":
            order = seeding.greedy(train, ctx["candidates"], K, lambda s, e: metrics.soft_min_rate(s, e, cfg["tau"]), outcome)
        elif m == "Worst-first greedy":
            order = seeding.worst_first_greedy(train, ctx["candidates"], K, outcome, cfg["tol"], "total")
        elif m == "Worst-first, random ties":
            order = seeding.worst_first_greedy(train, ctx["candidates"], K, outcome, cfg["tol"], "random", cfg["tie_seed"])
        elif m == "Proportional":
            out[m] = {k: seeding.proportional_seeds(ctx["G"], groups, k, ctx["S_N"]) for k in ks}
            continue
        elif m == "Analytic E/a":
            out[m] = {k: analytic_exposure_seeds(ctx, train, groups, k) for k in ks}
            continue
        else:
            raise ValueError(f"unknown method {m!r}")
        out[m] = {k: order[:k] for k in ks}
    return out


def analytic_exposure_seeds(ctx, train, groups, k):
    """Claim 1's allocation, k_g proportional to E_g / mean a_g (training exposure),
    top-degree nodes within each infected group."""
    a = ctx["a"]
    weights = {g: train.exposed[g] / max(float(a[groups == g].mean()), 1e-6) for g in metrics.infected(train.exposed)}
    counts = seeding.allocate(weights, k)
    # A group can be smaller than its share; spill the excess to the largest remaining weights.
    seeds = []
    for g in sorted(counts, key=lambda g: -weights[g]):
        pool = [v for v in np.flatnonzero(groups == g).tolist() if v not in set(ctx["S_N"]) | set(seeds)]
        take = min(counts[g], len(pool))
        seeds += seeding.top_degree(ctx["G"], pool, take)
    if len(seeds) < k:
        rest = [v for v in ctx["G"].nodes if v not in set(ctx["S_N"]) | set(seeds)]
        seeds += seeding.top_degree(ctx["G"], rest, k - len(seeds))
    return seeds


def measure(ev, seeds, outcome):
    """Test-world summary of one seed set: maximin value, total, and per-group rates."""
    counts = ev.evaluate(seeds)[{"saved": 0, "converted": 1}[outcome]]
    exposed = ev.exposed
    rates = metrics.rates(counts, exposed)
    inf = metrics.infected(exposed)
    return dict(
        min_rate=metrics.min_rate(counts, exposed),
        frac=metrics.total_saved(counts) / sum(exposed.values()),
        rates=rates,
        worst=metrics.worst_group(counts, exposed),
        n_infected=len(inf),
        n_zero=sum(rates[g] <= 1e-12 for g in inf),
    )


def louvain_labels(G, resolution, seed=0):
    """Community label per node from Louvain at the given resolution (as FWRRS uses)."""
    comms = nx.community.louvain_communities(G, resolution=resolution, seed=seed)
    labels = np.empty(G.number_of_nodes(), dtype=object)
    for i, c in enumerate(sorted(comms, key=lambda c: min(c))):
        for v in c:
            labels[v] = f"c{i:03d}"
    return labels.astype(str)


def relabel(ev, labels):
    """Same worlds and rumor seeds as ev, grouped by new labels."""
    return BlockingEvaluator(ev.worlds, labels, ev.S_N)
