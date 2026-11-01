"""Truth-seed selection: one greedy routine for any objective, plus baselines.

A score function takes (saved, exposed) dicts (see fairblock.metrics) and
returns something comparable with > (a float, or a tuple compared
lexicographically). Examples:
    total:     lambda s, e: metrics.total_saved(s)
    parity:    lambda s, e: metrics.parity_objective(s, e, lam)
    soft min:  lambda s, e: metrics.soft_min_rate(s, e, tau)
Worst-first (maximin) greedy needs the current worst group at each step, so it
has its own routine, worst_first_greedy.
"""

import numpy as np

from fairblock import metrics


def top_degree(G, nodes, k, exclude=()):
    """The k highest-degree nodes among `nodes`, skipping `exclude` (ties by node id)."""
    exclude = set(exclude)
    ranked = sorted((v for v in nodes if v not in exclude), key=lambda v: (-G.degree(v), v))
    if len(ranked) < k:
        raise ValueError(f"only {len(ranked)} eligible nodes, need {k}")
    return ranked[:k]


def candidate_pool(G, groups, per_group, exclude=()):
    """Top-degree nodes from each group, to keep greedy fast."""
    groups, exclude = np.asarray(groups), set(exclude)
    pool = []
    for g in sorted(set(groups.tolist())):
        eligible = [v for v in np.flatnonzero(groups == g).tolist() if v not in exclude]
        pool += top_degree(G, eligible, min(per_group, len(eligible)))
    return pool


def greedy(evaluator, candidates, k, score, outcome="saved"):
    """Add the candidate with the best score k times (always uses the full budget).

    outcome picks which count the score sees: "saved" (rumor prevented, the
    papers' definition) or "converted" (saved and believes the truth).
    Returns the seeds in the order chosen, so seeds[:j] is greedy's answer for
    budget j. Ties go to the earlier candidate.
    """
    idx = {"saved": 0, "converted": 1}[outcome]
    seeds, remaining = [], list(candidates)
    for _ in range(k):
        best, best_score = None, None
        for v in remaining:
            counts = evaluator.evaluate(seeds + [v])[idx]
            s = score(counts, evaluator.exposed)
            if best is None or s > best_score:
                best, best_score = v, s
        seeds.append(best)
        remaining.remove(best)
    return seeds


def worst_first_greedy(evaluator, candidates, k, outcome="saved", tol=0.005, tiebreak="total", seed=None):
    """Maximin seeding (FWRRS-style), one seed at a time.

    Each step finds the currently worst infected group c* and scores every
    candidate by c*'s rate after adding it. Candidates within tol of the best
    such rate count as tied: tiebreak="total" picks the one with the most total
    saved (lexicographic maximin), tiebreak="random" picks one at random (plain
    maximin, indifferent among ties). tol is needed because rates are Monte
    Carlo averages, so exact ties almost never occur.
    Returns the seeds in the order chosen, as greedy does.
    """
    idx = {"saved": 0, "converted": 1}[outcome]
    rng = np.random.default_rng(seed)
    exposed = evaluator.exposed
    seeds, remaining = [], list(candidates)
    for _ in range(k):
        target = metrics.worst_group(evaluator.evaluate(seeds)[idx], exposed)
        scored = []
        for v in remaining:
            counts = evaluator.evaluate(seeds + [v])[idx]
            scored.append((v, metrics.rates(counts, exposed)[target], metrics.total_saved(counts)))
        best_rate = max(r for _, r, _ in scored)
        tied = [(v, total) for v, r, total in scored if r >= best_rate - tol]
        if tiebreak == "total":
            best = max(tied, key=lambda t: t[1])[0]  # max keeps the earliest of equal totals
        elif tiebreak == "random":
            best = tied[rng.integers(len(tied))][0]
        else:
            raise ValueError(f"unknown tiebreak {tiebreak!r}")
        seeds.append(best)
        remaining.remove(best)
    return seeds


def allocate(weights, k):
    """Split integer budget k across groups in proportion to weights
    (largest-remainder rounding, ties by group label)."""
    total = sum(weights.values())
    exact = {g: k * w / total for g, w in weights.items()}
    counts = {g: int(np.floor(x)) for g, x in exact.items()}
    leftover = k - sum(counts.values())
    for g in sorted(exact, key=lambda g: (-(exact[g] - counts[g]), g))[:leftover]:
        counts[g] += 1
    return counts


def seeds_by_allocation(G, groups, counts, exclude=()):
    """Top-degree seeds within each group, counts[g] from group g."""
    groups = np.asarray(groups)
    seeds = []
    for g, c in sorted(counts.items()):
        seeds += top_degree(G, np.flatnonzero(groups == g).tolist(), c, exclude)
    return seeds


def proportional_seeds(G, groups, k, exclude=()):
    """Parity Seeding baseline: k_g proportional to group size n_g."""
    labels, sizes = np.unique(np.asarray(groups), return_counts=True)
    return seeds_by_allocation(G, groups, allocate(dict(zip(labels.tolist(), sizes)), k), exclude)


def analytic_seeds(G, groups, a, k, exclude=()):
    """Allocation from the ceiling model, k_g proportional to n_g / a_g (needs a_g > 0)."""
    labels, sizes = np.unique(np.asarray(groups), return_counts=True)
    weights = {g: n / a[g] for g, n in zip(labels.tolist(), sizes)}
    return seeds_by_allocation(G, groups, allocate(weights, k), exclude)
