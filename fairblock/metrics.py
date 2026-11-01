"""Fairness metrics over per-group blocking rates.

All functions take dicts keyed by group label:
  saved[g]   -- expected number of saved nodes in g, B_g(S_P)
  exposed[g] -- expected rumor-infected nodes in g with no truth seeds, s^-_g(S_N, {})
"""

import numpy as np


def rates(saved, exposed):
    """Per-group blocking rate r_g = B_g / s^-_g (0 for groups the rumor never reaches)."""
    return {g: saved[g] / exposed[g] if exposed[g] > 0 else 0.0 for g in exposed}


def total_saved(saved):
    return sum(saved.values())


# --- Parity (Rachel) ---

def dp_gap(saved, exposed):
    """Demographic-parity gap: max_g r_g - min_g r_g."""
    r = rates(saved, exposed).values()
    return max(r) - min(r)


def dp_ratio(saved, exposed):
    """Ratio form of parity: min_g r_g / max_g r_g (1 = perfect parity)."""
    r = rates(saved, exposed).values()
    return min(r) / max(r) if max(r) > 0 else 1.0


def parity_objective(saved, exposed, lam):
    """F_lambda = B / E - lam * dp_gap: total blocked fraction minus a parity penalty.

    Neither monotone nor submodular: one seed can widen the gap more than it adds.
    """
    return total_saved(saved) / sum(exposed.values()) - lam * dp_gap(saved, exposed)


def eps_parity_score(saved, exposed, eps):
    """Score for greedy under the eps-DP constraint dp_gap <= eps.

    Returned as a tuple (compared lexicographically): first shrink any violation,
    then maximize total saved. Not a guarantee of feasibility at the end.
    """
    return (min(0.0, eps - dp_gap(saved, exposed)), total_saved(saved))


# --- Maximin (Matthew) ---

def infected(exposed):
    """Groups the rumor reaches with no truth seeds (C_infected); rates are 0/0 elsewhere."""
    return [g for g in sorted(exposed) if exposed[g] > 0]


def min_rate(saved, exposed):
    """Maximin value M = min over infected groups of r_g (1 if no group is infected)."""
    r = rates(saved, exposed)
    return min((r[g] for g in infected(exposed)), default=1.0)


def worst_group(saved, exposed):
    """The infected group with the lowest rate, the target of worst-first greedy.

    Ties (e.g. every group at 0 before any seed) go to the group with the most
    exposed nodes, then by label.
    """
    r = rates(saved, exposed)
    return min(infected(exposed), key=lambda g: (r[g], -exposed[g], g))


def soft_min_rate(saved, exposed, tau):
    """Soft minimum -tau * log sum_g exp(-r_g / tau) over infected groups.

    Lies in [M - tau log C, M] and tends to M as tau -> 0, but every group's
    rate moves it, so greedy sees a gain even while some group is stuck at 0.
    """
    r = np.array([rates(saved, exposed)[g] for g in infected(exposed)])
    if r.size == 0:
        return 1.0
    m = r.min()  # shift for numerical stability
    return float(m - tau * np.log(np.exp(-(r - m) / tau).sum()))
