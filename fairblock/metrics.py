"""Fairness metrics over per-group blocking rates.

All functions take dicts keyed by group label:
  saved[g]   -- expected number of saved nodes in g, B_g(S_P)
  exposed[g] -- expected rumor-infected nodes in g with no truth seeds, s^-_g(S_N, {})
"""


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


# --- Maximin (Matthew) ---

def min_rate(saved, exposed):
    """Maximin objective: the worst-off group's blocking rate."""
    return min(rates(saved, exposed).values())
