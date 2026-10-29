"""COICM with group-dependent acceptance (shared simulator).

Rules (paper, Further research direction 1):
  - Rumor (negative) seeds S_N and truth (positive) seeds S_P spread under the
    campaign-oblivious IC model with shared edge probability p.
  - Truth wins ties: if both reach v at the same step, v takes the truth.
  - A node v in group g accepts the truth with probability a[g]. A node that
    rejects the truth stays inactive and can still be infected by the rumor.
  - Common random numbers: live edges and acceptance coins are sampled once
    (sample_worlds) and reused for every seed set compared.

States returned per node: 0 = inactive, 1 = rumor (M), 2 = truth (T).
"""

INACTIVE, RUMOR, TRUTH = 0, 1, 2


def sample_worlds(G, groups, p, a, R, seed=None):
    """Pre-sample R possible worlds: (live-edge subgraph, acceptance coins).

    a: dict mapping group label -> acceptance probability.
    """
    raise NotImplementedError


def simulate(world, S_N, S_P):
    """Run one cascade in a pre-sampled world. Returns a numpy array of states."""
    raise NotImplementedError


def saved_by_group(worlds, groups, S_N, S_P):
    """Average over worlds of (saved nodes per group, rumor-exposed per group
    with S_P empty). Feeds directly into fairblock.metrics."""
    raise NotImplementedError
