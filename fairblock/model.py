"""COICM with group-dependent acceptance (shared simulator).

Rules (paper, Further research direction 1):
  - Rumor (negative) seeds S_N and truth (positive) seeds S_P spread under the
    campaign-oblivious IC model with shared edge probability p. Each undirected
    edge gives two independent directed trials, u->v and v->u.
  - Seeds keep their campaign; S_N and S_P must be disjoint.
  - Each node v has its own acceptance probability a_v (its bias toward the
    truth), fixed across worlds. node_acceptance draws it from a normal
    centered on v's community mean, so communities differ on average while
    individuals vary; across two communities the distribution is bimodal.
  - In each world v accepts the truth with probability a_v. The coin is
    flipped once per world, so a node that rejects the truth rejects it every
    time. It stays inactive and can still be infected by the rumor, after
    which it spreads the rumor.
  - Truth wins ties: if both campaigns reach v in the same step and v accepts,
    v takes the truth; if v rejects, the rumor takes it.
  - Common random numbers: live edges and acceptance coins are sampled once
    (sample_worlds) and reused for every seed set compared.

A node is *exposed* if the rumor reaches it with S_P empty; rumor seeds are
not counted, since they can never be saved.
A node is *saved* if it is exposed but the rumor does not reach it with S_P.
This includes rejecting nodes shielded by truth nodes, so a group's saved
rate can exceed its acceptance rate. A node is *converted* if it is saved and
ends in the truth state.

States returned per node: 0 = inactive, 1 = rumor (M), 2 = truth (T).
"""

from collections import namedtuple

import numpy as np

INACTIVE, RUMOR, TRUTH = 0, 1, 2

# live_out[u]: list of v such that the directed trial u->v succeeds.
# accepts[v]: whether v accepts the truth in this world.
World = namedtuple("World", ["live_out", "accepts"])


def node_acceptance(groups, means, sigma, seed=None):
    """Per-node acceptance a_v = clip(means[g_v] + sigma * z_v, 0, 1), z_v ~ N(0, 1).

    The same z_v are drawn for a given seed whatever the means, so sweeping a
    community mean shifts every member's a_v by the same amount (before clipping).
    Clipping puts some mass exactly at 0 or 1 (nodes that never / always accept).
    """
    z = np.random.default_rng(seed).standard_normal(len(groups))
    mu = np.array([means[g] for g in groups], dtype=float)
    return np.clip(mu + sigma * z, 0.0, 1.0)


def sample_worlds(G, groups, p, a, R, seed=None):
    """Pre-sample R possible worlds: (live-edge out-lists, acceptance coins).

    G must have nodes 0..n-1; groups[v] is v's label. a is either a per-node
    array of acceptance probabilities (see node_acceptance) or a dict
    label -> probability shared by the whole group.
    """
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    if set(G.nodes) != set(range(n)):
        raise ValueError("G must have nodes 0..n-1")
    edges = np.array(G.edges, dtype=np.int64).reshape(-1, 2)
    directed = np.vstack([edges, edges[:, ::-1]])
    if isinstance(a, dict):
        accept_prob = np.array([a[g] for g in groups], dtype=float)
    else:
        accept_prob = np.asarray(a, dtype=float)
        if accept_prob.shape != (n,):
            raise ValueError("per-node acceptance must have one entry per node")

    worlds = []
    for _ in range(R):
        live = directed[rng.random(len(directed)) < p]
        live_out = [[] for _ in range(n)]
        for u, v in live.tolist():
            live_out[u].append(v)
        accepts = (rng.random(n) < accept_prob).tolist()
        worlds.append(World(live_out, accepts))
    return worlds


def simulate(world, S_N, S_P):
    """Run one cascade in a pre-sampled world. Returns a numpy array of states."""
    live_out, accepts = world
    S_N, S_P = list(S_N), list(S_P)
    if set(S_N) & set(S_P):
        raise ValueError("S_N and S_P must be disjoint")

    state = [INACTIVE] * len(live_out)
    for v in S_N:
        state[v] = RUMOR
    for v in S_P:
        state[v] = TRUTH

    truth_front, rumor_front = S_P, S_N
    while truth_front or rumor_front:
        reached_t = {v for u in truth_front for v in live_out[u] if state[v] == INACTIVE}
        reached_r = {v for u in rumor_front for v in live_out[u] if state[v] == INACTIVE}
        new_t = [v for v in reached_t if accepts[v]]
        new_r = [v for v in reached_r if not (v in reached_t and accepts[v])]
        for v in new_t:
            state[v] = TRUTH
        for v in new_r:
            state[v] = RUMOR
        truth_front, rumor_front = new_t, new_r
    return np.array(state, dtype=np.int8)


class BlockingEvaluator:
    """Expected per-group blocking for a fixed set of worlds and rumor seeds.

    The no-truth baseline is simulated once in the constructor; evaluate(S_P)
    then only simulates the competing cascade.
    """

    def __init__(self, worlds, groups, S_N):
        self.worlds = worlds
        self.S_N = list(S_N)
        groups = np.asarray(groups)
        self.labels = sorted(set(groups.tolist()))
        self.masks = {g: groups == g for g in self.labels}
        # Rumor seeds can never be saved, so they are left out of exposure.
        self.baseline = []
        for w in worlds:
            base = simulate(w, self.S_N, []) == RUMOR
            base[self.S_N] = False
            self.baseline.append(base)
        self.exposed = self._mean_by_group(self.baseline)

    def _mean_by_group(self, per_world):
        stacked = np.array(per_world)
        return {g: float(stacked[:, m].sum(axis=1).mean()) for g, m in self.masks.items()}

    def evaluate(self, S_P):
        """Returns (saved, converted): expected counts per group."""
        saved, converted = [], []
        for w, base in zip(self.worlds, self.baseline):
            state = simulate(w, self.S_N, S_P)
            s = base & (state != RUMOR)
            saved.append(s)
            converted.append(s & (state == TRUTH))
        return self._mean_by_group(saved), self._mean_by_group(converted)


def saved_by_group(worlds, groups, S_N, S_P):
    """Average over worlds of (saved nodes per group, rumor-exposed per group
    with S_P empty). Feeds directly into fairblock.metrics."""
    ev = BlockingEvaluator(worlds, groups, S_N)
    saved, _ = ev.evaluate(S_P)
    return saved, ev.exposed
