import itertools

import networkx as nx
import numpy as np
import pytest

from fairblock.model import (
    INACTIVE, RUMOR, TRUTH, World, BlockingEvaluator, node_acceptance, sample_worlds, simulate,
)


def world_all_live(G, accepts):
    """Deterministic world: every directed edge live."""
    live_out = [sorted(G.neighbors(u)) for u in range(G.number_of_nodes())]
    return World(live_out, list(accepts))


# --- Deterministic dynamics (p = 1) ---

def diamond():
    # Paper, direction 2: r=0, x=1, y=2, w=3; edges r-x, r-y, x-w, y-w.
    return nx.Graph([(0, 1), (0, 2), (1, 3), (2, 3)])


def test_shielding_example_from_paper():
    w = world_all_live(diamond(), [True, True, True, False])  # w rejects
    # x alone: w rejects the truth, then gets the rumor via y.
    assert simulate(w, [0], [1]).tolist() == [RUMOR, TRUTH, RUMOR, RUMOR]
    # x and y: every rumor path to w is cut; w stays inactive (shielded).
    assert simulate(w, [0], [1, 2]).tolist() == [RUMOR, TRUTH, TRUTH, INACTIVE]


def test_tie_goes_to_truth_only_if_accepted():
    path = nx.Graph([(0, 1), (1, 2)])  # rumor 0, truth 2, both reach 1 at step 1
    assert simulate(world_all_live(path, [1, 1, 1]), [0], [2])[1] == TRUTH
    assert simulate(world_all_live(path, [1, 0, 1]), [0], [2])[1] == RUMOR


def test_rejector_still_spreads_rumor():
    path = nx.Graph([(0, 1), (1, 2), (2, 3)])  # truth 0, rumor 3
    # Step 1: 1 rejects the truth, 2 gets the rumor. Step 2: 1 gets the rumor via 2.
    state = simulate(world_all_live(path, [1, 0, 1, 1]), [3], [0])
    assert state.tolist() == [TRUTH, RUMOR, RUMOR, RUMOR]


def test_seed_sets_must_be_disjoint():
    with pytest.raises(ValueError):
        simulate(world_all_live(diamond(), [1] * 4), [0], [0])


def test_saved_and_converted_counts():
    groups = ["H", "H", "H", "L"]
    w = world_all_live(diamond(), [True, True, True, False])
    ev = BlockingEvaluator([w], groups, [0])
    assert ev.exposed == {"H": 2.0, "L": 1.0}  # rumor seed 0 not counted
    saved, converted = ev.evaluate([1, 2])
    assert saved == {"H": 2.0, "L": 1.0}
    assert converted == {"H": 2.0, "L": 0.0}  # w is saved but not converted
    saved, _ = ev.evaluate([1])
    assert saved == {"H": 1.0, "L": 0.0}


def test_zero_acceptance_converts_only_seeds():
    G = nx.gnp_random_graph(40, 0.15, seed=1)
    groups = ["H"] * 20 + ["L"] * 20
    worlds = sample_worlds(G, groups, 0.3, {"H": 0.0, "L": 0.0}, 30, seed=2)
    ev = BlockingEvaluator(worlds, groups, [0, 1])
    S_P = [5, 25]
    _, converted = ev.evaluate(S_P)
    expected = np.mean([[b[5], b[25]] for b in ev.baseline], axis=0)
    assert converted["H"] == pytest.approx(expected[0])
    assert converted["L"] == pytest.approx(expected[1])


def test_truth_never_adds_rumor_nodes():
    # Saved = baseline rumor minus rumor-with-truth relies on this subset property.
    G = nx.gnp_random_graph(60, 0.08, seed=3)
    groups = ["H"] * 30 + ["L"] * 30
    worlds = sample_worlds(G, groups, 0.4, {"H": 0.9, "L": 0.3}, 50, seed=4)
    for w in worlds:
        base = simulate(w, [0, 31], []) == RUMOR
        with_truth = simulate(w, [0, 31], [5, 6, 40]) == RUMOR
        assert not (with_truth & ~base).any()


# --- Sampler ---

def test_sampler_marginals_and_independent_directions():
    G = nx.Graph([(0, 1)])
    groups = ["H", "L"]
    R, p = 20000, 0.3
    worlds = sample_worlds(G, groups, p, {"H": 0.9, "L": 0.2}, R, seed=5)
    fwd = np.array([1 in w.live_out[0] for w in worlds])
    bwd = np.array([0 in w.live_out[1] for w in worlds])
    tol = 5 * np.sqrt(p * (1 - p) / R)
    assert fwd.mean() == pytest.approx(p, abs=tol)
    assert bwd.mean() == pytest.approx(p, abs=tol)
    # Independent directions: P(both live) = p^2 (correlated sampling would give p).
    assert (fwd & bwd).mean() == pytest.approx(p**2, abs=tol)
    acc = np.array([w.accepts for w in worlds])
    assert acc[:, 0].mean() == pytest.approx(0.9, abs=0.01)
    assert acc[:, 1].mean() == pytest.approx(0.2, abs=0.015)


# --- Per-node acceptance (community-centered bias) ---

def test_node_acceptance_centered_on_community_mean():
    groups = ["H"] * 5000 + ["L"] * 5000
    a = node_acceptance(groups, {"H": 0.8, "L": 0.3}, 0.1, seed=0)
    assert ((a >= 0) & (a <= 1)).all()
    assert a[:5000].mean() == pytest.approx(0.8, abs=0.01)  # 2.3% clipped at 1
    assert a[5000:].mean() == pytest.approx(0.3, abs=0.01)
    assert a[:5000].std() == pytest.approx(0.1, abs=0.01)
    # Bimodal overall: a 0.1-wide bin between the modes holds ~2% of nodes
    # (|z| in 2..3), a bin at a mode ~19% (|z| < 0.5, half the nodes).
    dip = ((a > 0.5) & (a < 0.6)).mean()
    peak = ((a > 0.75) & (a < 0.85)).mean()
    assert dip < peak / 4


def test_node_acceptance_common_noise_across_means():
    groups = ["H"] * 50 + ["L"] * 50
    lo = node_acceptance(groups, {"H": 0.8, "L": 0.3}, 0.1, seed=1)
    hi = node_acceptance(groups, {"H": 0.8, "L": 0.5}, 0.1, seed=1)
    assert np.array_equal(lo[:50], hi[:50])
    assert (hi[50:] >= lo[50:]).all()


def test_sampler_uses_per_node_acceptance():
    G = nx.Graph([(0, 1), (1, 2)])
    worlds = sample_worlds(G, ["H"] * 3, 0.5, [0.1, 0.5, 0.9], 20000, seed=6)
    acc = np.array([w.accepts for w in worlds]).mean(axis=0)
    assert acc == pytest.approx([0.1, 0.5, 0.9], abs=0.015)
    with pytest.raises(ValueError):
        sample_worlds(G, ["H"] * 3, 0.5, [0.1, 0.5], 1)


# --- Probabilistic: closed form, exact enumeration, Monte Carlo ---

def exact_expectation(G, groups, p, a, S_N, S_P):
    """Exact expected (saved, converted, exposed) per group, plus each node's
    exposure probability, by enumerating every live-edge configuration and
    acceptance outcome. a: dict per group or list per node."""
    n = G.number_of_nodes()
    a = [a[g] for g in groups] if isinstance(a, dict) else list(a)
    directed = [(u, v) for u, v in G.edges] + [(v, u) for u, v in G.edges]
    labels = sorted(set(groups))
    out = {k: dict.fromkeys(labels, 0.0) for k in ("saved", "converted", "exposed")}
    out["p_exposed"] = np.zeros(n)
    for live in itertools.product([0, 1], repeat=len(directed)):
        p_live = np.prod([p if l else 1 - p for l in live])
        live_out = [[] for _ in range(n)]
        for (u, v), l in zip(directed, live):
            if l:
                live_out[u].append(v)
        for acc in itertools.product([0, 1], repeat=n):
            q = np.prod([a[v] if acc[v] else 1 - a[v] for v in range(n)])
            w = World(live_out, list(acc))
            base = simulate(w, S_N, []) == RUMOR
            base[list(S_N)] = False
            state = simulate(w, S_N, S_P)
            saved = base & (state != RUMOR)
            out["p_exposed"] += p_live * q * base
            for v in range(n):
                g = groups[v]
                out["exposed"][g] += p_live * q * base[v]
                out["saved"][g] += p_live * q * saved[v]
                out["converted"][g] += p_live * q * (saved[v] and state[v] == TRUTH)
    return out


def test_exact_enumeration_matches_closed_form():
    # r=0 (H), t=1 (H), v=2 (L); edges r-v, t-v. Rumor at r, truth at t.
    # v saved iff r->v, t->v live and v accepts: p^2 a. t saved iff r->v, v->t: p^2.
    p, a = 0.6, 0.3
    G = nx.Graph([(0, 2), (1, 2)])
    out = exact_expectation(G, ["H", "H", "L"], p, {"H": 1.0, "L": a}, [0], [1])
    assert out["exposed"] == pytest.approx({"H": p**2, "L": p})
    assert out["saved"] == pytest.approx({"H": p**2, "L": p**2 * a})
    assert out["converted"] == pytest.approx({"H": p**2, "L": p**2 * a})


def test_converted_bound_is_exact_with_per_node_acceptance():
    # E[converted_c] <= sum_{v in c, v not in S_P} a_v P(v exposed) + sum_{v in S_P ∩ c} P(v exposed)
    G = nx.Graph([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (2, 4)])
    groups = ["H", "H", "L", "L", "L"]
    a = [0.9, 0.7, 0.2, 0.5, 0.1]
    S_P = [2]
    out = exact_expectation(G, groups, 0.5, a, [0], S_P)
    for g in ("H", "L"):
        bound = sum((1.0 if v in S_P else a[v]) * out["p_exposed"][v]
                    for v in range(5) if groups[v] == g)
        assert out["converted"][g] <= bound + 1e-12


def test_monte_carlo_matches_exact_enumeration():
    # Diamond plus a tail, per-node acceptance, p = 0.5: checks sampling + averaging.
    G = nx.Graph([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)])
    groups = ["H", "H", "L", "L", "L"]
    p, a = 0.5, [0.9, 0.7, 0.45, 0.3, 0.6]
    exact = exact_expectation(G, groups, p, a, [0], [1])
    R = 40000
    ev = BlockingEvaluator(sample_worlds(G, groups, p, a, R, seed=0), groups, [0])
    saved, converted = ev.evaluate([1])
    # Each per-group count is in [0, 3], so SE <= 1.5 / sqrt(R); allow 5 SE.
    tol = 5 * 1.5 / np.sqrt(R)
    for g in ("H", "L"):
        assert ev.exposed[g] == pytest.approx(exact["exposed"][g], abs=tol)
        assert saved[g] == pytest.approx(exact["saved"][g], abs=tol)
        assert converted[g] == pytest.approx(exact["converted"][g], abs=tol)
