import networkx as nx
import pytest

from fairblock import metrics, seeding
from fairblock.model import BlockingEvaluator, sample_worlds


# --- Metrics ---

def test_min_rate_ignores_uninfected_groups():
    saved = {"A": 3.0, "B": 1.0, "C": 0.0}
    exposed = {"A": 6.0, "B": 4.0, "C": 0.0}  # C never reached: excluded, not a 0 rate
    assert metrics.infected(exposed) == ["A", "B"]
    assert metrics.min_rate(saved, exposed) == pytest.approx(0.25)
    assert metrics.worst_group(saved, exposed) == "B"
    assert metrics.min_rate({"C": 0.0}, {"C": 0.0}) == 1.0  # nothing to protect


def test_worst_group_ties_by_exposure_then_label():
    assert metrics.worst_group({"A": 0.0, "B": 0.0}, {"A": 2.0, "B": 5.0}) == "B"
    assert metrics.worst_group({"B": 1.0, "A": 1.0}, {"B": 2.0, "A": 2.0}) == "A"


def test_soft_min_bounds_and_limit():
    saved = {"A": 3.0, "B": 1.0, "C": 2.0}
    exposed = {"A": 6.0, "B": 4.0, "C": 4.0}  # rates 0.5, 0.25, 0.5
    m = metrics.min_rate(saved, exposed)
    for tau in (0.5, 0.1, 0.02):
        s = metrics.soft_min_rate(saved, exposed, tau)
        assert m - tau * __import__("math").log(3) - 1e-12 <= s <= m
    assert metrics.soft_min_rate(saved, exposed, 1e-4) == pytest.approx(m, abs=1e-3)


def test_soft_min_rewards_non_worst_groups():
    # Hard min is stuck at 0 (group A untouched); soft min still rises when B improves.
    exposed = {"A": 4.0, "B": 4.0}
    lo = metrics.soft_min_rate({"A": 0.0, "B": 1.0}, exposed, 0.1)
    hi = metrics.soft_min_rate({"A": 0.0, "B": 3.0}, exposed, 0.1)
    assert metrics.min_rate({"A": 0.0, "B": 3.0}, exposed) == 0.0
    assert hi > lo


# --- Worst-first greedy on a hand-built evaluator ---

class TableEvaluator:
    """Each candidate saves a fixed amount per group; savings add up across seeds."""

    def __init__(self, gains, exposed):
        self.gains, self.exposed = gains, exposed

    def evaluate(self, seeds):
        saved = {g: min(sum(self.gains[v][g] for v in seeds), self.exposed[g]) for g in self.exposed}
        return saved, saved


EXPOSED = {"H": 10.0, "L": 10.0}
GAINS = {
    "h_big": {"H": 6.0, "L": 0.0},   # best for the total, useless for L
    "l_small": {"H": 0.0, "L": 2.0},
    "l_plus": {"H": 3.0, "L": 2.0},  # same help for L as l_small, more total
    "l_best": {"H": 0.0, "L": 4.0},
}


def test_worst_first_targets_worst_group():
    ev = TableEvaluator(GAINS, EXPOSED)
    # Both groups start at 0 with equal exposure; ties by label make H the first target, so h_big
    # goes first. Then L is worst and l_best helps it most.
    seeds = seeding.worst_first_greedy(ev, list(GAINS), 2, tol=0.0)
    assert seeds == ["h_big", "l_best"]


def test_lexicographic_tiebreak_prefers_total():
    ev = TableEvaluator({k: v for k, v in GAINS.items() if k != "l_best"}, {"H": 10.0, "L": 10.0})
    # After h_big, L is worst; l_small and l_plus help L equally, l_plus saves more in total.
    seeds = seeding.worst_first_greedy(ev, ["l_small", "l_plus", "h_big"], 2, tol=0.0)
    assert seeds == ["h_big", "l_plus"]


def test_tolerance_merges_near_ties():
    gains = {"h": {"H": 9.0, "L": 0.0}, "a": {"H": 0.0, "L": 2.0}, "b": {"H": 5.0, "L": 1.99}}
    ev = TableEvaluator(gains, {"H": 10.0, "L": 10.0})
    # After h, L is worst. Exactly, a helps L more (0.200 vs 0.199), so a wins.
    assert seeding.worst_first_greedy(ev, ["h", "a", "b"], 2, tol=0.0)[1] == "a"
    # With tol = 0.005 they tie, and b's larger total wins.
    assert seeding.worst_first_greedy(ev, ["h", "a", "b"], 2, tol=0.005)[1] == "b"


def test_random_tiebreak_is_seeded_and_stays_within_ties():
    ev = TableEvaluator(GAINS, EXPOSED)
    a = seeding.worst_first_greedy(ev, list(GAINS), 3, tol=0.0, tiebreak="random", seed=7)
    b = seeding.worst_first_greedy(ev, list(GAINS), 3, tol=0.0, tiebreak="random", seed=7)
    assert a == b
    assert a[0] == "h_big"  # the only best choice for H, so no randomness at step 1


def test_worst_first_runs_on_simulator():
    G = nx.gnp_random_graph(40, 0.12, seed=0)
    groups = ["H"] * 25 + ["L"] * 15
    worlds = sample_worlds(G, groups, 0.3, {"H": 0.9, "L": 0.3}, 20, seed=1)
    ev = BlockingEvaluator(worlds, groups, [0, 30])
    seeds = seeding.worst_first_greedy(ev, list(range(1, 30)), 4)
    assert len(set(seeds)) == 4 and not {0, 30} & set(seeds)
    # Monotone: each added seed never lowers the maximin value.
    vals = [metrics.min_rate(ev.evaluate(seeds[:j])[0], ev.exposed) for j in range(5)]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))
