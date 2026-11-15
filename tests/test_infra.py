"""Shared project infrastructure: loaders, draw-aware setup, CI helper."""

import math

import networkx as nx
import numpy as np
import pytest

from experiments import common
from experiments.runner import mean_ci, run_tasks
from fairblock import graphs


# --- Loaders ---

def test_from_labels_takes_component_and_renumbers():
    G = nx.DiGraph([("a", "b"), ("b", "a"), ("b", "c"), ("c", "c"), ("x", "y")])
    labels = {"a": "red", "b": "blue", "c": "red", "x": "red", "y": "red"}
    H, groups = graphs.from_labels(G, labels, low="red")
    assert sorted(H.nodes) == [0, 1, 2]           # a, b, c; x-y component dropped
    assert sorted(H.edges) == [(0, 1), (1, 2)]    # undirected, no duplicate, no self-loop
    assert groups.tolist() == ["L", "H", "L"]


def test_karate_club_groups():
    G, groups = graphs.karate_club()
    assert G.number_of_nodes() == 34 and set(G.nodes) == set(range(34))
    assert (groups == "L").sum() == 17 and (groups == "H").sum() == 17
    assert groups[0] == "H"  # node 0 is Mr. Hi; default low side is "Officer"
    _, flipped = graphs.karate_club(low="Mr. Hi")
    assert (flipped != groups).all()


def test_sbm_probs_fix_mean_degree():
    for ratio in (0.02, 0.2, 1.0):
        p_in, p_out = graphs.sbm_probs(350, 150, ratio, 16.55)
        within = 350 * 349 / 2 + 150 * 149 / 2
        assert p_out == pytest.approx(ratio * p_in)
        assert 2 * (p_in * within + p_out * 350 * 150) / 500 == pytest.approx(16.55)
    # The exploratory network: ratio 0.2 gives back p_in = 0.05.
    deg = 2 * (0.05 * (350 * 349 / 2 + 150 * 149 / 2) + 0.01 * 350 * 150) / 500
    assert graphs.sbm_probs(350, 150, 0.2, deg) == pytest.approx((0.05, 0.01))


# --- Draws ---

SMALL = dict(common.CONFIG, R_train=5, R_test=5)


def test_draw_zero_is_the_exploratory_setup():
    assert common.draw_seeds(common.CONFIG, 0) == {k: common.CONFIG[k] for k in common.SEED_KEYS}


def test_draws_differ_and_are_reproducible():
    a, b, a2 = common.setup(0.3, SMALL, draw=1), common.setup(0.3, SMALL, draw=2), common.setup(0.3, SMALL, draw=1)
    assert sorted(a["G"].edges) != sorted(b["G"].edges)
    assert sorted(a["G"].edges) == sorted(a2["G"].edges)
    assert a["S_N"] == a2["S_N"] and np.array_equal(a["a"], a2["a"])


def test_setup_on_given_graph():
    G, groups = graphs.karate_club()
    ctx = common.setup(0.3, dict(SMALL, n_rumor=2, per_group=5), graph=(G, groups))
    assert ctx["G"] is G and len(ctx["candidates"]) == 10
    assert set(ctx["test"].exposed) == {"H", "L"}


# --- CI helper and runner ---

def test_mean_ci():
    mean, half = mean_ci([1.0, 2.0, 3.0])
    assert mean == pytest.approx(2.0)
    assert half == pytest.approx(4.303 * 1.0 / math.sqrt(3))
    assert math.isnan(mean_ci([5.0])[1])
    assert mean_ci([1.0, float("nan"), 3.0])[0] == pytest.approx(2.0)


def _add(x, y):
    return x + y


def test_run_tasks_keeps_order():
    tasks = [(i, 10 * i) for i in range(6)]
    assert run_tasks(_add, tasks, processes=1, progress=False) == [11 * i for i in range(6)]
    assert run_tasks(_add, tasks, processes=2, progress=False) == [11 * i for i in range(6)]
