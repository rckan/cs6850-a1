import networkx as nx

from fairblock import metrics, seeding
from fairblock.model import BlockingEvaluator, World, sample_worlds


def test_allocate_largest_remainder():
    assert seeding.allocate({"H": 350, "L": 150}, 10) == {"H": 7, "L": 3}
    # n/a weights: 350/1 vs 150/0.25 = 600 -> 10*350/950 = 3.68, 10*600/950 = 6.32
    assert seeding.allocate({"H": 350.0, "L": 600.0}, 10) == {"H": 4, "L": 6}
    assert seeding.allocate({"A": 1, "B": 1, "C": 1}, 4) == {"A": 2, "B": 1, "C": 1}
    assert sum(seeding.allocate({"H": 3, "L": 7}, 0).values()) == 0


def test_top_degree_and_exclude():
    G = nx.star_graph(4)  # 0 is the hub
    assert seeding.top_degree(G, range(5), 2) == [0, 1]
    assert seeding.top_degree(G, range(5), 2, exclude=[0]) == [1, 2]


def test_baselines_respect_groups_and_exclusions():
    G = nx.gnp_random_graph(50, 0.1, seed=0)
    groups = ["H"] * 35 + ["L"] * 15
    seeds = seeding.analytic_seeds(G, groups, {"H": 1.0, "L": 0.25}, 10, exclude=[0, 40])
    assert sum(groups[v] == "L" for v in seeds) == 6 and 0 not in seeds and 40 not in seeds
    seeds = seeding.proportional_seeds(G, groups, 10)
    assert sum(groups[v] == "L" for v in seeds) == 3


def test_greedy_on_shielding_example():
    # Diamond r=0, x=1, y=2, w=3, all edges live, w rejects. Seeding w itself
    # wins the step-1 ties at x and y, saving all 3 nodes.
    G = nx.Graph([(0, 1), (0, 2), (1, 3), (2, 3)])
    world = World([sorted(G.neighbors(u)) for u in range(4)], [True, True, True, False])
    ev = BlockingEvaluator([world], ["H"] * 4, [0])
    total = lambda s, e: metrics.total_saved(s)
    assert seeding.greedy(ev, [1, 2, 3], 1, total) == [3]
    assert seeding.greedy(ev, [1, 2], 2, total) == [1, 2]


def test_greedy_first_pick_is_best_single_seed():
    G = nx.gnp_random_graph(40, 0.1, seed=1)
    groups = ["H"] * 25 + ["L"] * 15
    worlds = sample_worlds(G, groups, 0.3, {"H": 0.9, "L": 0.4}, 50, seed=2)
    ev = BlockingEvaluator(worlds, groups, [0, 30])
    score = lambda s, e: metrics.parity_objective(s, e, 1.0)
    cands = [v for v in range(40) if v not in (0, 30)]
    best = max(cands, key=lambda v: (score(ev.evaluate([v])[0], ev.exposed), -v))
    assert seeding.greedy(ev, cands, 1, score) == [best]
    # Budget 2 equals brute force over pairs that extend the first pick.
    first = best
    second = max((v for v in cands if v != first),
                 key=lambda v: (score(ev.evaluate([first, v])[0], ev.exposed), -v))
    assert seeding.greedy(ev, cands, 2, score) == [first, second]
