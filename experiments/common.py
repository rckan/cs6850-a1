"""Shared setup for the parity experiments: network, rumor seeds, worlds, methods.

Greedy seeds are chosen on training worlds and every result is reported on a
separate set of test worlds, so greedy is not graded on the samples it fit.
Per-node acceptance a_v ~ clip(N(mu_g, sigma^2), 0, 1) uses the same noise for
every mu_L, and worlds use the same seed, so live edges are identical across
mu_L and each node's acceptance (and coin) is monotone in mu_L.
"""

import json
from pathlib import Path

import numpy as np

from fairblock import metrics, seeding
from fairblock.graphs import two_group_sbm
from fairblock.model import BlockingEvaluator, node_acceptance, sample_worlds

ROOT = Path(__file__).resolve().parent.parent
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"

CONFIG = dict(
    n_high=350, n_low=150, p_in=0.05, p_out=0.01,  # two-group SBM
    p=0.08,                 # IC edge probability (0.05 barely spreads; 0.1 reaches most nodes)
    n_rumor=5,              # rumor seeds, uniform over all nodes
    R_train=100, R_test=200,
    per_group=50,           # greedy candidates: top-degree nodes per group (> max k)
    mu_high=0.8,            # mean acceptance of H; a_v ~ clip(N(mu_g, sigma^2), 0, 1)
    sigma=0.1,              # spread within a community (bimodal once |mu_H - mu_L| > 2 sigma)
    lam=2.0,                # DP-greedy penalty weight for Experiments A and B
    eps=0.05,               # eps-DP tolerance
    graph_seed=0, rumor_seed=1, train_seed=2, test_seed=3, bias_seed=4,
)
OUTCOMES = ("saved", "converted")
_IDX = {"saved": 0, "converted": 1}


def setup(mu_low, cfg=CONFIG):
    """Everything needed to run methods with community mean acceptance {H: mu_high, L: mu_low}."""
    G, groups = two_group_sbm(cfg["n_high"], cfg["n_low"], cfg["p_in"], cfg["p_out"], seed=cfg["graph_seed"])
    rng = np.random.default_rng(cfg["rumor_seed"])
    S_N = sorted(rng.choice(G.number_of_nodes(), cfg["n_rumor"], replace=False).tolist())
    a = node_acceptance(groups, {"H": cfg["mu_high"], "L": mu_low}, cfg["sigma"], seed=cfg["bias_seed"])
    # Realized group means (clipping moves them slightly off mu_g).
    a_mean = {g: float(a[groups == g].mean()) for g in ("H", "L")}
    train = BlockingEvaluator(sample_worlds(G, groups, cfg["p"], a, cfg["R_train"], seed=cfg["train_seed"]), groups, S_N)
    test = BlockingEvaluator(sample_worlds(G, groups, cfg["p"], a, cfg["R_test"], seed=cfg["test_seed"]), groups, S_N)
    candidates = seeding.candidate_pool(G, groups, cfg["per_group"], exclude=S_N)
    return dict(G=G, groups=groups, S_N=S_N, a=a, a_mean=a_mean, train=train, test=test, candidates=candidates)


# Fixed method order: also the color order in every figure.
METHODS = ("Total greedy", "DP greedy", "eps-DP greedy", "Analytic n/a", "Proportional")


def greedy_scores(cfg=CONFIG):
    return {
        "Total greedy": lambda s, e: metrics.total_saved(s),
        "DP greedy": lambda s, e: metrics.parity_objective(s, e, cfg["lam"]),
        "eps-DP greedy": lambda s, e: metrics.eps_parity_score(s, e, cfg["eps"]),
    }


def seed_sets(ctx, ks, outcome, methods=METHODS, cfg=CONFIG):
    """{method: {k: seeds}}. Greedy runs once to max(ks) and is read off by prefix."""
    scores = greedy_scores(cfg)
    out = {}
    for m in methods:
        if m in scores:
            order = seeding.greedy(ctx["train"], ctx["candidates"], max(ks), scores[m], outcome)
            out[m] = {k: order[:k] for k in ks}
        elif m == "Analytic n/a":
            out[m] = {k: seeding.analytic_seeds(ctx["G"], ctx["groups"], ctx["a_mean"], k, ctx["S_N"]) for k in ks}
        elif m == "Proportional":
            out[m] = {k: seeding.proportional_seeds(ctx["G"], ctx["groups"], k, ctx["S_N"]) for k in ks}
    return out


def measure(ev, seeds, outcome):
    """Test-world summary of one seed set under the given outcome."""
    counts = ev.evaluate(seeds)[_IDX[outcome]]
    exposed = ev.exposed
    return dict(
        total=metrics.total_saved(counts),
        frac=metrics.total_saved(counts) / sum(exposed.values()),
        rates=metrics.rates(counts, exposed),
        dp_gap=metrics.dp_gap(counts, exposed),
    )


def save_json(name, obj):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(obj, indent=1))


def save_table(name, header, rows):
    """Markdown table next to each figure (the figure's table view)."""
    FIGURES.mkdir(exist_ok=True)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in r) + " |" for r in rows]
    (FIGURES / f"{name}.md").write_text("\n".join(lines) + "\n")
