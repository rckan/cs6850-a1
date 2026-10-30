import pytest

from fairblock import metrics
from fairblock.graphs import two_group_sbm


def test_rates_and_gaps():
    saved = {"H": 30.0, "L": 5.0}
    exposed = {"H": 60.0, "L": 20.0}
    assert metrics.rates(saved, exposed) == {"H": 0.5, "L": 0.25}
    assert metrics.dp_gap(saved, exposed) == pytest.approx(0.25)
    assert metrics.dp_ratio(saved, exposed) == pytest.approx(0.5)
    assert metrics.total_saved(saved) == 35.0


def test_parity_objectives():
    saved = {"H": 30.0, "L": 5.0}
    exposed = {"H": 60.0, "L": 20.0}
    # B/E = 35/80, gap = 0.25
    assert metrics.parity_objective(saved, exposed, 0.0) == pytest.approx(0.4375)
    assert metrics.parity_objective(saved, exposed, 2.0) == pytest.approx(0.4375 - 0.5)
    assert metrics.eps_parity_score(saved, exposed, 0.3) == (0.0, 35.0)
    violation, total = metrics.eps_parity_score(saved, exposed, 0.1)
    assert violation == pytest.approx(-0.15) and total == 35.0


def test_unexposed_group_has_zero_rate():
    assert metrics.rates({"H": 1.0, "L": 0.0}, {"H": 2.0, "L": 0.0})["L"] == 0.0


def test_two_group_sbm_labels():
    G, groups = two_group_sbm(30, 10, 0.2, 0.01, seed=0)
    assert G.number_of_nodes() == 40
    assert (groups[:30] == "H").all() and (groups[30:] == "L").all()
