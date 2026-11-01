"""Figure style for the maximin experiments (rcParams and save() come from experiments.plotting).

Methods shared with the parity figures keep their colors there (Total greedy blue,
DP greedy orange, Proportional green). The three maximin variants are one violet
family told apart by line style and marker. Legend order (maximin_common.METHODS)
passes the dataviz palette checks for adjacent pairs; magenta is below 3:1 on
white, so every figure has markers, a legend, and a markdown table.
"""

from experiments.maximin_common import METHODS
from experiments.plotting import STYLE as PARITY_STYLE
from experiments.plotting import GRID, INK, MUTED, save  # noqa: F401  (re-exported)

VIOLET = "#6250d6"
STYLE = {
    "Total greedy": PARITY_STYLE["Total greedy"],
    "DP greedy": PARITY_STYLE["DP greedy"],
    "Worst-first greedy": dict(color=VIOLET, marker="D"),
    "Worst-first, random ties": dict(color=VIOLET, marker="x", linestyle="--"),
    "Greedy on M": dict(color=VIOLET, marker="1", linestyle=":"),
    "Soft-min greedy": dict(color="#e34948", marker="*"),
    "Proportional": PARITY_STYLE["Proportional"],
    "Analytic E/a": dict(color="#e87ba4", marker="v"),
}
assert set(STYLE) == set(METHODS)


def line(ax, x, y, method, **kw):
    ax.plot(x, y, **{"label": method, **STYLE[method], **kw})
