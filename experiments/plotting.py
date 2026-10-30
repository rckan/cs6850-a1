"""Shared figure style. Each method keeps one color and marker in every figure.

Colors are slots 1, 2, 3, 5, 6 of the validated categorical palette (dataviz
skill reference palette, light mode). Two slots are below 3:1 contrast on
white, so every series also has its own marker and a legend, and each figure
gets a markdown table of its numbers (experiments.common.save_table).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.common import FIGURES, METHODS

STYLE = {
    "Total greedy": dict(color="#2a78d6", marker="o"),
    "DP greedy": dict(color="#eb6834", marker="s"),
    "eps-DP greedy": dict(color="#1baf7a", marker="^"),
    "Analytic n/a": dict(color="#e87ba4", marker="v"),
    "Proportional": dict(color="#008300", marker="P"),
}
assert set(STYLE) == set(METHODS)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "lines.markersize": 4,
    "figure.dpi": 200, "savefig.bbox": "tight",
})


def line(ax, x, y, method, **kw):
    ax.plot(x, y, **{"label": method, **STYLE[method], **kw})


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png")
    fig.savefig(FIGURES / f"{name}.pdf")  # vector, for the paper
    plt.close(fig)
