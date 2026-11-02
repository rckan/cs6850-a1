"""Why DP greedy puts its seeds in L, at mu_L = 0.3 on the converted outcome.

Four tables, no figure:
  parity_allocation_steps   best H vs best L candidate at each greedy step (lambda = 2)
  parity_allocation_splits  greedy forced to n of 10 seeds in L: is all-L just greedy's order?
  parity_allocation_noseed  DP greedy with truth seeds left out of the rates
  parity_same_seeds         each seed set scored on both outcomes, with L's non-seed rate
"""

import numpy as np

from experiments.common import CONFIG, METHODS, OUTCOMES, measure, save_json, save_table, seed_sets, setup
from fairblock import metrics
from fairblock.model import TRUTH, simulate

MU_LOW = 0.3
K = 10
LAM = CONFIG["lam"]
_IDX = {"saved": 0, "converted": 1}


def score(ev, seeds):
    return metrics.parity_objective(ev.evaluate(seeds)[1], ev.exposed, LAM)


def steps_table(ctx):
    """Follow DP greedy, reporting the best candidate of each group at every step."""
    tr, g = ctx["train"], ctx["groups"]
    seeds, rows = [], []
    for step in range(1, K + 1):
        f0 = score(tr, seeds) if seeds else 0.0
        r0 = metrics.rates(tr.evaluate(seeds)[1], tr.exposed)
        best = {}
        for v in ctx["candidates"]:
            if v in seeds:
                continue
            f = score(tr, seeds + [v])
            if g[v] not in best or f > best[g[v]][0]:
                r = metrics.rates(tr.evaluate(seeds + [v])[1], tr.exposed)
                best[g[v]] = (f, v, r["H"] - r0["H"], r["L"] - r0["L"])
        for h in ("H", "L"):
            f, _, dh, dl = best[h]
            rows.append((step, h, f - f0, dh, dl))
        seeds.append(max(best.values(), key=lambda t: t[0])[1])
    return rows


def split_table(ctx):
    """Greedy on F under a quota of n_low seeds in L, for every n_low."""
    tr, te, g = ctx["train"], ctx["test"], ctx["groups"]
    exposure = np.mean(te.baseline, axis=0)
    rows = []
    for n_low in range(K + 1):
        quota, seeds = {"L": n_low, "H": K - n_low}, []
        for _ in range(K):
            allowed = [v for v in ctx["candidates"] if v not in seeds and sum(g[u] == g[v] for u in seeds) < quota[g[v]]]
            seeds.append(max(allowed, key=lambda v: score(tr, seeds + [v])))
        m = measure(te, seeds, "converted")
        seed_share = sum(exposure[v] for v in seeds if g[v] == "L") / te.exposed["L"]
        rows.append((n_low, score(tr, seeds), m["frac"], m["dp_gap"], m["rates"]["H"], m["rates"]["L"], m["rates"]["L"] - seed_share))
    return rows


def converted_without_seeds(ev, seeds):
    out = dict.fromkeys(ev.labels, 0.0)
    for w, base in zip(ev.worlds, ev.baseline):
        counted = base & (simulate(w, ev.S_N, seeds) == TRUTH)
        counted[seeds] = False
        for g, mask in ev.masks.items():
            out[g] += counted[mask].sum()
    return {g: x / len(ev.worlds) for g, x in out.items()}


def noseed_row(ctx):
    tr, te, g = ctx["train"], ctx["test"], ctx["groups"]
    seeds = []
    for _ in range(K):
        options = [v for v in ctx["candidates"] if v not in seeds]
        seeds.append(max(options, key=lambda v: metrics.parity_objective(converted_without_seeds(tr, seeds + [v]), tr.exposed, LAM)))
    r = metrics.rates(converted_without_seeds(te, seeds), te.exposed)
    return [(int(sum(g[v] == "L" for v in seeds)), r["H"], r["L"], abs(r["H"] - r["L"]))]


def same_seeds_table(ctx):
    te, g = ctx["test"], ctx["groups"]
    exposure = np.mean(te.baseline, axis=0)
    rows = []
    for o in OUTCOMES:
        for k in (10, 30):
            seeds = seed_sets(ctx, [k], o)
            for m in METHODS:
                s = seeds[m][k]
                sav, conv = measure(te, s, "saved"), measure(te, s, "converted")
                share = sum(exposure[v] for v in s if g[v] == "L") / te.exposed["L"]
                rows.append((o, m, k, int(sum(g[v] == "L" for v in s)), sav["frac"], conv["frac"],
                             sav["rates"]["L"], conv["rates"]["L"], conv["rates"]["L"] - share))
    return rows


def main():
    ctx = setup(MU_LOW)
    tables = {
        "parity_allocation_steps": (["step", "best seed in", "change in F", "change in r_H", "change in r_L"], steps_table(ctx)),
        "parity_allocation_splits": (["seeds in L", "F (train)", "frac", "dp_gap", "r_H", "r_L", "r_L non-seed"], split_table(ctx)),
        "parity_allocation_noseed": (["seeds in L", "r_H", "r_L", "dp_gap"], noseed_row(ctx)),
        "parity_same_seeds": (["optimized for", "method", "k", "seeds in L", "saved frac", "converted frac", "saved r_L", "converted r_L", "converted r_L non-seed"], same_seeds_table(ctx)),
    }
    for name, (header, rows) in tables.items():
        save_table(name, header, rows)
    save_json("parity_allocation", {n: dict(header=h, rows=r) for n, (h, r) in tables.items()})


if __name__ == "__main__":
    main()
