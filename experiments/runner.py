"""Run an experiment over many independent draws in parallel and summarize with CIs.

Typical use (see experiments/sweep_homophily.py):

    tasks = [(x, d) for x in XS for d in range(n_draws)]
    rows = run_tasks(one_task, tasks)          # one_task(x, d) -> dict
    mean, half = mean_ci([r["pof"] for r in rows if r["x"] == x])

one_task must be a module-level function (worker processes import it by name).
"""

import math
import os
from multiprocessing import Pool

import numpy as np

# Two-sided 95% Student t quantiles t_{0.975, df}; df > 30 uses the normal 1.96.
_T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
         2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
         2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def mean_ci(values):
    """(mean, half-width of the 95% t confidence interval). Half-width is nan for one value."""
    x = np.asarray([v for v in values if v is not None and not math.isnan(v)], dtype=float)
    if x.size == 0:
        return math.nan, math.nan
    if x.size == 1:
        return float(x[0]), math.nan
    df = x.size - 1
    t = _T975[df - 1] if df <= len(_T975) else 1.96
    return float(x.mean()), float(t * x.std(ddof=1) / math.sqrt(x.size))


def _call(args):
    fn, task = args
    return fn(*task)


def run_tasks(fn, tasks, processes=None, progress=True):
    """[fn(*task) for task in tasks], spread over processes (default: all cores).

    Results come back in task order. processes=1 runs serially (easier to debug).
    """
    tasks = list(tasks)
    processes = processes or os.cpu_count() or 1
    if processes == 1:
        out = []
        for i, t in enumerate(tasks):
            out.append(fn(*t))
            if progress:
                print(f"{i + 1}/{len(tasks)} done", flush=True)
        return out
    out = []
    with Pool(processes) as pool:
        for i, r in enumerate(pool.imap(_call, [(fn, t) for t in tasks])):
            out.append(r)
            if progress:
                print(f"{i + 1}/{len(tasks)} done", flush=True)
    return out
