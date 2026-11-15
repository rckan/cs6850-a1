# Experiments

One script per experiment, each importing from `fairblock`. Scripts write raw
numbers to `results/` (git-ignored) and final plots to `figures/` (committed,
included in the paper).

Name scripts by owner to avoid conflicts, e.g. `parity_sweep_acceptance.py`,
`maximin_sweep_acceptance.py`. Run from the repo root:

```
python -m experiments.parity_sweep_acceptance
```

## Project infrastructure (multi-draw runs)

- `common.setup(mu_low, cfg, draw=0, graph=None)`: `draw` picks an independent
  replicate (new SBM, rumor seeds, acceptance noise, worlds; every seed shifts by
  100·draw). `draw=0` is exactly the exploratory paper's setup. Pass
  `graph=(G, groups)` to run on a fixed network instead of the SBM.
- `runner.run_tasks(fn, tasks)` runs `fn(*task)` over all cores (keep `fn` at
  module level); `runner.mean_ci(values)` gives the mean and 95% t-interval
  half-width.
- `plotting.line_ci(ax, x, mean, half, style, label)` draws a mean line with a CI band.
- `fairblock.graphs`: `sbm_fixed_degree` / `sbm_probs` (homophily at fixed mean
  degree), `karate_club(low=...)`.

Example: `python -m experiments.sweep_homophily --draws 20` (about 8 min on 8
cores; `--draws 2` for a quick check, `--plot-only` to redraw).
