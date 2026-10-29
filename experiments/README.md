# Experiments

One script per experiment, each importing from `fairblock`. Scripts write raw
numbers to `results/` (git-ignored) and final plots to `figures/` (committed,
included in the paper).

Name scripts by owner to avoid conflicts, e.g. `parity_sweep_acceptance.py`,
`maximin_sweep_acceptance.py`. Run from the repo root:

```
python -m experiments.parity_sweep_acceptance
```
