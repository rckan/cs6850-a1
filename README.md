# cs6850-a1

CS/INFO 6850 exploratory paper (due **Mon Nov 2, 6pm**): *Fairness in Influence
Blocking*. Rachel Kan (demographic parity) and Matthew Zhang (maximin).

## Layout

```
a1.tex              main paper (2-page summary + working notes)
sections/           paper sections, one file per owner, \input from a1.tex
  parity.tex        Rachel
  maximin.tex       Matthew
  simulation.tex    shared: setup + results
fairblock/          shared Python package
  graphs.py         two-group SBM
  model.py          COICM + group acceptance simulator (shared)
  metrics.py        rates, dp_gap / dp_ratio / parity objectives (Rachel)
  seeding.py        greedy for any objective + proportional / analytic baselines
experiments/        one script per experiment, named by owner
  common.py         shared config, setup, method list (train/test worlds)
  plotting.py       shared figure style (one color + marker per method)
figures/            committed plots (.png) + their numbers (.md) for the paper
results/            raw experiment output (git-ignored)
docs/               design notes and decisions (parity.md: Rachel)
tests/              pytest: dynamics, exact enumeration, sampler, greedy
```

## Setup

```
git clone git@github.com:rckan/cs6850-a1.git && cd cs6850-a1
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Build the paper with `latexmk -pdf -outdir=build a1.tex` (or LaTeX Workshop).

## Working together

- `git pull --rebase` before you start and before you push; push small commits often.
- Edit your own files (`sections/<you>.tex`, `experiments/<you>_*.py`) freely.
- Shared files (`a1.tex`, `fairblock/model.py`, `fairblock/graphs.py`,
  `sections/simulation.tex`): say in chat before a big edit, or use a branch + PR.
- Don't change a function signature in `fairblock/` without telling the other person.
- Use the same pre-sampled worlds (`model.sample_worlds`, same seed) so
  parity and maximin results are directly comparable.
