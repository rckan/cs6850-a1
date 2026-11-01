# Maximin fairness: notes

Running log for Matthew's part: maximin fairness (FWRRS's objective) for fair
influence blocking when groups accept the truth at different rates. Math is in
`sections/maximin.tex`; this file holds the reasoning, decisions and checks.
Companion to `docs/parity.md` (Rachel), whose simulator, setup and worlds this
part reuses unchanged.

## Status

| Piece | Where | State |
|---|---|---|
| Maximin metrics (min rate, worst group, soft min) | `fairblock/metrics.py` (Maximin section) | done, tested |
| Worst-first greedy (lexicographic / random ties) | `fairblock/seeding.py` | done, tested |
| Maximin methods + Louvain relabelling | `experiments/maximin_common.py` | done |
| Figure style | `experiments/maximin_plotting.py` | done |
| Experiments A, B, C | `experiments/maximin_*.py` | done, see Results (redraw: `--plot-only`) |
| Math write-up + results | `sections/maximin.tex` | drafted; **Matthew to review** |
| Figures in paper | `sections/maximin_figures.tex` (after `sections/maximin`) | done |
| Simulation write-up | `sections/simulation.tex` | **Rachel + Matthew** |

Run everything: `pytest`, then `python -m experiments.maximin_sweep_acceptance`
(same for `maximin_sweep_budget`, `maximin_fragility`). About 10-15 minutes
with the three run in parallel. Add `--plot-only` to redraw figures from `results/`.

## Decisions (change these if you disagree)

1. **Lexicographic maximin, as in FWRRS.** Maximize the worst infected
   community's rate, then total saved among ties (equation 6). Plain maximin
   (random tie-break) is run only as a comparison, to test Claim 2.
2. **Worst-first greedy, not greedy on M.** M is monotone but not submodular,
   and greedy on M sees zero gain while any community is untouched. Each step
   targets the currently worst infected community c* and scores candidates by
   c*'s rate after adding them. "Greedy on M" is kept as a baseline in
   Experiment C.
3. **Tie tolerance tol = 0.005.** Rates are Monte Carlo means, so exact ties
   almost never occur and a literal tie-break would never fire; candidates
   within tol of the best target rate count as tied. 0.005 is about half the
   noise level Rachel reports (~0.01). Not tuned.
4. **Ties between equally worst communities go to the most exposed one**, then
   by label. Before any seed every rate is 0; breaking that tie by label alone
   would make the first target arbitrary (alphabetical).
5. **Soft min with tau = 0.05.** M_tau = -tau log sum_c exp(-r_c / tau), within
   tau log C of M. With C = 123 that is at most 0.24 below M, so it is a
   different objective at large C, not just a smoothed M. Not tuned.
6. **C_infected = groups with exposed > 0** on the evaluator's own worlds
   (training worlds when choosing seeds, test worlds when reporting).
7. **Analytic E/a uses exposure, not group size.** Claim 1 says k_g ∝ E_g / a_g.
   Rachel's "Analytic n/a" baseline uses n_g; mine uses the training exposure
   E_g and the realized group mean of a_v.
8. **Same setup as parity** (`experiments.common.setup`): same graph, rumor
   seeds, biases, train/test worlds and candidate pool, so every number is
   comparable with `docs/parity.md`. Parity's `METHODS` list is untouched, so
   her figures do not change; maximin's methods live in `maximin_common.py`.
9. **Experiment C partitions with Louvain** (`networkx`, seed 0) at
   resolutions 0.5-8, giving 2-123 communities on the same graph. Acceptance
   still follows the H/L bias model. The candidate pool adds each community's
   top-degree node so no community is unseedable for lack of candidates.
10. **Colors.** Total greedy, DP greedy and Proportional keep their parity
    colors. Worst-first variants share violet and differ by line style and
    marker. The legend order passes the dataviz palette checks for adjacent
    pairs (yellow next to orange failed, so yellow is unused); magenta is below
    3:1 contrast, so every figure has markers, a legend and a markdown table.

## Verification

- **Metrics** (`tests/test_maximin.py`): min rate skips uninfected groups
  (no 0/0), worst-group ties by exposure then label, soft min lies in
  [M - tau log C, M] and tends to M, and soft min rises when a non-worst group
  improves while M stays at 0.
- **Worst-first greedy** on a hand-built additive evaluator: targets the worst
  group, prefers the larger total among tied candidates, merges near-ties only
  within tol (0.200 vs 0.199 tie at tol 0.005, not at 0), and the random
  tie-break is reproducible from its seed.
- **On the simulator:** M never decreases as worst-first adds seeds (monotone,
  as claimed in the Properties paragraph).
- All 32 tests pass (23 existing + 9 new).

## Math (sections/maximin.tex), with what is exact vs. modeled

- **Monotone:** argued per realization (truth only arrives earlier, rumor
  only later). Consistent with Rachel's subset test (truth never adds rumor
  nodes) and the monotone-M test; no formal proof written.
- **Not submodular:** the two-community example is exact.
- **Claim 1** is parity's Claim 1 restated for maximin; exact in the ceiling model.
- **Claim 2 is my derivation**, in the ceiling model. Both totals are piecewise
  linear in k. Below min(E_H/b, k*) the ratio is constant (w_H + r w_L). After
  that, the optimum's slope drops (H saturates) or maximin's rises to b a_H
  (L saturates, tie-break sends seeds to H); either way d(PoF)/dk <= 0 since
  PoF >= 1. Both reach a_H E_H + a_L E_L at k = N/b. Plain maximin's worst
  case keeps r_H = a_L, i.e. DP's total a_L N and PoF r w_H + w_L.
- **Claim 3** reuses Rachel's exact converted bound; the "different worst
  community" statement is a possibility, not a measured result.

## Results

k = 10, mu_L = 0.3 unless stated; test worlds; PoF = total greedy's total /
the method's total. Tables: `figures/maximin_*.md` (`maximin_pof.md` has the
PoF table).

**A (sweep mu_L).**
- Saved: worst-first = total greedy at mu_L <= 0.3 (identical seeds, PoF 1.00;
  formula 1.34-1.89). 1.02 at 0.45, 1.06 at 0.8. Shielding makes maximin over
  saved nodes almost free, as Rachel found for DP.
- Converted: worst-first PoF 1.14 (0.8), 1.21, 1.40, 1.48 (0.3), 1.71 (0.15).
  Above the formula for mu_L >= 0.3, below it at 0.15. DP greedy: 2.34 at 0.3.
- DP greedy (all 10 seeds in L) reaches M_conv 0.143 vs worst-first 0.130 at
  mu_L = 0.3: **worst-first is not the maximin optimum.** It trades 0.013 of M
  for 58% more total (0.246 vs 0.156). The tol and greedy order both
  contribute; untested which matters more.
- Soft-min tracks worst-first closely on both outcomes.

**B (sweep k, mu_L = 0.3).**
- Converted PoF: worst-first rises to ~1.49 at k ≈ 11, then falls to 1.36 at
  k = 30. Random ties: 1.96 (k = 10) to 1.45. DP: 2.34 to 1.55.
- At k = 30 worst-first and DP have the same M_conv (0.337 vs 0.338) and
  worst-first saves 14% more in total (0.388 vs 0.340): the tie-break effect of
  Claim 2. The rise before the fall is not in the ceiling model, which has PoF
  constant at small k; worst-first's first target is H (most exposed when all
  rates are 0), so its first seed matches total greedy.
- L's converted rate passes mean a_L = 0.28 only via seeds (27 of 30 in L at k = 30).
- Saved PoF stays within 1.00-1.03 for worst-first throughout.
- PoF below 1 and spikes at k <= 3 are train/test noise.

**C (more communities).**
- **No infected community was ever at exactly 0**, for any method, up to 123
  communities: rates are means over 200 worlds and the truth leaks everywhere.
  The "M = 0 for every seed set" scenario did not occur on this graph.
- Worst-first degrades with more communities: M_sav 0.57 (2) to 0.38 (123),
  below total greedy's 0.49, and total 0.64 to 0.51. It spends one seed per
  worst small community.
- Soft-min holds up best: M_sav 0.50 and total 0.65 at 123.
- Converted: all methods fall to M ≈ 0.03-0.04 at 123.

**Caveats.** One graph, one set of rumor seeds, one draw of biases, candidates
limited to top-degree nodes; 200 test worlds, so ~0.01 differences are noise.
tol and tau untuned. No error bars.

## To do

Decide:
- [ ] Saved or converted as the headline maximin outcome (same open question as parity).
- [ ] Whether truth seeds count toward their group's rate (Rachel's Decision 9; affects both parts).
- [ ] Keep Experiment C's soft-min as a recommendation, or present it as exploratory.

Review:
- [ ] `sections/maximin.tex` line by line, especially Claim 2's derivation above.
- [ ] Results paragraph numbers against `figures/maximin_*.md`.
- [ ] With Rachel: the cross-references to parity (Claim 1, bound, ceiling model).

Optional:
- [ ] Sensitivity to tol (0, 0.005, 0.02) and tau (0.02, 0.05, 0.1).
- [ ] Brute force on a small graph to measure worst-first's gap to the true lexicographic optimum.
- [ ] Error bars over graph, rumor and bias seeds.

Before submitting:
- [ ] Page count (now 14 including working notes); shrink or drop Fig. C if over.