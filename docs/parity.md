# Parity fairness: notes

Running log for Rachel's part: demographic parity (DP) for fair influence
blocking when groups accept the truth at different rates. Math is in
`sections/parity.tex`; this file holds the reasoning, decisions and checks.

## Status

| Piece | Where | State |
|---|---|---|
| Simulator (COICM + acceptance) | `fairblock/model.py` | done, tested |
| Parity metrics and objectives | `fairblock/metrics.py` | done, tested |
| Greedy + baselines | `fairblock/seeding.py` | done, tested |
| Experiments A, B, C + seed-placement checks | `experiments/parity_*.py` | done, see Results (redraw: `--plot-only`) |
| Math write-up | `sections/parity.tex` | drafted; **Rachel to review** |
| Figures in paper | `sections/parity_figures.tex` (after the two-column body) | done |
| Parity results text | `sections/simulation.tex` | drafted, no TODOs left; **Rachel to review** |
| Parity vs maximin | `sections/comparison.tex` | drafted; **Rachel + Matthew to review** |

Run everything: `pytest`, then `python -m experiments.parity_sweep_acceptance`
(same for `parity_sweep_budget`, `parity_pareto`, `parity_allocation`). About 15-20 minutes in total. Add `--plot-only` to redraw figures from `results/`.

## Decisions (change these if you disagree)

1. **Two outcomes, both reported.** *Saved* = the rumor would reach the node
   with no truth seeds but doesn't with them (the papers' definition). It counts
   rejecting nodes that truth nodes shield. *Converted* = saved and ends up
   believing the truth. **Decided (Rachel, Nov 1): the paper keeps both.**
2. **Rumor seeds don't count as exposed**, because they can never be saved.
   Including them would only lower every rate.
3. **Bias (acceptance) is per node, centered on its community.** Each node gets
   a_v = clip(mu_g + sigma z_v, 0, 1) with z_v ~ N(0, 1), drawn once and fixed.
   Communities differ on average, individuals vary, and the whole network is
   bimodal once |mu_H - mu_L| > 2 sigma. This follows the idea in `a1.tex`.
   Clipping is a choice: it creates a few nodes with a_v exactly 0 (never
   accept) or 1 (always accept). A truncated normal or a Beta distribution would
   avoid that. In each world v accepts with probability a_v, one coin per world:
   a node that rejects the truth rejects it every time, stays inactive, and can
   still get and spread the rumor.
4. **Ties:** both campaigns reach v in the same step, then truth wins if v
   accepts, otherwise the rumor wins. Truth seeds are truth regardless of coin.
5. **Greedy always uses the full budget k.** Fang's problem allows |S_P| <= k.
   With a fixed k, DP greedy has to put extra seeds where they hurt parity least.
6. **Train/test split.** Greedy picks seeds on 100 sampled worlds; all numbers
   are reported on 200 different worlds. The z_v and the worlds use the same
   random seeds for every mu_L, so live edges are identical and each node's
   acceptance is monotone in mu_L.
7. **No maximin code here.** Maximin is Matthew's task. Experiments compare DP
   with total greedy and the two allocation baselines only. Plug in his method
   when it's ready (add it to `METHODS` and `greedy_scores` in
   `experiments/common.py`).
8. **Parameters** (`experiments/common.py`): SBM 350 H / 150 L, p_in 0.05,
   p_out 0.01, edge probability p = 0.08, 5 random rumor seeds, mu_H = 0.8,
   sigma = 0.1, candidates = top 50 by degree per group, lambda = 2 in Experiments
   A and B (a strict setting; C sweeps 14 values from 0 to 8), eps = 0.05.
   - p = 0.05 barely spreads (about 23 exposed nodes); p = 0.1 reaches about
     2/3 of the network. p = 0.08 exposes about 184 H / 42 L nodes.
   - mu_H = 0.8 rather than 1 so H nodes vary on both sides of the mean.
     sigma is 0.1. edit if needed
   - The pool was 30 at first, but at k = 30 DP greedy had used every L
     candidate. 50 > max k.
9. **Truth seeds count toward their group's rate.** **Decided (Rachel, Nov 1).**
   L has only about 40 exposed nodes, so seeds can dominate its rate; the paper
   reports L's non-seed rate alongside where that matters.
10. **No error bars for now** (Rachel, Nov 1): the paper is exploratory. It
    states this as a limitation.
11. **Analytic baseline uses E_g / mean a_g** ("Analytic E/a"), matching Claim 1 and
    Matthew's baseline. It used n_g before, which gave 6 of 10 seeds in L instead of 4.

## Verification

- **Dynamics** (`tests/test_model.py`): hand-worked cases with every edge live.
  These cover the direction 2 diamond (x alone: w gets the rumor; x and y: w
  shielded), ties going to the truth only on acceptance, and a rejecting node
  later spreading the rumor.
- **Closed form:** for r - v - t with rumor at r and truth at t, the expected saved
  in L is p^2 a. An exact enumeration over all live-edge and coin outcomes
  matches it.
- **Monte Carlo vs exact enumeration** on a 5-node graph (40k worlds, within 5 SE).
- **Sampler:** each direction is live with probability p, the two directions
  are independent (P(both) = p^2), and acceptance frequencies equal each
  node's a_v.
- **Bias sampling:** group means and spread match mu_g and sigma, the result is
  bimodal (a bin between the modes holds well under a quarter of a bin at a
  mode), and the same noise is reused across mu_L.
- **Bound, exactly:** on a 5-node graph with per-node acceptance, exact
  enumeration confirms the converted bound below.
- **Mutation checks:** I broke the code on purpose twice: coupling the two
  directions of each edge, and giving ties to the rumor. The tests catch both.
  An earlier version of the tests missed the first one, which is why the direct
  sampler test exists.
- **Subset property:** with truth seeds, the rumor's set is always a subset of
  its set without them (checked on random worlds). The definition of saved
  depends on this.
- **Greedy** matches brute force for the first two picks on a random graph.
- **The lambda identity** F = (w_H - lam) r_H + (w_L + lam) r_L when H leads is tested.
- **Same setup as maximin:** Matthew's three experiments, rerun on a separate
  copy with networkx 3.7, reproduce all four of his result tables exactly.

## Math (sections/parity.tex), with what is exact vs. modeled

- **Exact in the simulator:** E[converted_c] <= ā_c E_c + E|S_P ∩ exposed_c|,
  where ā_c = sum_{v in c} a_v P(v exposed) / E_c is the exposure-weighted
  mean acceptance. Exposure depends only on live edges, never on acceptance
  coins, so P(v exposed and accepts) = a_v P(v exposed). Saved has **no** such
  cap, because of shielding.
- The ceiling model and Claims 1-3 use group mean acceptances a_g = ā_g.
  Experiment A's formula uses the realized simple group means of a_v. These are
  close to ā_g but not identical; the exposure weighting is not computed.
- **Ceiling model (stylized):** each seed in g converts b a_g nodes of g only,
  k_g is continuous, r_g = min(b a_g k_g / E_g, a_g). Claims 1-3 hold exactly
  *inside this model*. I re-derived each:
  - Claim 1: maximin = DP-max-total below saturation (analysis only; no
    maximin code), k_g ∝ E_g / a_g,
    PoF = w_H + r w_L. This needs b k <= E_H so that the "all seeds to H"
    benchmark doesn't saturate.
  - Claim 2: k* = (E_L / b)(1 + E_H a_L / (E_L a_H)). Past k*, DP's total stays
    at a_L N; lexicographic maximin gains min(b a_H (k - k*), (a_H - a_L) E_H).
    DP's PoF is capped at r w_H + w_L, so it is bounded in k and unbounded only
    as a_L -> 0.
  - Claim 3: past k*, eps-DP lets r_H reach min(a_L + eps, a_H); the ratio form
    lets it reach min(a_L / tau, a_H).

### Corrections to the original plan
- "PoF grows without bound in k" was wrong: it is capped (see Claim 2).
- "a_L = 0 implies nothing saved" was wrong, twice over. Shielded rejecting
  nodes count as saved, and truth seeds save themselves. The correct statement:
  with a = 0 everywhere, only the seeds are converted (tested).
- The ceiling r_g <= a_g holds for the converted rate (plus seeds), not the
  saved rate.

## Results

Per-node acceptance, mu_H = 0.8, sigma = 0.1. Tables are in `figures/` (`parity_*.md`).
Unless stated: k = 10, mu_L = 0.3 (realized mean acceptance 0.808 in H, 0.280 in L),
lambda = 2, test worlds. PoF = total greedy's total / the method's total.

**Saved: acceptance matters little (shielding).**
- Total greedy's total falls only from 0.707 (mu_L = 0.8) to 0.681 (0.15). Every
  method's gap is at most 0.12. DP greedy's PoF is 1.06-1.11.
- Total greedy's saved-run seeds (0 in L) save 0.60 of L's exposed nodes but
  convert 0.08.

**Converted: acceptance matters a lot.**
- Total greedy: 0 seeds in L, gap 0.354 (r_H 0.43, r_L 0.08); gap 0.145 even at
  equal target means. Position vs L's slightly lower realized mean was not separated.
- DP greedy: 10 seeds in L, gap 0.011, total 0.149, PoF 2.47 (formula 1.35).
  eps-DP: 9 seeds, gap 0.040, total 0.157. Analytic E/a: 4 seeds, gap 0.263, PoF 1.10.
- DP's seeds score 0.264 on the saved outcome, vs 0.678 for total greedy's
  converted-run seeds.
- **Claim 2's plateau is not seen for k <= 30:** DP's total is 0.15, 0.28, 0.36 at
  k = 10, 20, 30, gap below 0.02. Open whether it appears at larger k or with early stopping.
- **Bound holds:** L's rate passes mean acceptance at k = 21; at k = 30, seeds are
  76% of its rate (0.366 overall, 0.087 non-seed).
- **Strict parity levels down:** H falls from 0.43 to 0.15, and L's non-seed rate
  is lower than under total greedy (0.051 vs 0.077).

**Why DP greedy puts its seeds in L (`parity_allocation.py`, lambda = 2).**
- Not simply H's negative weight past w_H: an H seed also raises r_L. With the best
  first seeds at lambda = 2, the H seed's gain in F stays positive until lambda ~ 1.05,
  but the L seed scores higher once lambda > 0.67 (audit, Nov 2; checked). At lambda = 2
  the best H seed lowers F at every step.
- Not greedy's order: forcing 0-10 seeds into L, F rises steadily (-0.27 to 0.15).
  Each split is still built greedily, so this is not a full brute-force check.
- Not seed counting: with seeds left out of the rates it still picks 10 of 10
  (r_H 0.123, r_L 0.052).
- Claim 1's mismatch: at lambda = 2 the best first H seed converts about 11 H nodes;
  the best first L seed converts under 1 L node (counting itself) and about 4 H nodes.
  Ratio far above a_H / a_L = 2.9, and spillover breaks the within-group assumption.
- More L seeds do not persuade more of L: non-seed rate falls from 0.075 to 0.051.

**Lambda sweep (converted).** Seeds in L: 0, 4, 6, 8, 9, 10 at lambda = 0.25, 0.5, 0.6,
0.7, 0.75, 0.85. Gradual, not abrupt. Earlier notes tied all-L to w_H = 0.81; that was wrong (see above).

**Parity vs maximin (`sections/comparison.tex`).**
- lambda = 2: worst rate within noise of worst-first (0.140, 0.137), total far lower
  (0.149, 0.271). k = 30: 0.355, 0.344; worst-first saves 13% more.
- lambda = 0.6: 0.306 total, 0.143 worst rate; at least matches worst-first.
- lambda = 0.75 is within noise of soft-min (0.213/0.152/0.075 vs 0.217/0.155/0.075).
- So "maximin is much cheaper than parity" holds for strict lambda only.
- Untested: looser lambda at other mu_L and k, and with many communities.

**Caveats.** One 500-node synthetic graph, one set of rumor seeds, one draw of biases,
top-degree candidates, 200 test worlds, so differences of about 0.01 are noise. No
error bars. Homophily, sigma and p were not varied.

## For Rachel to do

Full implementation details and every choice:
https://claude.ai/artifact/2kKTydPZnuP52xQZt6GSNZ

Review:
- [ ] `sections/simulation.tex` and `sections/comparison.tex`, against `figures/parity_*.md`
      and `figures/maximin_*.md`.
- [ ] `sections/parity.tex`: re-derive the lambda identity and Claims 2-3.
- [ ] With Matthew: `sections/comparison.tex`, and the "at lambda = 2" qualifiers added
      to his Experiments A and B text.

Confirm or change: sigma = 0.1, clipping, mu_H = 0.8, eps = 0.05, p = 0.08.

Before submitting:
- [ ] Direction 1 in `a1.tex` still says "acceptance rate a_g" and plans a homophily sweep.
- [ ] Page count: 8 including references (target about 5); cut something.
- [ ] Check the course AI-use policy and acknowledge AI help as it requires.
- [ ] Commit and push.
