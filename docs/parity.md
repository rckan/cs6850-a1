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
| Experiments A, B, C | `experiments/parity_*.py` | done, see Results (redraw: `--plot-only`) |
| Math write-up | `sections/parity.tex` | drafted; **Rachel to review** |
| Figures in paper | `sections/parity_figures.tex` (after the two-column body) | done |
| Simulation write-up | `sections/simulation.tex` | **Rachel + Matthew** |

Run everything: `pytest`, then `python -m experiments.parity_sweep_acceptance`
(same for `parity_sweep_budget`, `parity_pareto`). Total about 5 minutes. Add `--plot-only` to redraw figures from `results/`.

## Decisions (change these if you disagree)

1. **Two outcomes, both reported.** *Saved* = the rumor would reach the node
   with no truth seeds but doesn't with them (the papers' definition). It counts
   rejecting nodes that truth nodes shield. *Converted* = saved and ends up
   believing the truth. Every experiment runs both. **Open question for Rachel:**
   which one the paper's DP metric uses. See Results for why it matters.
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
   sigma = 0.1, candidates = top 50 by degree per group, lambda = 2, eps = 0.05.
   - p = 0.05 barely spreads (about 23 exposed nodes); p = 0.1 reaches about
     2/3 of the network. p = 0.08 exposes about 184 H / 42 L nodes.
   - mu_H = 0.8 rather than 1 so H nodes vary on both sides of the mean.
     sigma is 0.1. edit if needed
   - The pool was 30 at first, but at k = 30 DP greedy had used every L
     candidate. 50 > max k.
9. **Truth seeds count toward their group's rate** (as in FWRRS's saved-node
   count). **TODO:** L has only about 40 exposed nodes, so a
   method that seeds L directly raises L's rate mostly through the seeds
   themselves. At k = 30 (old pool), 11 of L's 16.5 converted nodes were seeds.
   Excluding seeds from the counts would measure persuasion only.

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

Per-node acceptance, mu_H = 0.8, sigma = 0.1. Figures and number tables are in
`figures/` (`parity_*.png`, `parity_*.md`; `parity_pof.md` has the
price-of-fairness table). Unless stated: k = 10, mu_L = 0.3 (realized mean
acceptance 0.808 in H, 0.280 in L; exposure-weighted means are the same to 3
decimals), test worlds. PoF = total greedy's total divided by the method's total.
An earlier run used a_H = 1 with group-level acceptance; its numbers are
superseded and differ in places (noted below).

**Saved outcome: acceptance matters little (shielding).**
- Total greedy's saved fraction falls only from 0.707 (mu_L = 0.8) to 0.689
  (mu_L = 0.3). Every method's DP gap is at most 0.12.
- DP greedy's PoF is about 1.06-1.11 at every mu_L. The ceiling formula (1.35 at
  mu_L = 0.3) does not apply to the saved rate.
- The lambda sweep trades total from 0.70 down to 0.58 to shrink the gap from
  0.11 to 0.03. Analytic n/a reaches a gap of 0.007 but saves only 0.57.

**Converted outcome: acceptance matters a lot.**
- Even with equal means (mu_L = 0.8), total greedy's gap is 0.145. L is smaller
  and sparser (mean degree 11 vs 19), so it is harder to reach regardless of
  bias.
- At mu_L = 0.3, total greedy puts 0 of 10 seeds in L: gap 0.35 (r_H 0.43,
  r_L 0.08).
- DP greedy puts all 10 seeds in L and reaches near-exact parity: gap 0.011,
  total 0.149, PoF 2.47. With a_H = 1, the earlier run could not get the gap
  below 0.2. Lower H acceptance means less spillover into H from the L seeds.
- eps-DP greedy (eps = 0.05, 9 seeds in L) is now feasible: gap 0.040, total 0.157.
- Analytic n/a (k_g ∝ n_g / mean a_g, 6 seeds in L) has PoF 1.19, below the
  formula's 1.35. The formula overstates the "all seeds to H" benchmark, which
  in practice has diminishing returns.
- **How DP greedy gets parity (checked directly, k = 30):** all 30 seeds are in
  L, and 76% of L's converted rate is the seeds themselves. L's non-seed
  converted rate is 0.087, and H is converted only through spillover from the L
  seeds (0.355). At k = 10, DP greedy's L non-seed rate (0.051) is *lower* than
  total greedy's (0.077). So parity comes from holding H back and counting L
  seeds, not from persuading more of L. This is the clearest form of leveling
  down here, and it depends on Decision 9 (whether seeds count).
- **Claim 2's plateau is not observed for k <= 30.** DP greedy's total keeps
  rising (0.15 at k = 10, 0.36 at k = 30) because each new L seed adds itself
  plus spillover. L's converted rate passes its mean acceptance (0.28) at
  k = 21, which the bound allows only through the seeds.
- **Claim 1's allocation does not match DP greedy.** It predicts k_g ∝ E_g / mean a_g,
  4 of 10 seeds in L at mu_L = 0.3 (the "Analytic n/a" baseline uses n_g and gets 6);
  DP greedy places all 10 in L for every mu_L <= 0.6 (5 of 10 at 0.8). This
  supersedes the earlier "partial support for Claim 1", which rested on the
  removed maximin baseline.
- **Saved vs converted on the same seeds:** total greedy's saved-run seeds (0 in L)
  save 0.60 of L's exposed nodes but convert 0.08. DP greedy's converted-run seeds
  score only about 0.26 on the saved outcome, vs 0.68 for total greedy's
  converted-run seeds.
- The converted bound held for every method, group, and k checked.
- On the lambda front, everything from lambda = 1 up lands at gap <= 0.03 and
  total about 0.15-0.17. There is a big jump between lambda = 0.5 (gap 0.26) and
  lambda = 1 (gap 0.007).
- DP greedy's PoF is not monotone in mu_L (2.47 at 0.3, 2.40 at 0.15). That is
  within noise for a greedy heuristic on one graph.

**Caveats.** One graph, one set of rumor seeds, one draw of biases, and greedy
limited to the top 50 nodes by degree per group. Test means use 200 worlds, so
differences of about 0.01 are noise. No error bars yet.

## For Rachel to do


Decide first (later steps depend on these):
- [ ] Saved or converted rate for the DP metric. Then keep both rows of Figs 1-2 or cut one.
- [ ] Whether truth seeds count toward their group's rate. If not, the
      evaluator must exclude them and all three experiments rerun (about 5 min).
- [ ] sigma (now 0.1), clip vs truncated normal vs Beta, and mu_H = 0.8,
      lambda = 2, eps = 0.05, p = 0.08.
- [ ] Claim 2: try early stopping (|S_P| <= k), or reword it as a prediction the
      simulation does not show and report the spillover finding instead.

Review:
- [ ] `sections/parity.tex` line by line: re-derive the bound and Claims 1-3.
- [ ] Notation vs the direction 1 paragraph in `a1.tex` (n_g vs E_g, a_g vs mean a_g,
      "saves" vs "converts").
- [ ] With Matthew: keep or cut the DP-vs-maximin comparisons in Claims 1-2.
- [ ] Figs 1-3 and captions (`sections/parity_figures.tex`) against `figures/parity_*.md`.
- [ ] The implementation doc above; flag any choice you disagree with.

Write:
- [ ] `sections/simulation.tex`: drafted; resolve its red TODOs and rewrite the takeaway.
- [ ] Direction 1 in `a1.tex`: describe the per-node bias model, and replace the
      "Pursue:" sentence with a pointer to the results.
- [ ] Add Matthew's maximin method to `METHODS` in `experiments/common.py` once
      it exists, and rerun.

Before submitting:
- [ ] Remove or comment out the working notes (they print after the paper).
- [ ] Check the page count (about 5) after Matthew's section; shrink figures
      (`width=0.82\textwidth`) or drop Fig 3 if over.
- [ ] Commit and push.
- [ ] Optional: error bars over several graph, rumor and bias seeds.
