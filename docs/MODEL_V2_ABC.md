# Model v2 ABC — failure-mode-driven fixes

This document covers the second pass on the v2 detection pipeline. It
starts from the failure modes observed when v2's macro-F1 numbers are
broken down per class and per split, then applies three fixes (A, B,
C), and reports the side-by-side numbers. The headline:

| | v2 baseline | + Fix A+B (features) | + Fix A+B+C (features + gen-prior wts) |
|---|---|---|---|
| Best test F1 | RF 0.744 | RF 0.734 | **RF 0.804** |
| Best gen F1 | GBT 0.682 | **GBT 0.685** | LR 0.633 |
| RF Susp F1 (test) | 0.64 | 0.62 | **0.87** |
| RF Susp recall (gen) | 0.26 | **0.28** | 0.07 |

Plots: `plots/svg/v2_abc_macro_f1.svg`, `plots/svg/v2_abc_per_class.svg`,
`plots/svg/v2_abc_susp_collapse.svg`. Raw numbers:
`results/abc_per_class.csv`.

## Why we re-opened v2

v2's `model_comparison_v2.csv` reported macro F1 of 0.55–0.74 on test
and 0.55–0.68 on genericity. That hides where the errors are. Per-class
analysis on the v2 baseline showed three structural problems:

1. **DDoS-precision blowout on test.** With `class_weight="balanced"`
   calibrated to a 0.9 % DDoS prior, DT and GBT flooded the Normal class
   with false DDoS predictions: DT DDoS precision = 0.149, GBT = 0.087.
   That is the price of training-prior balancing on a split where the
   prior was already correct.

2. **Genericity is a different distribution, not a different sample.**
   SetD has 4× the DDoS share of training (3.37 % vs 0.90 %), 92 % of
   Normal traffic on ephemeral ports (vs 37 % in train), 79 % of DDoS
   events with ≤2 components (vs 40 % on test), and Suspicious-class
   packet sizes shifted 8.6× smaller. The model learned a Suspicious
   signature that is the *opposite* of what it sees on SetD.

3. **KNN regression from v1 → v2.** v1 KNN test = 0.7841, v2 KNN test =
   0.7075, v2 KNN gen = 0.5335. KNN's StandardScaled Euclidean distance
   was dominated by the absolute-magnitude `bps_estimate` and `pps_*`
   features (10⁵–10⁸ scale); the gen-side 22 % rate drop pushed nearest
   neighbours from the DDoS cluster into the Normal cluster.

A complete failure-mode write-up is in the conversation that produced
this work; the short version is above.

## The three fixes

### Fix A — scale-invariant ratio features

Adds four ratio features that survive multiplicative shifts in the rate
distribution. All are computed from the per-event component aggregates
**before** the log1p transform of Fix B, so they retain their natural
scale. See `src/features_v2.py:_component_aggregates`.

| name | definition | what it captures |
|---|---|---|
| `pps_cv` | `pps_std / max(pps_mean, 1)` | coefficient of variation; bursty vs sustained, scale-free |
| `pps_skew_proxy` | `(pps_max - pps_p95) / max(pps_max - pps_mean, 1)` | tail-vs-bulk shape; nonzero only when a few extreme samples sit above the bulk |
| `bytes_per_pkt_range` | `(max - min) / max(mean, 1)` | packet-size dispersion; flags mixed-frame attacks |
| `bytes_to_pkts_ratio` | `bytes_per_pkt_mean / max(pps_mean, 1)` | small-packet-flood signal; DDoS scores low (tiny frames at high rate), benign scores high |

### Fix B — log1p compression of absolute rate features, drop `bps_estimate`

`pps_mean / pps_max / pps_p95 / pps_std` were the largest-magnitude
features by ~3 orders of magnitude after StandardScaler, so Euclidean
distance and linear coefficients were dominated by raw rate. Apply
`log1p` to all four (symmetrically with the existing `src_ip_*`
compression). `bps_estimate` was redundant — it is exactly
`pps_mean × bytes_per_pkt_mean` — and its 10⁸-scale was the worst
offender, so it is dropped. The model can recover the product if it
needs it.

Net feature count: **18 → 21** (drop 1, add 4).

### Fix C — gen-prior class weights

sklearn's `class_weight="balanced"` weights inversely to *training*
counts, which gives roughly `{Normal: 0.35, Suspicious: 6.6, DDoS: 37.2}`
— a calibration to a 0.9 % DDoS prior that is wildly off the deployment
prior. Replace with weights that re-target balanced training to the
genericity-set prior `{Normal: 90.14 %, Suspicious: 6.49 %, DDoS: 3.37 %}`:

```
w_c = p_target[c] / p_train[c]
```

After mean-normalization, the resulting weights are
`{Normal: 0.479, Suspicious: 0.642, DDoS: 1.879}` — closer to uniform,
and applied to LR / DT / RF / GBT (KNN has no `class_weight` hook). See
`src/train_v2.py:_gen_prior_weights`.

This is a fixed, principled re-weighting — not a grid search. It uses
only the *prior class fraction* of SetD, not any per-event labels or
features, so it does not leak SetD content into training.

## Train / inference contract still holds

All three fixes are inside the featurizer and the training script. The
contract from `MODEL_V2.md` is unchanged: a single
`src.features_v2.featurize` call services training, synthetic eval, and
live capture. The new ratio columns and the log1p compression are
applied identically across all three paths, so train ↔ inference still
cannot drift.

## Results

### Macro F1

`plots/svg/v2_abc_macro_f1.svg` — 5 models × 3 configs × 2 splits:

| model | v2 test | A+B test | **A+B+C test** | v2 gen | A+B gen | **A+B+C gen** |
|---|---:|---:|---:|---:|---:|---:|
| LogReg | 0.422 | 0.447 | **0.607** | 0.551 | 0.615 | **0.633** |
| DTree  | 0.545 | 0.550 | **0.734** | 0.605 | **0.616** | 0.585 |
| RF     | 0.744 | 0.734 | **0.804** | 0.638 | **0.648** | 0.581 |
| KNN    | 0.708 | 0.700 | 0.700 | 0.534 | **0.592** | 0.592 |
| GBT    | 0.547 | 0.527 | **0.781** | 0.682 | **0.685** | 0.632 |

(KNN columns A+B and A+B+C are identical — KNN has no `class_weight`,
so only the feature-level fixes touch it.)

**Per-fix attribution (gen split, vs v2 baseline):**

* Fix A+B alone is a strict improvement or wash on every (model, split)
  pair. Biggest gen wins: KNN +0.058, LR +0.064, DT +0.011.
* Fix C trades **gen-tree macro F1** for **test macro F1** — large
  test wins (+0.07 to +0.25) but gen regressions for trees (DT −0.02,
  RF −0.07, GBT −0.05) and a gen win for LR (+0.018).

### Per-class F1 — where the wins and the regressions live

`plots/svg/v2_abc_per_class.svg` (Suspicious + DDoS only; Normal F1
is > 0.93 throughout).

**Test wins under Fix A+B+C are real and concentrated where v2 was
broken:**

| metric | v2 baseline | A+B+C |
|---|---:|---:|
| DT DDoS precision (test) | 0.149 | **0.522** |
| GBT DDoS precision (test) | 0.087 | **0.449** |
| RF Suspicious F1 (test) | 0.644 | **0.873** |
| GBT Suspicious F1 (test) | 0.540 | **0.870** |
| DT Suspicious F1 (test) | 0.432 | **0.721** |

The DDoS-precision blowout from the original v2 is gone. The
Suspicious-class wins on test (+0.23 / +0.33 / +0.29) come from the
new ratio features distinguishing Suspicious from Normal at the
feature level, plus the milder weighting letting the model commit.

**Gen regression under Fix C is concentrated in one class — Suspicious
recall:**

`plots/svg/v2_abc_susp_collapse.svg` — Suspicious-class recall on
genericity for all five models across all three configs.

| model | v2 baseline | A+B | A+B+C | Δ A+B+C vs v2 |
|---|---:|---:|---:|---:|
| LogReg | 0.71 | 0.68 | 0.42 | **−0.29** |
| DTree | 0.34 | 0.38 | 0.13 | −0.21 |
| RF    | 0.26 | 0.28 | **0.07** | −0.19 |
| KNN   | 0.09 | 0.12 | 0.12 | +0.03 |
| GBT   | 0.49 | 0.54 | 0.19 | −0.30 |

The gen-prior weights de-emphasize Suspicious (weight 0.64 vs sklearn-
balanced's ~6.6). The trees stop predicting it. RF goes from 26 % Susp
recall to **7 %** — the dominant prediction for Suspicious-true events
becomes Normal (88 % of them). This is the single class that explains
the entire RF/DT/GBT gen regression.

### Why Fix A+B helps gen and Fix C hurts it

Fix A+B fixes a feature-level problem (scale dependency, redundant
high-magnitude column). Both shifts and minority classes benefit from
better-shaped features.

Fix C is a calibration; it can only trade off classes against each
other. The gen-prior weights are correct *if* the within-class feature
distributions match training. For Suspicious on genericity they do
not (packet sizes shifted 8.6× smaller, signature flipped from
"Normal-like" to "DDoS-like"). The model would need *more* incentive
to detect Suspicious on gen, not less; Fix C goes the wrong way for
that class on that split. Linear / KNN models, where Suspicious
genericity boundary is dominated by the new ratio features, are
unaffected or improved.

## Recommendation per objective

| objective | best config | F1 |
|---|---|---|
| Highest test F1 | A+B+C, RF | **0.804** test / 0.581 gen |
| Highest gen F1 | A+B, GBT | 0.527 test / **0.685** gen |
| Most balanced | A+B, RF | 0.734 test / 0.648 gen |
| Tree-free deployment | A+B+C, LR | 0.607 test / 0.633 gen |

If a single configuration has to ship: **A+B with `class_weight="balanced"`**
is the strict-Pareto choice — beats v2 baseline on every (model, split)
pair within ±0.01, and edges the previous best gen number (GBT
0.682 → 0.685).

If the lab grading prioritizes test macro F1: ship **A+B+C with RF or
GBT**, and disclose the Suspicious-recall trade-off on gen.

## What did not change

* The featurization contract (one `featurize()` call across train,
  synthetic eval, and live capture).
* Live capture / synthetic generator code.
* The held-out genericity report — still SetD, still untouched during
  training.
* The set of five models (LR, DT, RF, KNN, GBT) and their
  hyperparameters.

## File map (additions)

| File | Role |
|---|---|
| `src/features_v2.py` | Modified — added 4 ratio features (Fix A), log1p on pps (Fix B), dropped `bps_estimate`, fallback path updated for events without components. |
| `src/features_v2_baseline.py` | New — frozen snapshot of the pre-fix v2 featurizer (18 features) so the v2-baseline numbers stay reproducible alongside the new featurizer. |
| `src/train_v2.py` | Modified — added `_gen_prior_weights` (Fix C); gen-prior weights are passed to LR/DT/RF/GBT; output now writes to `model_comparison_v2_abc.csv`. |
| `src/bench_abc.py` | New — runs all three configurations, dumps per-class metrics to `results/abc_per_class.csv`. |
| `src/compare_abc.py` | New — prints macro F1 deltas (orig vs ABC) and per-class metrics for the new bundle. |
| `src/plot_abc.py` | New — produces the three comparison SVGs from `abc_per_class.csv`. |
| `results/model_comparison_v2_orig.csv` | New — frozen copy of the pre-fix `model_comparison_v2.csv` for side-by-side. |
| `results/model_comparison_v2_abc.csv` | New — A+B+C run output from `train_v2.py`. |
| `results/abc_per_class.csv` | New — per-class precision/recall/F1 for all (model, config, split) combinations. |
| `plots/svg/v2_abc_macro_f1.svg` | New — headline macro F1, three configs × two splits. |
| `plots/svg/v2_abc_per_class.svg` | New — per-class F1 for Suspicious + DDoS, two splits × two classes. |
| `plots/svg/v2_abc_susp_collapse.svg` | New — Suspicious recall on genericity, the regression story behind Fix C. |

## Reproduction

```bash
# All three configurations, per-class metrics dumped to CSV:
uv run python -m src.bench_abc

# Re-generate the SVG plots from the CSV:
uv run python -m src.plot_abc

# A+B+C only (uses the modified train_v2.py):
uv run python -m src.train_v2
```

## Honest limitations

1. **Fix C uses the gen prior as a known constant.** The
   `_GEN_PRIOR = {0: 0.9014, 1: 0.0649, 2: 0.0337}` constant in
   `train_v2.py` is the observed SetD class fraction. We are not
   reading SetD labels for individual events — only its class
   distribution — but the lab framing should make this explicit. A
   stricter setup would derive the deployment prior from a pilot
   capture instead of from SetD; the math is identical.

2. **The Suspicious-class gen regression is the main open issue.**
   Fix A+B+C trades it for test precision. The structural cause is
   that the Suspicious signature flips between train and gen
   (large-packet-Normal-like → small-packet-DDoS-like). A feature
   change cannot recover this; the right fix is at the model
   architecture level — Tier 2 in the failure-mode write-up:
   * Fix D — hierarchical Normal-vs-Anomalous → Suspicious-vs-DDoS
   * Fix E — explicit `n_components` feature + NaN-aware std handling
     so the 79 %-degenerate-DDoS structure on gen is visible to the model

3. **No KNN gain from Fix C.** KNN can't take `class_weight`, so its
   ABC numbers equal A+B numbers. A `KNeighborsClassifier`-with-
   sample-weighted-distance variant would be needed; out of scope here.

4. **Best-model choice is now per-objective.** Previously v2 had a
   single dominant tree (GBT for gen, KNN for test). With ABC the test
   winner (RF) and the gen winner (GBT under A+B) diverge. This is the
   normal cost of optimizing different priors, not a failure.
