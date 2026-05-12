# Per-class GMM cluster features

A small, targeted experiment on top of the v2 A+B feature set: fit one
Gaussian Mixture Model per training class on the 21 v2 features, and
add the soft membership probabilities (each event's likelihood under
each per-class sub-cluster) as new input columns. Goal: give the
classifier a "which sub-archetype of this class does this event look
like" signal that the single-class label cannot express.

## Why per-class clustering

The Suspicious class fails on genericity because its packet-size
signature flips between train (Normal-like, large packets) and
genericity (DDoS-like, small packets). One natural reading of that
finding is that **training Suspicious already contains both
sub-archetypes**, but each is too rare to dominate the class-level
statistics. If that is true, exposing per-class sub-cluster membership
to the classifier should let it learn "Suspicious sub-type A or B" as
a single decision instead of "Suspicious as a single distribution".

Per-class GMMs operationalize that. We fit one GMM on each class's
training rows, ask BIC how many sub-components there really are, then
score every event (regardless of true label) under every per-class
GMM. The resulting probability vector lives next to the v2 features.

This is feature engineering, not data augmentation in the SMOTE
sense — no synthetic rows are generated. The hypothesis is that the
existing rows carry enough sub-structure to help, but the classifier
cannot find it without the cluster prior baked in as features.

## Design

| choice | value | rationale |
|---|---|---|
| algorithm | Gaussian Mixture | soft probabilities — Suspicious is structurally borderline |
| K per class | BIC over {2, 3, 4, 5} | "how many genuine sub-types" question |
| covariance | diagonal | DDoS class has only 2,373 training rows; full cov is too many params |
| input scaling | StandardScaler before fit | diag-cov GMM assumes comparable axis variances |
| label use | only at fit time | inference is a deterministic transform; no train/serve skew |
| output features | `sum_c K_c` columns | concatenated per-class membership probabilities |

Implementation: `src/cluster_features.py` (`PerClassGMMFeaturizer`).
Fit/transform contract is identical to `StandardScaler`. Same
lifecycle inside the training pipeline.

In the run for this experiment, BIC selected K=5 (the cap) for all
three classes. That is a soft signal that even more sub-structure
might exist; raising the cap is a follow-up. Total cluster features:
3 × 5 = **15**, stacked onto the 21 v2 A+B features for a 36-column
input matrix.

## Results

`results/cluster_per_class.csv`,
`plots/svg/v2_cluster_macro_f1.svg`,
`plots/svg/v2_cluster_per_class.svg`.

### Macro F1 — Δ vs A+B (no Fix C, balanced class weights throughout)

| model | A+B test | + cluster | Δ test | A+B gen | + cluster | Δ gen |
|---|---:|---:|---:|---:|---:|---:|
| LogReg | 0.447 | **0.460** | +0.013 | 0.615 | 0.579 | **−0.036** |
| DTree  | 0.550 | 0.549 | −0.001 | 0.616 | **0.625** | +0.009 |
| RF     | 0.734 | **0.751** | +0.018 | 0.648 | 0.645 | −0.003 |
| KNN    | 0.700 | **0.707** | +0.008 | 0.592 | **0.616** | **+0.025** |
| GBT    | 0.527 | 0.522 | −0.006 | 0.685 | 0.678 | −0.007 |

Net: **modest, model-specific gains**. Best two improvements are KNN
gen (+0.025) and RF test (+0.018). Two regressions: LR gen (−0.036)
and small ones for GBT.

### Per-class — where the gains actually concentrate

`plots/svg/v2_cluster_per_class.svg`. Numbers in F1, A+B → A+B+cluster.

**DDoS class (test):** small but consistent improvement across most
models — DT 0.21 → 0.25, RF 0.60 → 0.63, KNN 0.39 → 0.41, LR 0.09 →
0.12. The cluster features distinguish DDoS sub-archetypes well
enough to nudge precision/recall up where the class was previously
ambiguous.

**Suspicious class — KNN is the winner:** KNN Suspicious recall on
genericity moved 0.117 → **0.185** (+0.068), the single largest
class-level improvement in the experiment. KNN had been the worst
model on this metric (Susp recall 9 % under v2 baseline, the original
failure mode this work was trying to address). Cluster features give
KNN a meaningful proxy for the within-class structure that distance
alone cannot find.

**Suspicious class — LR regresses on gen:** LR Suspicious recall
0.678 → 0.567 on genericity. The cluster features add 15 extra
columns and LR's L2 weights spread across them; with no class-weight
adjustment, this dilutes the earlier wins from the A+B ratio
features. Tree-based models are insensitive to extra columns; LR is
not.

## What this experiment said about the original hypothesis

The hypothesis was "Suspicious has hidden sub-archetypes that the
single label hides; per-class clustering will surface them and fix
the gen-Susp regression." Partial confirmation:

* The hypothesis works for **KNN gen Suspicious recall** — large win.
* It does not work for **GBT / RF / DT** because A+B ratio features
  already carried the relevant signal; adding cluster features is
  redundant for tree models that do their own multivariate splits.
* It backfires for **LR gen** because LR cannot afford 15 extra
  noisy linear coefficients without dimension-aware regularization.

A reasonable read: **per-class GMM features are a KNN-specific win
and a wash for trees**. They do not move the headline gen number
(GBT 0.685 still leads).

## Recommendation

* Ship **A+B + cluster features only with KNN** if KNN is the
  deployment model. KNN gen jumps 0.534 (v2) → 0.592 (A+B) →
  **0.616** (A+B+cluster).
* Keep GBT / RF on **A+B alone** — cluster features add cost
  (training time +50 %, inference +30 %) for no measurable benefit.
* Do not pair cluster features with LR.

If the goal is to push gen macro F1 above 0.685 with a single config:
this approach is not the lever. The next experiments to try (from the
original failure-mode write-up) are Tier 2:
* **Fix E (component-count guard)** — make the 79 %-degenerate-DDoS
  structure on gen visible to the model rather than hiding it as
  zeros.
* **Fix D (hierarchical Susp-vs-DDoS)** — splits the calibration
  problem so per-class Susp weighting can be tuned independently of
  DDoS.

## File map (additions)

| File | Role |
|---|---|
| `src/cluster_features.py` | `PerClassGMMFeaturizer` — fit/transform contract identical to `StandardScaler`. |
| `src/bench_cluster.py` | Runs A+B and A+B+cluster across the 5 models; saves per-class metrics. |
| `src/plot_cluster.py` | Generates the two SVGs from the per-class CSV. |
| `results/cluster_per_class.csv` | Per-class precision/recall/F1 for both configs. |
| `plots/svg/v2_cluster_macro_f1.svg` | Macro F1 comparison, test + gen. |
| `plots/svg/v2_cluster_per_class.svg` | Per-class F1 for Suspicious + DDoS, test + gen. |

## Reproduction

```bash
uv run python -m src.bench_cluster   # ~2 min: 12 GMM fits + 10 model trainings
uv run python -m src.plot_cluster    # ~5 s
```

## Honest limitations

1. **K=5 was the BIC-selected cap for every class.** That is suspicious
   — it likely means the K range was too narrow. Re-running with
   K∈{2, …, 10} could either reveal the "true" K or expose that BIC
   keeps adding components to fit training noise. Did not chase this
   in the current pass; flagging for follow-up.

2. **No regularization on cluster features for LR.** LR with 36
   features and balanced class weights spreads its coefficients more
   thinly than with 21 features. A targeted L1 penalty or pre-feature
   selection (drop cluster columns whose train mutual-information
   with the label is below threshold) would likely recover the LR-gen
   loss.

3. **Cluster features fit on the same training rows used downstream.**
   This is the standard pattern but it can over-fit slightly. The
   honest implementation is **out-of-fold cluster features**:
   K-fold-fit the GMMs on K-1 folds, transform the held-out fold,
   stitch together, then re-fit on full train for inference. ~6×
   slower; not done here.

4. **Per-class GMMs don't model gen-distribution shift directly.**
   The clusters are defined by training distribution. If gen has a
   sub-archetype that doesn't exist in training, no GMM component
   captures it and all per-class probabilities go low — which can
   actually be a useful "out-of-distribution" signal but isn't
   exploited here. Adding a max-of-soft-probabilities feature
   (essentially "how confident is *any* class GMM about this row")
   would give the classifier an OOD indicator at near-zero cost.
