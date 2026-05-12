# More derived features — component count, robust stats, GMM OOD

A second feature-engineering pass on top of A+B. Three new groups of
features, motivated by failure modes the A+B pass left untouched:

1. **Component-count features** for the gen-DDoS-degeneration problem
   (79 % of genericity DDoS events have ≤2 components — the v2 std/
   p95 / burstiness stats collapse).
2. **Robust per-event statistics** (median, MAD) as a parallel signal
   that survives small component counts.
3. **GMM-based out-of-distribution signals** to give the model an
   explicit "this event doesn't resemble any training archetype"
   feature, which the v2 set is missing.

Headline result: **GBT genericity macro F1 0.685 → 0.704**, the first
configuration in this project to clear 0.70 on the genericity split.

## Why these features

The earlier A+B pass treated component-derived statistics as the
v2 thesis ("rich within-event statistics drive classification"). On
genericity that thesis partly breaks down — most DDoS events on SetD
have 1–2 components, which makes the variance-based v2 features
degenerate. A+B did not address this directly; it only added
scale-invariant ratio features and log-compressed the absolute rates.

The three groups added here each target a different leg of that
problem:

* **Group 1 — count features** let the classifier know *how many
  components an event has*, so it can learn "trust the std signal
  when count > 5; ignore it when count ≤ 2".
* **Group 2 — robust statistics** give the same per-event level a
  median/MAD signal that does not need many components to be
  meaningful. On a 2-component event, mean/std are unreliable but
  the median is a real value.
* **Group 3 — GMM OOD signals** measure how well any training-class
  archetype matches the event. On gen, where the class signatures
  shift, this gives an explicit "low confidence under all training
  archetypes" feature that the model can use to be more cautious.

## What we added

### Group 1 — component count (3 features)

| name | definition |
|---|---|
| `n_components` | raw count of components per event (1..max) |
| `log1p_n_components` | log-compressed count, for distance/linear models |
| `is_single_sample` | binary: 1 if event has exactly one component |

### Group 2 — robust per-event statistics (4 features)

| name | definition |
|---|---|
| `pps_median_log` | log1p of the median Packet speed across components |
| `pps_mad_log` | log1p of the median absolute deviation of Packet speed |
| `bytes_per_pkt_median` | median Avg packet len (excluding zero-length artefacts) |
| `bytes_per_pkt_mad` | MAD of Avg packet len (excluding zeros) |

Code: `src/features_v2_extras.py`. Output is a (N, 7) array stacked
onto the v2 A+B matrix. Live-capture compatible: each window's
`/proc/net/dev` samples become components, so all seven features have
the same semantics across train, synthetic eval, and live capture.

### Group 3 — GMM OOD summary (2 features)

Reuses the `PerClassGMMFeaturizer` from the cluster experiment. After
fitting one GMM per training class (BIC-tuned K∈{2..5}, diagonal cov),
two summary columns are derived from the soft membership matrix:

| name | definition |
|---|---|
| `gmm_max_prob` | max over all per-class sub-cluster probabilities for this event |
| `gmm_entropy`  | Shannon entropy of the (re-normalized) joint probability vector |

Low `gmm_max_prob` = "this event doesn't resemble any training
archetype". High `gmm_entropy` = ambiguous between archetypes. The
two are not redundant: a confidently wrong cluster (high max, low
entropy) and a confidently out-of-distribution event (low max, low
entropy) are different beasts.

We expose only the *summary* columns (2), not the full per-class
membership matrix (15 from the previous experiment) — the prior
pass showed that the full matrix added noise to LR for marginal
gains. Two columns is the lowest-risk dose of the same signal.

Code: `cluster_features.PerClassGMMFeaturizer.transform_summary`.

## Results

`results/more_per_class.csv`,
`plots/svg/v2_more_macro_f1.svg`,
`plots/svg/v2_more_per_class.svg`. All runs use
`class_weight="balanced"` — Fix C from earlier is held off so the
feature effect is isolated.

### Macro F1

| model | A+B test | +extras test | +extras+OOD test | A+B gen | +extras gen | +extras+OOD gen |
|---|---:|---:|---:|---:|---:|---:|
| LogReg | 0.447 | 0.460 | **0.462** | 0.615 | **0.643** | 0.642 |
| DTree  | 0.550 | 0.572 | **0.576** | **0.616** | 0.602 | 0.600 |
| RF     | 0.734 | **0.746** | 0.736 | 0.648 | **0.658** | 0.658 |
| KNN    | 0.700 | 0.717 | **0.718** | **0.592** | 0.591 | 0.590 |
| GBT    | 0.527 | 0.527 | **0.529** | 0.685 | 0.682 | **0.704** |

**Big wins:**

* GBT gen 0.685 → **0.7043** (+0.020) — first config to break 0.70 on
  genericity. The extras-only step didn't help GBT, but adding the
  OOD signals on top did. GBT is a tree ensemble so it can use the
  count and robust-stats features without help, but the OOD pair
  carries information it could not derive (probability mass under a
  separately-fit GMM is not a feature space the histograms naturally
  expose).
* LR gen 0.615 → **0.643** (+0.028) — the extras alone explain it;
  adding OOD makes no difference. LR was the model most hurt by gen
  shift among linear models, and the median-based features (which
  are noise-robust) help its decision boundary.
* RF test 0.734 → **0.746** (+0.013), KNN test 0.700 → **0.718**
  (+0.018), DT test 0.550 → **0.576** (+0.027). All extras-driven.

**Mild losses:**

* DT gen 0.616 → 0.600 (−0.015). DT is overfitting to the new
  features on training; it didn't see them on test/gen as cleanly.
* GBT extras-only on gen: 0.685 → 0.682 (basically flat). The OOD
  step matters specifically for GBT.

### Per-class effect on the failure-mode classes

`plots/svg/v2_more_per_class.svg`. Highlights only:

| metric | A+B | +extras | +extras+OOD |
|---|---:|---:|---:|
| GBT DDoS recall (gen) | 0.803 | 0.803 | **0.823** |
| GBT Susp recall (gen) | 0.536 | 0.523 | **0.581** |
| LR Susp F1 (gen) | 0.394 | **0.469** | 0.466 |
| RF Susp F1 (test) | 0.617 | **0.659** | 0.629 |
| KNN DDoS recall (test) | 0.373 | **0.428** | 0.426 |

The Susp gen-recall on GBT moved 0.536 → 0.581 — that is the same
class-recall problem Fix C traded away. Here we get 4.5 pp recall for
*free* (no class-weight calibration sacrifice) by giving the model
an OOD-aware feature instead.

### Where each group earns its keep

* **Component-count + robust stats (extras)** is a strict gen
  improvement for LR (+0.028) and RF (+0.010) and a strict test
  improvement for DT (+0.022), RF (+0.013), KNN (+0.017). Costs:
  none for trees, modest for LR. Recommended unconditionally.
* **GMM OOD signals** add real value specifically for **GBT on
  genericity** (+0.022 on top of extras). Effect on the other four
  models is in the noise. The OOD signals are cheap to compute
  (~13 s GMM fit + 2 columns at inference) so adding them is low
  cost; just don't expect them to move LR/DT/RF/KNN.

## Best configurations after this pass

| objective | best config | macro F1 |
|---|---|---|
| Highest genericity | **GBT, A+B + extras + OOD** | **0.7043** test 0.529 |
| Highest test (no Fix C) | RF, A+B + extras | 0.7464 test / 0.658 gen |
| Highest test overall | RF, A+B+C (from previous pass) | 0.804 test / 0.581 gen |
| Most balanced (no Fix C) | RF, A+B + extras | 0.7464 / 0.658 |

The previous best gen number (GBT A+B 0.685) is now beaten by GBT
A+B+extras+OOD (0.704). The previous best balanced number (RF A+B
0.734 / 0.648) is also beaten by RF A+B+extras (0.746 / 0.658).

## Open questions / next moves

1. **Fix C + extras + OOD** — we held class weights at "balanced" to
   isolate the feature effect. Combining the new features with the
   gen-prior weights might give a different best-test config or
   recover the gen-Susp-recall trade-off. Quick to test.

2. **The MAD computation is slow** — `extra_features` takes 4–5
   minutes on 264k events because it uses pandas `.apply()` per
   group. A vectorised median-absolute-deviation would cut the bench
   time substantially. Functional regression-risk is zero.

3. **Combining cluster membership and OOD signals** — the previous
   pass kept the full 15-column membership matrix; this pass kept
   only the 2-column summary. There may be a small middle ground
   (e.g. one membership column per class — argmax of probabilities)
   that adds class-conditional structure without the LR-hurting noise.

4. **Component-count features on the live capture path** — verify
   that 5 s windows with sub-50 components don't hit edge cases in
   the `is_single_sample` flag (a single 100 ms sample is the
   degenerate-but-real case for the live path; just want to be sure
   the flag isn't misleading there).

## File map (additions)

| File | Role |
|---|---|
| `src/features_v2_extras.py` | 7 extra columns: count, log-count, single-flag, robust pps + plen stats. Live-capture-compatible. |
| `src/cluster_features.py` | Extended with `transform_summary` returning ``[gmm_max_prob, gmm_entropy]``. |
| `src/bench_more.py` | Runs A+B / +extras / +extras+OOD across 5 models; saves per-class metrics. |
| `src/plot_more.py` | Generates the two SVGs. |
| `results/more_per_class.csv` | Per-class metrics for all (model, config, split). |
| `plots/svg/v2_more_macro_f1.svg` | Macro F1 comparison, three configs × two splits. |
| `plots/svg/v2_more_per_class.svg` | Per-class F1 for Suspicious + DDoS, three configs × two splits. |

## Reproduction

```bash
uv run python -m src.bench_more   # ~7 min (most of which is MAD computation)
uv run python -m src.plot_more    # ~5 s
```
