# Wiring v2+cluster DDoS detector into the interactive UI

**Date:** 2026-05-26
**Status:** Approved — ready for implementation plan

## Goal

Make the FastAPI dashboard at `src/api/app.py` serve predictions from the v2 component-aggregate pipeline with per-class GMM cluster features, instead of the v1 raw-event pipeline it currently uses. The `features_v2.featurize` contract already binds training, synthetic eval, and live eval; the UI becomes the fourth caller of the same contract.

## Why now

`src/api/model_registry.py` and `src/api/stream_engine.py` still import the v1 `FEATURE_COLS` and train the v1 six-model lineup at startup. `src/features_v2.py` documents that v1 has train/serve skew on `Detect count` and `Avg source IP count`, and `src/cluster_features.py` (most recent addition) materially improves the Suspicious-class generalization gap. The interactive demo should reflect the latest model.

## Scope decisions

| Decision | Choice |
|---|---|
| Variant | v2 + per-class GMM cluster features (single variant) |
| v1 fate | Removed from the UI entirely |
| Synthetic scenarios | Kept; we use the generator's existing `generate_components(events)` |
| Live scenarios | Kept; all four live scenarios reuse `LIVE_SCENARIOS` from `src/live_capture.py` (already imported by `live_capture_v2.py`) |
| Default model | v2 Random Forest (top v2 entry in `results/model_comparison_v2.csv` test split, f1_macro 0.7442) |
| UI / SSE schema | Unchanged |

## Architecture

State that moves into `ModelRegistry`:

1. `PerClassGMMFeaturizer` (`src/cluster_features.py`) — fit on raw v2 features `(X_v2, y)`; BIC-selected K per class over {2, 3, 4, 5}. The featurizer encapsulates its own internal `StandardScaler` for GMM input (see `cluster_features.py`).
2. `StandardScaler` fit on the **augmented** matrix `X_aug = hstack(X_v2, gmm.transform(X_v2))`. Used at predict time only by the scale-sensitive classifiers (LR, KNN); the tree-based classifiers (DT, RF, HistGBT) see raw `X_aug`. This mirrors `_train_and_eval` in `src/bench_cluster.py`.
3. The five v2 classifiers (LogisticRegression, DecisionTree, RandomForest, KNN, HistGradientBoosting) — wrapped with the existing `_ScaledPredictor` helper when they need scaling, otherwise stored raw. Each predictor accepts an `X_aug` row and returns the class label.

All three are serialized into `/app/models/` alongside a `manifest.joblib` that carries a `feature_version` field so v1 caches are rejected on next boot.

The registry exposes one boundary method for the engine:

```python
def featurize_window(events: pd.DataFrame, components: pd.DataFrame) -> np.ndarray:
    """Returns the augmented matrix X_aug ready for current_model.predict()."""
```

`current_model.predict(X_aug)` internally applies the post-augmentation `StandardScaler` for scale-sensitive models via `_ScaledPredictor`. The engine has no v2-specific knowledge beyond producing the (events, components) pair.

## Components & changes

### `src/api/model_registry.py` — rewrite

- Drop `FEATURE_COLS` and `_build_models` (v1).
- New `load_or_train()`:
  - `events = load_events("train")`, `components = load_components("train")`.
  - `fm = features_v2.featurize(events, components, "Type")` → `X, y` (call signature matches `bench_cluster.py:103`).
  - Fit `PerClassGMMFeaturizer(K_grid=(2,3,4,5), cov_type="diag")` on `(X, y)`. The featurizer handles its own internal scaling.
  - `X_aug = np.hstack([X, gmm.transform(X)])`.
  - Fit a `StandardScaler` on `X_aug` → `aug_scaler`.
  - Train the five classifiers (mirroring `bench_cluster._make_models`): LR and KNN are wrapped with `_ScaledPredictor(model, aug_scaler)` and fit on `aug_scaler.transform(X_aug)`; DT, RF, HistGBT fit on raw `X_aug`.
  - Persist: `gmm_featurizer.joblib`, `aug_scaler.joblib`, one `<model>.joblib` per classifier (the `_ScaledPredictor` wrapper serializes the inner scaler reference), `manifest.joblib = {names, default: "Random Forest", feature_version: "v2+cluster"}`.
- New `featurize_window(events, components)`:
  - `fm = features_v2.featurize(events, components, "Type")` → `X`.
  - Clamp NaN aggregates (empty-components edge case) to 0.
  - Return `np.hstack([X, gmm.transform(X)])` — the post-augmentation scaling, if needed, happens inside `_ScaledPredictor.predict`.
- Cache invalidation: if `manifest["feature_version"] != "v2+cluster"`, ignore the cache and retrain.

### `src/api/stream_engine.py` — adapt

- Remove `from src.api.model_registry import FEATURE_COLS`.
- `_run_synthetic`:
  - After `gen.generate_stream(...)`, also call `components = gen.generate_components(events)` once.
  - In `_process_windows`, slice `events` by `Start time` window **and** `components` by `Attack ID ∈ window_events["Attack ID"]`.
  - Replace the inline `X = window_events[FEATURE_COLS]...` block in `_process_and_publish_window` with `X_arr = registry.featurize_window(window_events, window_components)`.
- `_run_live`:
  - Switch from `LiveTrafficGenerator` to `LiveTrafficGeneratorV2`.
  - Drive its new streaming method (see next section) instead of `pop_events()`. Each yielded `(events_df, components_df)` becomes one window passed to `_process_and_publish_window`.
- `_process_and_publish_window`: take `window_components` as a new argument; otherwise unchanged. Stats currently computed off `window_events["Packet speed" / "Data speed" / "Port number"]` keep working (those columns still exist on the events frame in both synthetic and v2-live paths).

### `src/live_capture_v2.py` — add streaming variant

`LiveTrafficGeneratorV2.run_scenario` currently builds windows one at a time in a loop, then returns everything at the end. Add a sibling generator method:

```python
def stream_scenario(self, scenario: LiveScenario) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield (events_df, components_df) for each window as it closes."""
```

Same per-window construction; instead of appending the `CapturedWindow` to a list, materialize it to `(events_df, components_df)` immediately and `yield`. `run_scenario` can stay as a thin wrapper that consumes the iterator.

In the engine, run this iterator in a background thread (matching the current v1 pattern); a thread-safe queue passes the yielded tuples to the asyncio loop.

### `src/api/app.py` — no changes

The lifespan, health endpoint, model-listing, and SSE endpoints work as-is. `ModelRegistry` exposing the same `model_names` / `select(name)` / `current_name` / `current_model` API keeps `app.py` ignorant of the swap.

### `src/api/static/index.html` — no changes

The model dropdown will display v2 model names (which happen to have the same labels as v1: "Logistic Regression", "Decision Tree", "Random Forest", "KNN (k=5)", "Gradient Boosting" — minus the "Baseline (majority)" which v2 doesn't ship). No UI logic depends on the name strings.

## Data flow (per window)

```
synthetic: TrafficGenerator → events_df  +  components_df
                                     │
live:      LiveTrafficGeneratorV2.stream_scenario → events_df + components_df
                                     │
                                     ▼
                          features_v2.featurize → X_v2
                                     │
                          gmm.transform (internal scaling) → P_clusters
                                     │
                       hstack(X_v2, P_clusters) → X_aug
                                     │
                                     ▼
                          model.predict(X_aug) → y_pred
                          (LR/KNN apply aug_scaler internally via _ScaledPredictor)
                                     │
                                     ▼
                  _process_and_publish_window stats → Redis → SSE
```

## Error handling & edge cases

- **Empty window** — already skipped upstream in `_process_windows`; unchanged.
- **Events with empty components** (live race during a brief phase) — `featurize` returns NaN aggregates. The registry's `featurize_window` clamps NaN → 0 at the boundary so downstream code sees a finite matrix.
- **Attack ID join sanity** — add a one-line assertion at registry init that `events["Attack ID"]` is a unique key and `components["Attack ID"]` is non-empty. Synthetic generator already satisfies this; the assertion fails loudly if a future generator change breaks the contract.
- **Stale v1 cache** — `manifest["feature_version"]` gate forces retrain; old v1 joblibs on disk are not deleted but become inert.

## Cache & startup cost

Cold start additionally:
- Fits one `StandardScaler` on the v2 training matrix.
- Runs a BIC sweep over K ∈ {2, 3, 4, 5} per class (3 classes → up to 12 GMM fits at startup) on subsampled training data (same 50k cap as today).
- Trains the five v2 classifiers.

All inside the existing background-thread training; `/api/health` continues to return `"loading"` until done. On a warm cache (joblibs + manifest present and `feature_version == "v2+cluster"`), startup is unchanged.

## Testing

- **Smoke (manual):** `docker compose up`, `GET /api/health` → ready, `GET /api/models` returns the v2 five-model list with `"Random Forest"` active, start `low_rate_ddos` and `live_syn_flood`, watch SSE for non-zero accuracy on both. Synthetic accuracy should track the figures in `results/model_comparison_v2.csv`; live accuracy should track `results/live_evaluation_v2.csv`.
- **Featurize-window sanity (one assertion-style check):** call `registry.featurize_window(events_subset, components_subset)` on a hand-built (3-event, 9-component) input; assert shape `(3, n_v2_features + sum_K_c)` and no NaNs after the clamp.
- **Cache invalidation:** with a v1-style manifest on disk (no `feature_version` key), `load_or_train` must retrain rather than load.

## Out of scope (deliberately)

- No UI / SSE schema changes.
- No new scenarios; no porting of v1-only scenarios.
- No side-by-side v1↔v2 comparison in the live UI (offline `model_comparison*_v2*.csv` already covers that).
- No retraining trigger from the UI.
- No deletion of `src/live_capture.py` v1 module — `live_capture_v2.py` still imports `LIVE_SCENARIOS` / `HPING_ATTACKS` from it.

## Open questions

None at design time. Two will be answered empirically during the smoke test:
- Is the cold-start time with GMM fits still acceptable under the 50k-row training subsample?
- Does the v2 RF actually feel responsive in the demo, or do we want to swap the default to a faster-prediction v2 model after watching live latency?
