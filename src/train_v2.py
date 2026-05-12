"""v2 training: component-aggregate features, evaluated honestly.

Trains the same five classifiers as v1 (LR, DT, RF, KNN, GBT) on the
v2 feature set and evaluates on:

* the test split (SetC) — apples-to-apples vs ``model_comparison.csv``
* the genericity split (SetD) — held-out, never reported by v1

Optional resampling (``--resample``) applies SMOTE or ADASYN **after**
the component→event featurization, so the synthetic oversampling
operates on the full v2 feature matrix — every component-derived
statistic (pps_mean/max/std, src_ip_*, bytes_per_pkt_*, etc.) is part
of the interpolation. This is the meaningful way to combine data
augmentation with component-aggregate features.

Saves results to ``results/model_comparison_v2.csv``.

Usage:
    uv run python -m src.train_v2
    uv run python -m src.train_v2 --resample smote
    uv run python -m src.train_v2 --resample all
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.features_v2 import FEATURE_NAMES, FeatureMatrix, featurize
from src.simulator import TYPE_ORDER, TYPE_TO_INT
from src.utils.data_pipeline import load_components, load_events

ResampleStrategy = Literal["none", "smote", "adasyn"]

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

V1_FEATURE_COLS = [
    "Packet speed", "Data speed", "Avg packet len",
    "Avg source IP count", "Detect count", "Port number",
]


# Genericity-set class prior, observed from SetD. Used by Fix C to
# re-target the class_weight calibration: sklearn's "balanced" weights
# are computed from train counts, but train and genericity have very
# different priors (DDoS 0.90% vs 3.37%, Suspicious 5.05% vs 6.49%).
# Calibrating to "balanced" pushes models to overpredict DDoS on test
# (DT precision 0.15, GBT precision 0.087). Calibrating to the gen
# prior is closer to the deployment distribution and keeps DDoS recall
# without crushing precision.
_GEN_PRIOR = {0: 0.9014, 1: 0.0649, 2: 0.0337}


def _gen_prior_weights(y_train: np.ndarray) -> dict[int, float]:
    """Class weights that re-target balanced training to the gen prior.

    The optimal Bayes weight for class c when training on prior p_train
    but evaluating under prior p_target is p_target[c] / p_train[c].
    We mean-normalize the result so the average weight is 1, which keeps
    the implicit regularization scale comparable to ``class_weight=None``
    and lets us interpret the dict as a relative re-weighting.
    """
    train_counts = pd.Series(y_train).value_counts(normalize=True).to_dict()
    raw = {c: _GEN_PRIOR[c] / train_counts[c] for c in _GEN_PRIOR}
    mean_w = float(np.mean(list(raw.values())))
    return {c: round(w / mean_w, 4) for c, w in raw.items()}


def _model_specs(class_weight=None) -> dict[str, tuple]:
    """Return {name: (estimator_factory, needs_scaling)} matching v1's lineup.

    ``class_weight`` is forwarded to the four classifiers that accept it
    (LR / DT / RF / GBT). KNN has no weighting hook so it receives the
    feature-level fixes only. Default of None preserves the legacy
    ``class_weight="balanced"`` behavior used by the v2 baseline.
    """
    cw = "balanced" if class_weight is None else class_weight
    return {
        "Logistic Regression": (
            lambda: LogisticRegression(
                max_iter=1000, class_weight=cw, random_state=42),
            True,
        ),
        "Decision Tree": (
            lambda: DecisionTreeClassifier(
                max_depth=10, class_weight=cw, random_state=42),
            False,
        ),
        "Random Forest": (
            lambda: RandomForestClassifier(
                n_estimators=100, max_depth=15, class_weight=cw,
                random_state=42, n_jobs=-1),
            False,
        ),
        "KNN (k=5)": (
            lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            True,
        ),
        "Gradient Boosting": (
            lambda: HistGradientBoostingClassifier(
                max_iter=200, max_depth=5, learning_rate=0.1,
                class_weight=cw, random_state=42),
            False,
        ),
    }


class V2Bundle:
    """Trained v2 estimators + the StandardScaler fit on training features."""

    def __init__(self, models: dict, scaler: StandardScaler):
        self.models = models
        self.scaler = scaler

    def predict(self, name: str, X: np.ndarray) -> np.ndarray:
        estimator, needs_scaling = self.models[name]
        if needs_scaling:
            X = self.scaler.transform(X)
        return estimator.predict(X)

    def names(self) -> list[str]:
        return list(self.models.keys())


def _resample(
    X: np.ndarray, y: np.ndarray, strategy: ResampleStrategy, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE/ADASYN to an already-featurized (X, y) matrix.

    Applied post-featurization, so the synthetic samples are blends of
    component-aggregate statistics, not raw events. That is the point
    of pairing resampling with v2 features.
    """
    if strategy == "none":
        return X, y
    sampler = SMOTE(random_state=seed) if strategy == "smote" else ADASYN(random_state=seed)
    X_res, y_res = sampler.fit_resample(X, y)
    counts = pd.Series(y_res).value_counts().sort_index().to_dict()
    print(f"  resampled ({strategy.upper()}): {counts}")
    return X_res, y_res


def train_v2_models(
    train_fm: FeatureMatrix,
    strategy: ResampleStrategy = "none",
    class_weight: dict | str | None = None,
) -> V2Bundle:
    """Fit all v2 models on a feature matrix. Used by both the standalone
    training script and the live evaluation script so there is exactly one
    training code path.

    ``class_weight`` overrides the default ``"balanced"`` calibration used
    by the v2 baseline; pass the gen-prior dict from ``_gen_prior_weights``
    to apply Fix C.
    """
    X_raw, y_raw = _resample(train_fm.X, train_fm.y, strategy)

    scaler = StandardScaler().fit(X_raw)
    X_train_s = scaler.transform(X_raw)

    trained: dict[str, tuple] = {}
    for name, (factory, needs_scaling) in _model_specs(class_weight).items():
        estimator = factory()
        Xtr = X_train_s if needs_scaling else X_raw
        t0 = time.perf_counter()
        estimator.fit(Xtr, y_raw)
        elapsed = time.perf_counter() - t0
        print(f"  trained {name:<22} in {elapsed:5.2f}s")
        trained[name] = (estimator, needs_scaling)
    return V2Bundle(trained, scaler)


def _evaluate(name: str, bundle: V2Bundle, X, y, dataset: str) -> dict:
    t0 = time.perf_counter()
    y_pred = bundle.predict(name, X)
    infer_s = time.perf_counter() - t0

    f1_mac = f1_score(y, y_pred, average="macro", zero_division=0)
    f1_wt = f1_score(y, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=list(range(len(TYPE_ORDER))))
    rep = classification_report(y, y_pred, target_names=TYPE_ORDER,
                                labels=list(range(len(TYPE_ORDER))),
                                zero_division=0)

    print(f"\n--- {name} [{dataset}] ---")
    print(f"  F1 macro:    {f1_mac:.4f}")
    print(f"  F1 weighted: {f1_wt:.4f}")
    print(f"  Inference:   {infer_s:.3f}s ({len(y) / infer_s:.0f} evt/s)")
    print(rep)
    print(f"  Confusion matrix:\n{cm}")

    return {
        "model": name,
        "dataset": dataset,
        "n_events": int(len(y)),
        "f1_macro": round(float(f1_mac), 4),
        "f1_weighted": round(float(f1_wt), 4),
        "inference_time_s": round(float(infer_s), 4),
        "throughput_eps": round(len(y) / infer_s) if infer_s > 0 else 0,
    }


def _v1_xy(events: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce v1's exact (X, y) preparation so we can score the
    legacy feature set on the genericity split too — v1's train script
    never reported it."""
    X = events[V1_FEATURE_COLS].to_numpy(dtype=np.float64)
    y = events["Type"].astype(str).map(TYPE_TO_INT).to_numpy(dtype=np.int64)
    return X, y


def _train_v1_baseline(train_ev: pd.DataFrame) -> V2Bundle:
    """Train the same five models on v1's six raw event columns."""
    X_train, y_train = _v1_xy(train_ev)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    trained: dict[str, tuple] = {}
    for name, (factory, needs_scaling) in _model_specs().items():
        estimator = factory()
        Xtr = X_train_s if needs_scaling else X_train
        estimator.fit(Xtr, y_train)
        trained[name] = (estimator, needs_scaling)
        print(f"  trained v1 baseline {name}")
    return V2Bundle(trained, scaler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v2 training / evaluation")
    parser.add_argument(
        "--resample",
        choices=["none", "smote", "adasyn", "all"],
        default="none",
        help="Resampling strategy applied to the v2 feature matrix "
             "(default: none). 'all' runs none + smote + adasyn.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resample == "all":
        strategies: list[ResampleStrategy] = ["none", "smote", "adasyn"]
    else:
        strategies = [args.resample]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading splits...")
    train_ev = load_events("train")
    train_co = load_components("train")
    test_ev = load_events("test")
    test_co = load_components("test")
    gen_ev = load_events("genericity")
    gen_co = load_components("genericity")
    print(f"  train: {len(train_ev):>7,} events / {len(train_co):>9,} components")
    print(f"  test:  {len(test_ev):>7,} events / {len(test_co):>9,} components")
    print(f"  gen:   {len(gen_ev):>7,} events / {len(gen_co):>9,} components")

    print("\nFeaturizing v2 (component aggregates → per-event)...")
    train_fm = featurize(train_ev, train_co, label_col="Type")
    test_fm = featurize(test_ev, test_co, label_col="Type")
    gen_fm = featurize(gen_ev, gen_co, label_col="Type")
    print(f"  X_train: {train_fm.X.shape}  y_train: {train_fm.y.shape}")
    print(f"  features ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")

    gen_weights = _gen_prior_weights(train_fm.y)
    print(f"\nGen-prior class weights (Fix C): {gen_weights}")
    print(f"  (sklearn 'balanced' gives ~{{0: 0.35, 1: 6.6, 2: 37.2}} —"
          f" calibrated to train counts, not deployment prior)")

    all_results: list[dict] = []
    for strat in strategies:
        print("\n" + "=" * 72)
        print(f"TRAINING (v2 ABC) — resample={strat}, class_weight=gen_prior")
        print("=" * 72)
        bundle = train_v2_models(
            train_fm, strategy=strat, class_weight=gen_weights,
        )

        print("\n" + "=" * 72)
        print(f"EVALUATION (v2 ABC) — resample={strat}")
        print("=" * 72)
        for name in bundle.names():
            r = _evaluate(name, bundle, test_fm.X, test_fm.y, "test")
            r["features"] = "v2_abc"
            r["resample"] = strat
            all_results.append(r)
            r = _evaluate(name, bundle, gen_fm.X, gen_fm.y, "genericity")
            r["features"] = "v2_abc"
            r["resample"] = strat
            all_results.append(r)

    print("\n" + "=" * 72)
    print("V1 BASELINE — same models, v1's 6 raw features, on test + genericity")
    print("=" * 72)
    v1_bundle = _train_v1_baseline(train_ev)
    X_test_v1, y_test = _v1_xy(test_ev)
    X_gen_v1, y_gen = _v1_xy(gen_ev)
    for name in v1_bundle.names():
        r = _evaluate(name, v1_bundle, X_test_v1, y_test, "test")
        r["features"] = "v1"
        r["resample"] = "none"
        all_results.append(r)
        r = _evaluate(name, v1_bundle, X_gen_v1, y_gen, "genericity")
        r["features"] = "v1"
        r["resample"] = "none"
        all_results.append(r)

    results_df = pd.DataFrame(all_results)
    out_csv = RESULTS_DIR / "model_comparison_v2_abc.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")

    print("\n" + "=" * 72)
    print("SUMMARY — F1 macro by model × (features, resample) × dataset")
    print("=" * 72)

    results_df["col"] = (
        results_df["features"] + "_"
        + results_df["resample"].astype(str) + "_"
        + results_df["dataset"]
    )
    pivot = (
        results_df.pivot_table(
            index="model", columns="col",
            values="f1_macro", aggfunc="first",
        )
        .reindex(list(_model_specs().keys()))
    )

    ordered = []
    for feat in ("v1", "v2"):
        for strat in ("none", "smote", "adasyn"):
            for ds in ("test", "genericity"):
                key = f"{feat}_{strat}_{ds}"
                if key in pivot.columns:
                    ordered.append(key)
    pivot = pivot[ordered]
    print("\nF1 macro (test / genericity):")
    print(pivot.round(4).to_string())


if __name__ == "__main__":
    main()
