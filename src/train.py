"""Train and evaluate simple classifiers for DDoS detection.

Trains on SCLDDoS2024 train split, evaluates on test split,
then runs simulation benchmarks on synthetic traffic.

Usage:
    uv run python -m src.train
    uv run python -m src.train --resample smote
    uv run python -m src.train --resample adasyn
    uv run python -m src.train --resample all
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.simulator import Simulator, TYPE_ORDER, TYPE_TO_INT
from src.synthetic import TrafficGenerator
from src.utils.data_pipeline import load_events

FEATURE_COLS = [
    "Packet speed", "Data speed", "Avg packet len",
    "Avg source IP count", "Detect count", "Port number",
]

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

ResampleStrategy = Literal["none", "smote", "adasyn"]


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df[FEATURE_COLS].values.astype(np.float64)
    y = df["Type"].map(TYPE_TO_INT).values
    return X, y


def resample(
    X: np.ndarray, y: np.ndarray, strategy: ResampleStrategy, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Apply oversampling to balance training data."""
    if strategy == "none":
        return X, y
    if strategy == "smote":
        sampler = SMOTE(random_state=seed)
    elif strategy == "adasyn":
        sampler = ADASYN(random_state=seed)
    else:
        raise ValueError(f"Unknown resample strategy: {strategy}")
    X_res, y_res = sampler.fit_resample(X, y)
    counts = pd.Series(y_res).value_counts().sort_index()
    print(f"  Resampled with {strategy.upper()}: {dict(counts)}")
    return X_res, y_res


def _build_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    X_train_s: np.ndarray,
    X_test_s: np.ndarray,
    strategy: ResampleStrategy,
) -> dict[str, tuple]:
    """Build model definitions for a given resampling strategy."""
    tag = f" [{strategy.upper()}]" if strategy != "none" else ""
    models: dict[str, tuple] = {}

    if strategy == "none":
        models[f"Baseline (majority){tag}"] = (
            DummyClassifier(strategy="most_frequent"),
            X_train, X_test,
        )

    models[f"Logistic Regression{tag}"] = (
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        X_train_s, X_test_s,
    )
    models[f"Decision Tree{tag}"] = (
        DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42),
        X_train, X_test,
    )
    models[f"Random Forest{tag}"] = (
        RandomForestClassifier(
            n_estimators=100, max_depth=15, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
        X_train, X_test,
    )
    models[f"KNN (k=5){tag}"] = (
        KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        X_train_s, X_test_s,
    )
    models[f"Gradient Boosting{tag}"] = (
        HistGradientBoostingClassifier(
            max_iter=200, max_depth=5, learning_rate=0.1,
            class_weight="balanced", random_state=42,
        ),
        X_train, X_test,
    )
    return models


def train_and_evaluate(strategies: list[ResampleStrategy] | None = None) -> None:
    if strategies is None:
        strategies = ["none"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print("Loading data...")
    train_ev = load_events("train")
    test_ev = load_events("test")
    print(f"  Train: {train_ev.shape[0]:,} events")
    print(f"  Test:  {test_ev.shape[0]:,} events")

    X_train_raw, y_train_raw = prepare_features(train_ev)
    X_test, y_test = prepare_features(test_ev)

    # Collect all models across strategies
    all_models: dict[str, tuple] = {}

    for strat in strategies:
        print(f"\n--- Preparing resampling strategy: {strat.upper()} ---")
        X_train, y_train = resample(X_train_raw, y_train_raw, strat)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        models = _build_models(X_train, X_test, X_train_s, X_test_s, strat)
        # Attach scaler and y_train to each entry for later use
        for name, (model, Xtr, Xte) in models.items():
            all_models[name] = (model, Xtr, Xte, y_train, scaler)

    # --- Train & evaluate on real test set ---
    all_results = []
    trained_models = {}

    print("\n" + "=" * 70)
    print("TRAINING & TEST SET EVALUATION")
    print("=" * 70)

    for name, (model, Xtr, Xte, y_tr, sc) in all_models.items():
        print(f"\n--- {name} ---")

        t0 = time.perf_counter()
        model.fit(Xtr, y_tr)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = model.predict(Xte)
        infer_time = time.perf_counter() - t0

        f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_wt = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        report = classification_report(y_test, y_pred, target_names=TYPE_ORDER, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=list(range(len(TYPE_ORDER))))

        print(f"  Train time:    {train_time:.2f}s")
        print(f"  Inference:     {infer_time:.3f}s ({len(y_test)/infer_time:.0f} events/s)")
        print(f"  F1 (macro):    {f1_mac:.4f}")
        print(f"  F1 (weighted): {f1_wt:.4f}")
        print(report)
        print(f"  Confusion matrix:\n{cm}\n")

        needs_scaling = "Logistic Regression" in name or "KNN" in name
        trained_models[name] = (model, needs_scaling, sc)

        all_results.append({
            "model": name,
            "train_time_s": round(train_time, 3),
            "inference_time_s": round(infer_time, 4),
            "throughput_eps": round(len(y_test) / infer_time),
            "f1_macro": round(f1_mac, 4),
            "f1_weighted": round(f1_wt, 4),
            "dataset": "test",
        })

    # --- Simulation evaluation ---
    print("\n" + "=" * 70)
    print("SIMULATION EVALUATION (synthetic traffic with DDoS bursts)")
    print("=" * 70)

    gen = TrafficGenerator(seed=123)

    for name, (model, needs_scaling, sc) in trained_models.items():
        print(f"\n--- {name} ---")

        if needs_scaling:
            wrapped = _ScaledPredictor(model, sc)
        else:
            wrapped = model

        sim = Simulator(model=wrapped, feature_cols=FEATURE_COLS, alert_threshold=3)

        # Batch on synthetic data
        synth_events = gen.generate_events(n=5000, ddos_ratio=0.05, suspicious_ratio=0.10)
        batch = sim.evaluate_batch(synth_events)
        print(f"  [Batch] F1 macro: {batch.f1_macro:.4f}  Accuracy: {batch.accuracy:.4f}")

        # Streaming simulation
        sr = sim.run_stream(
            gen, duration_seconds=120, events_per_second=10.0,
            burst_attacks=True, burst_interval_s=30, burst_duration_s=8,
            burst_ddos_ratio=0.50,
        )
        print(f"  [Stream] Accuracy: {sr.overall_accuracy:.4f}")
        print(f"  [Stream] DDoS detected/actual: {sr.total_ddos_detected}/{sr.total_ddos_actual}")
        print(f"  [Stream] Alerts: {sr.alert_count}  False: {sr.false_alert_count}  Missed windows: {sr.missed_attack_windows}")
        print(f"  [Stream] Mean latency: {sr.mean_detection_latency_ms:.1f}ms")

        all_results.append({
            "model": name,
            "f1_macro_synth_batch": round(batch.f1_macro, 4),
            "accuracy_synth_batch": round(batch.accuracy, 4),
            "stream_accuracy": round(sr.overall_accuracy, 4),
            "stream_ddos_detected": sr.total_ddos_detected,
            "stream_ddos_actual": sr.total_ddos_actual,
            "stream_alerts": sr.alert_count,
            "stream_false_alerts": sr.false_alert_count,
            "stream_missed_windows": sr.missed_attack_windows,
            "stream_latency_ms": round(sr.mean_detection_latency_ms, 2),
            "dataset": "simulation",
        })

    # --- Save results ---
    results_df = pd.DataFrame(all_results)
    results_path = RESULTS_DIR / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    test_rows = results_df[results_df["dataset"] == "test"][["model", "f1_macro", "f1_weighted", "throughput_eps", "train_time_s"]]
    print("\nTest set performance:")
    print(test_rows.to_string(index=False))

    sim_rows = results_df[results_df["dataset"] == "simulation"][["model", "f1_macro_synth_batch", "stream_accuracy", "stream_false_alerts", "stream_missed_windows"]]
    print("\nSimulation performance:")
    print(sim_rows.to_string(index=False))


class _ScaledPredictor:
    """Wraps a model with a pre-fitted scaler for simulation compatibility."""

    def __init__(self, model, scaler: StandardScaler):
        self._model = model
        self._scaler = scaler

    def predict(self, X):
        return self._model.predict(self._scaler.transform(X))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DDoS classifiers")
    parser.add_argument(
        "--resample",
        choices=["none", "smote", "adasyn", "all"],
        default="none",
        help="Resampling strategy for class imbalance (default: none). "
             "'all' runs none + smote + adasyn for comparison.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.resample == "all":
        strategies: list[ResampleStrategy] = ["none", "smote", "adasyn"]
    else:
        strategies = [args.resample]
    train_and_evaluate(strategies)
