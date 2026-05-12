"""DDoS detection simulation harness.

Streams synthetic traffic through trained models and evaluates
detection performance with latency, accuracy, and throughput metrics.

Usage:
    from src.simulator import Simulator
    from src.synthetic import TrafficGenerator

    gen = TrafficGenerator(seed=42)
    sim = Simulator(model=my_sklearn_model, feature_cols=[...])

    # Batch evaluation
    results = sim.evaluate_batch(gen.generate_events(n=5000))

    # Streaming simulation with window-based detection
    stream_results = sim.run_stream(gen, duration_seconds=300)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.synthetic import TrafficGenerator

TYPE_ORDER = ["Normal traffic", "Suspicious traffic", "DDoS attack"]
TYPE_TO_INT = {t: i for i, t in enumerate(TYPE_ORDER)}
INT_TO_TYPE = {i: t for t, i in TYPE_TO_INT.items()}


class Predictor(Protocol):
    """Any object with a sklearn-compatible predict interface."""
    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray: ...


DEFAULT_FEATURE_COLS = [
    "Packet speed", "Data speed", "Avg packet len",
    "Avg source IP count", "Detect count", "Port number",
]


@dataclass
class BatchResult:
    """Results from batch model evaluation."""
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    confusion: np.ndarray
    report: str
    predictions: np.ndarray
    ground_truth: np.ndarray
    inference_time_s: float
    events_per_second: float

    def summary(self) -> str:
        return (
            f"Batch Evaluation\n"
            f"  Accuracy:        {self.accuracy:.4f}\n"
            f"  Precision (macro): {self.precision_macro:.4f}\n"
            f"  Recall (macro):    {self.recall_macro:.4f}\n"
            f"  F1 (macro):        {self.f1_macro:.4f}\n"
            f"  Inference time:    {self.inference_time_s:.3f}s\n"
            f"  Throughput:        {self.events_per_second:.0f} events/s\n"
            f"\n{self.report}\n"
            f"Confusion matrix (rows=true, cols=pred):\n"
            f"  Labels: {TYPE_ORDER}\n"
            f"{self.confusion}"
        )


@dataclass
class WindowResult:
    """Metrics for a single detection window."""
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    n_events: int
    n_predicted_ddos: int
    n_actual_ddos: int
    accuracy: float
    detection_latency_ms: float
    alert_raised: bool


@dataclass
class StreamResult:
    """Results from a streaming simulation run."""
    windows: list[WindowResult]
    total_events: int
    total_ddos_detected: int
    total_ddos_actual: int
    overall_accuracy: float
    mean_detection_latency_ms: float
    alert_count: int
    false_alert_count: int
    missed_attack_windows: int

    def summary(self) -> str:
        lines = [
            "Stream Simulation Results",
            f"  Total events:         {self.total_events}",
            f"  DDoS detected/actual: {self.total_ddos_detected}/{self.total_ddos_actual}",
            f"  Overall accuracy:     {self.overall_accuracy:.4f}",
            f"  Mean detect latency:  {self.mean_detection_latency_ms:.1f}ms",
            f"  Alerts raised:        {self.alert_count}",
            f"  False alerts:         {self.false_alert_count}",
            f"  Missed attack windows:{self.missed_attack_windows}",
            f"  Windows processed:    {len(self.windows)}",
        ]
        return "\n".join(lines)


class Simulator:
    """Run inference simulations against synthetic or real traffic."""

    def __init__(
        self,
        model: Predictor,
        feature_cols: list[str] | None = None,
        alert_threshold: int = 3,
        label_col: str = "Type",
    ):
        """
        Args:
            model: Trained sklearn-compatible classifier.
            feature_cols: Columns to use as model input.
            alert_threshold: Min predicted DDoS events in a window to raise alert.
            label_col: Column name for ground truth labels.
        """
        self.model = model
        self.feature_cols = feature_cols or DEFAULT_FEATURE_COLS
        self.alert_threshold = alert_threshold
        self.label_col = label_col

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].copy()
        # Ensure numeric
        for col in X.columns:
            if X[col].dtype.name == "category":
                X[col] = X[col].cat.codes
        return X.values

    def _encode_labels(self, labels: pd.Series) -> np.ndarray:
        return labels.map(TYPE_TO_INT).values

    def evaluate_batch(self, events: pd.DataFrame) -> BatchResult:
        """Run model on all events at once and compute metrics."""
        X = self._prepare_features(events)
        y_true = self._encode_labels(events[self.label_col])

        t0 = time.perf_counter()
        y_pred = self.model.predict(X)
        t1 = time.perf_counter()

        inference_time = t1 - t0
        n = len(events)

        return BatchResult(
            accuracy=accuracy_score(y_true, y_pred),
            precision_macro=precision_score(y_true, y_pred, average="macro", zero_division=0),
            recall_macro=recall_score(y_true, y_pred, average="macro", zero_division=0),
            f1_macro=f1_score(y_true, y_pred, average="macro", zero_division=0),
            confusion=confusion_matrix(y_true, y_pred, labels=list(range(len(TYPE_ORDER)))),
            report=classification_report(
                y_true, y_pred,
                target_names=TYPE_ORDER,
                zero_division=0,
            ),
            predictions=y_pred,
            ground_truth=y_true,
            inference_time_s=inference_time,
            events_per_second=n / inference_time if inference_time > 0 else float("inf"),
        )

    def run_stream(
        self,
        generator: TrafficGenerator,
        duration_seconds: int = 300,
        events_per_second: float = 10.0,
        window_size_s: float = 5.0,
        ddos_ratio: float = 0.05,
        suspicious_ratio: float = 0.10,
        burst_attacks: bool = True,
        burst_interval_s: int = 60,
        burst_duration_s: int = 10,
        burst_ddos_ratio: float = 0.50,
    ) -> StreamResult:
        """Simulate streaming detection with sliding time windows.

        Events arrive in time order. The model processes each window
        and raises alerts when DDoS count exceeds the threshold.
        """
        stream = generator.generate_stream(
            duration_seconds=duration_seconds,
            events_per_second=events_per_second,
            ddos_ratio=ddos_ratio,
            suspicious_ratio=suspicious_ratio,
            burst_attacks=burst_attacks,
            burst_interval_s=burst_interval_s,
            burst_duration_s=burst_duration_s,
            burst_ddos_ratio=burst_ddos_ratio,
        )

        t_start = stream["Start time"].min()
        t_end = stream["Start time"].max()
        window_delta = pd.Timedelta(seconds=window_size_s)

        windows: list[WindowResult] = []
        all_preds = []
        all_true = []

        current = t_start
        while current < t_end:
            w_end = current + window_delta
            mask = (stream["Start time"] >= current) & (stream["Start time"] < w_end)
            window_events = stream[mask]

            if len(window_events) == 0:
                current = w_end
                continue

            X = self._prepare_features(window_events)
            y_true = self._encode_labels(window_events[self.label_col])

            t0 = time.perf_counter()
            y_pred = self.model.predict(X)
            latency_ms = (time.perf_counter() - t0) * 1000

            ddos_label = TYPE_TO_INT["DDoS attack"]
            n_pred_ddos = int((y_pred == ddos_label).sum())
            n_actual_ddos = int((y_true == ddos_label).sum())
            alert = n_pred_ddos >= self.alert_threshold

            all_preds.extend(y_pred.tolist())
            all_true.extend(y_true.tolist())

            windows.append(WindowResult(
                window_start=current,
                window_end=w_end,
                n_events=len(window_events),
                n_predicted_ddos=n_pred_ddos,
                n_actual_ddos=n_actual_ddos,
                accuracy=accuracy_score(y_true, y_pred),
                detection_latency_ms=latency_ms,
                alert_raised=alert,
            ))

            current = w_end

        all_preds_arr = np.array(all_preds)
        all_true_arr = np.array(all_true)
        ddos_label = TYPE_TO_INT["DDoS attack"]

        alert_windows = [w for w in windows if w.alert_raised]
        false_alerts = [w for w in alert_windows if w.n_actual_ddos == 0]
        missed = [w for w in windows if w.n_actual_ddos > 0 and not w.alert_raised]

        latencies = [w.detection_latency_ms for w in windows]

        return StreamResult(
            windows=windows,
            total_events=len(all_preds_arr),
            total_ddos_detected=int((all_preds_arr == ddos_label).sum()),
            total_ddos_actual=int((all_true_arr == ddos_label).sum()),
            overall_accuracy=accuracy_score(all_true_arr, all_preds_arr) if len(all_preds_arr) > 0 else 0.0,
            mean_detection_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
            alert_count=len(alert_windows),
            false_alert_count=len(false_alerts),
            missed_attack_windows=len(missed),
        )

    def run_stress_test(
        self,
        generator: TrafficGenerator,
        batch_sizes: list[int] | None = None,
        ddos_ratio: float = 0.10,
    ) -> pd.DataFrame:
        """Measure inference throughput at various batch sizes."""
        if batch_sizes is None:
            batch_sizes = [100, 500, 1000, 5000, 10000, 50000]

        results = []
        for bs in batch_sizes:
            events = generator.generate_events(n=bs, ddos_ratio=ddos_ratio)
            X = self._prepare_features(events)

            # Warmup
            self.model.predict(X[:min(10, len(X))])

            t0 = time.perf_counter()
            self.model.predict(X)
            elapsed = time.perf_counter() - t0

            results.append({
                "batch_size": bs,
                "inference_time_s": elapsed,
                "events_per_second": bs / elapsed if elapsed > 0 else float("inf"),
                "ms_per_event": elapsed / bs * 1000 if bs > 0 else 0,
            })

        return pd.DataFrame(results)
