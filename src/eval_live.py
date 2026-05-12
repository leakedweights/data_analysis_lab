"""Evaluate trained models on real hping3 traffic and synthetic traffic.

Runs live hping3 scenarios, captures events, evaluates all model variants,
and saves results + comparison plots.

Usage (inside Docker with hping3 + NET_RAW):
    python -m src.eval_live
    python -m src.eval_live --scenarios live_syn_flood live_multi_vector
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.live_capture import (
    LIVE_SCENARIOS,
    LiveTrafficGenerator,
    hping3_available,
)
from src.simulator import TYPE_ORDER, TYPE_TO_INT
from src.synthetic import TrafficGenerator
from src.utils.data_pipeline import load_events

FEATURE_COLS = [
    "Packet speed", "Data speed", "Avg packet len",
    "Avg source IP count", "Detect count", "Port number",
]

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
PLOTS_DIR = Path(__file__).resolve().parents[1] / "plots" / "svg"


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df[FEATURE_COLS].values.astype(np.float64)
    y = df["Type"].map(TYPE_TO_INT).values
    return X, y


def _train_models(
    X_train: np.ndarray, y_train: np.ndarray
) -> dict[str, tuple]:
    """Train all baseline models, return {name: (model, needs_scaling, scaler)}."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            True,
        ),
        "Decision Tree": (
            DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42),
            False,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=100, max_depth=15, class_weight="balanced",
                random_state=42, n_jobs=-1,
            ),
            False,
        ),
        "KNN (k=5)": (
            KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            True,
        ),
        "Gradient Boosting": (
            HistGradientBoostingClassifier(
                max_iter=200, max_depth=5, learning_rate=0.1,
                class_weight="balanced", random_state=42,
            ),
            False,
        ),
    }

    trained = {}
    for name, (model, needs_scaling) in models.items():
        Xtr = X_train_s if needs_scaling else X_train
        print(f"  Training {name}...")
        model.fit(Xtr, y_train)
        trained[name] = (model, needs_scaling, scaler)
    return trained


def _evaluate(
    name: str,
    model,
    needs_scaling: bool,
    scaler: StandardScaler,
    events: pd.DataFrame,
    source_label: str,
) -> dict:
    """Evaluate a single model on an event DataFrame."""
    X, y_true = prepare_features(events)
    if needs_scaling:
        X = scaler.transform(X)

    t0 = time.perf_counter()
    y_pred = model.predict(X)
    infer_time = time.perf_counter() - t0

    n = len(events)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_wt = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    labels = list(range(len(TYPE_ORDER)))
    report = classification_report(y_true, y_pred, target_names=TYPE_ORDER,
                                   labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print(f"\n  --- {name} [{source_label}] ---")
    print(f"    Events:       {n}")
    print(f"    F1 (macro):   {f1_mac:.4f}")
    print(f"    F1 (weighted):{f1_wt:.4f}")
    print(f"    Inference:    {infer_time:.4f}s ({n / infer_time:.0f} evt/s)")
    print(report)
    print(f"  Confusion matrix:\n{cm}\n")

    return {
        "model": name,
        "source": source_label,
        "n_events": n,
        "f1_macro": round(f1_mac, 4),
        "f1_weighted": round(f1_wt, 4),
        "inference_time_s": round(infer_time, 4),
        "throughput_eps": round(n / infer_time) if infer_time > 0 else 0,
    }


def _generate_synthetic_for_scenario(scenario_name: str, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic events matching a live scenario's profile."""
    gen = TrafficGenerator(seed=seed)
    scenario = LIVE_SCENARIOS[scenario_name]

    # Count time spent in each traffic type
    type_seconds = {"Normal traffic": 0.0, "Suspicious traffic": 0.0, "DDoS attack": 0.0}
    from src.live_capture import HPING_ATTACKS
    for phase in scenario.phases:
        spec = HPING_ATTACKS[phase.attack_name]
        type_seconds[spec.traffic_type] += phase.duration_s

    total_s = sum(type_seconds.values())
    if total_s == 0:
        total_s = 1.0

    ddos_ratio = type_seconds["DDoS attack"] / total_s
    suspicious_ratio = type_seconds["Suspicious traffic"] / total_s

    # ~2 events/sec matches live sampling at 0.5s intervals
    n_events = int(scenario.duration_seconds * 2)

    events = gen.generate_events(
        n=n_events,
        ddos_ratio=ddos_ratio,
        suspicious_ratio=suspicious_ratio,
    )
    events["_source"] = "synthetic"
    return events


def _run_live_scenario(scenario_name: str) -> pd.DataFrame:
    """Run a live hping3 scenario and return captured events."""
    scenario = LIVE_SCENARIOS[scenario_name]
    gen = LiveTrafficGenerator(target="127.0.0.1", interface="lo")

    print(f"\n  Running live scenario '{scenario_name}' ({scenario.duration_seconds}s)...")
    gen.generate_stream_live(scenario)
    events = gen.pop_events()

    if not events:
        print(f"  WARNING: No events captured for {scenario_name}")
        return pd.DataFrame()

    df = pd.DataFrame(events)
    for col in ["Card", "Victim IP", "Attack code", "Type"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    df["_source"] = "hping3"
    print(f"  Captured {len(df)} events")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate models on live + synthetic traffic")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(LIVE_SCENARIOS.keys()),
        choices=list(LIVE_SCENARIOS.keys()),
        help="Which live scenarios to run",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load training data & train models ---
    print("Loading training data...")
    train_ev = load_events("train")
    X_train, y_train = prepare_features(train_ev)
    print(f"  Train: {train_ev.shape[0]:,} events")

    print("\nTraining models...")
    trained = _train_models(X_train, y_train)

    # --- Check hping3 ---
    has_hping3 = hping3_available()
    if has_hping3:
        print("\nhping3 found — will generate real traffic")
    else:
        print("\nhping3 NOT found — cannot run live scenarios")
        return

    # --- Run scenarios ---
    all_results = []

    for scenario_name in args.scenarios:
        print(f"\n{'=' * 70}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'=' * 70}")

        # Generate synthetic traffic matching the scenario profile
        print("\n  Generating synthetic baseline for this scenario...")
        synth_events = _generate_synthetic_for_scenario(scenario_name)
        print(f"  Generated {len(synth_events)} synthetic events")

        # Run live scenario
        live_events = _run_live_scenario(scenario_name)

        # Evaluate all models on both datasets
        for name, (model, needs_scaling, scaler) in trained.items():
            # Synthetic
            res = _evaluate(name, model, needs_scaling, scaler, synth_events,
                            f"synthetic_{scenario_name}")
            res["scenario"] = scenario_name
            all_results.append(res)

            # Live
            if len(live_events) > 0:
                res = _evaluate(name, model, needs_scaling, scaler, live_events,
                                f"hping3_{scenario_name}")
                res["scenario"] = scenario_name
                all_results.append(res)

    # --- Save results ---
    results_df = pd.DataFrame(all_results)
    results_path = RESULTS_DIR / "live_evaluation.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    for scenario_name in args.scenarios:
        print(f"\n--- {scenario_name} ---")
        sc_df = results_df[results_df["scenario"] == scenario_name]
        synth_df = sc_df[sc_df["source"].str.startswith("synthetic")]
        live_df = sc_df[sc_df["source"].str.startswith("hping3")]

        if len(synth_df) > 0 and len(live_df) > 0:
            merged = synth_df[["model", "f1_macro"]].rename(columns={"f1_macro": "synth_f1"})
            merged = merged.merge(
                live_df[["model", "f1_macro"]].rename(columns={"f1_macro": "live_f1"}),
                on="model",
            )
            merged["delta"] = merged["live_f1"] - merged["synth_f1"]
            print(merged.to_string(index=False))

    # --- Generate plots ---
    _generate_plots(results_df)


def _generate_plots(df: pd.DataFrame) -> None:
    """Generate comparison plots for live vs synthetic evaluation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = df["scenario"].unique()

    MODELS = [
        "Logistic Regression", "Decision Tree", "Random Forest",
        "KNN (k=5)", "Gradient Boosting",
    ]
    SHORT = {
        "Logistic Regression": "LogReg", "Decision Tree": "DTree",
        "Random Forest": "RF", "KNN (k=5)": "KNN", "Gradient Boosting": "GBT",
    }

    # --- Plot 1: Per-scenario comparison ---
    n_sc = len(scenarios)
    fig, axes = plt.subplots(1, n_sc, figsize=(6 * n_sc, 5.5), sharey=True, squeeze=False)
    axes = axes[0]

    for ax, scenario in zip(axes, scenarios):
        sc_df = df[df["scenario"] == scenario]
        synth = sc_df[sc_df["source"].str.startswith("synthetic")]
        live = sc_df[sc_df["source"].str.startswith("hping3")]

        models = [m for m in MODELS if m in synth["model"].values]
        x = np.arange(len(models))
        width = 0.35

        synth_f1 = [synth.loc[synth["model"] == m, "f1_macro"].values[0] for m in models]
        bars1 = ax.bar(x - width / 2, synth_f1, width, label="Synthetic",
                       color="#2563eb", edgecolor="white", linewidth=0.5)

        if len(live) > 0:
            live_f1 = [live.loc[live["model"] == m, "f1_macro"].values[0]
                       if m in live["model"].values else 0 for m in models]
            bars2 = ax.bar(x + width / 2, live_f1, width, label="hping3 (real)",
                           color="#ef4444", edgecolor="white", linewidth=0.5)

            for bars in [bars1, bars2]:
                for bar in bars:
                    h = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                            f"{h:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
        else:
            for bar in bars1:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

        sc_label = scenario.replace("live_", "").replace("_", " ").title()
        ax.set_title(sc_label, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in models], fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    axes[0].set_ylabel("F1 Macro", fontsize=11)
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Model Performance: Synthetic vs Real (hping3) Traffic",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "live_vs_synthetic.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved live_vs_synthetic.svg")

    # --- Plot 2: Aggregated across all scenarios ---
    fig, ax = plt.subplots(figsize=(10, 5.5))

    synth_all = df[df["source"].str.startswith("synthetic")]
    live_all = df[df["source"].str.startswith("hping3")]

    models = [m for m in MODELS if m in synth_all["model"].values]
    x = np.arange(len(models))
    width = 0.35

    synth_means = [synth_all.loc[synth_all["model"] == m, "f1_macro"].mean() for m in models]
    bars1 = ax.bar(x - width / 2, synth_means, width, label="Synthetic (avg)",
                   color="#2563eb", edgecolor="white", linewidth=0.5)

    if len(live_all) > 0:
        live_means = [live_all.loc[live_all["model"] == m, "f1_macro"].mean()
                      if m in live_all["model"].values else 0 for m in models]
        bars2 = ax.bar(x + width / 2, live_means, width, label="hping3 (avg)",
                       color="#ef4444", edgecolor="white", linewidth=0.5)

        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    else:
        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in models], fontsize=11)
    ax.set_ylabel("F1 Macro (avg across scenarios)", fontsize=11)
    ax.set_title("Average Model Performance: Synthetic vs Real Traffic",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "live_vs_synthetic_avg.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved live_vs_synthetic_avg.svg")


if __name__ == "__main__":
    main()
