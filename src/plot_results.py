"""Generate comparison plots from model_comparison.csv."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
PLOTS_DIR = Path(__file__).resolve().parents[1] / "plots" / "svg"

MODELS = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "KNN (k=5)",
    "Gradient Boosting",
]

SHORT_NAMES = {
    "Logistic Regression": "LogReg",
    "Decision Tree": "DTree",
    "Random Forest": "RF",
    "KNN (k=5)": "KNN",
    "Gradient Boosting": "GBT",
}

PALETTE = {
    "test": "#2563eb",
    "sim_batch": "#f59e0b",
    "sim_stream": "#10b981",
}

STRATEGY_PALETTE = {
    "none": "#2563eb",
    "SMOTE": "#f59e0b",
    "ADASYN": "#ef4444",
}


def load_data() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "model_comparison.csv")


def _extract_base_and_strategy(model_name: str) -> tuple[str, str]:
    for tag in ["[SMOTE]", "[ADASYN]"]:
        if tag in model_name:
            return model_name.replace(f" {tag}", ""), tag.strip("[]")
    return model_name, "none"


def plot_baseline_performance(df: pd.DataFrame) -> None:
    """Plot 1: baseline (no resampling) F1 macro across test / sim batch / sim stream."""
    test_rows = df[(df["dataset"] == "test")]
    sim_rows = df[(df["dataset"] == "simulation")]

    # Filter to baseline only (no SMOTE/ADASYN tag)
    test_base = test_rows[~test_rows["model"].str.contains(r"\[")].copy()
    sim_base = sim_rows[~sim_rows["model"].str.contains(r"\[")].copy()

    # Exclude dummy baseline
    test_base = test_base[test_base["model"] != "Baseline (majority)"]
    sim_base = sim_base[sim_base["model"] != "Baseline (majority)"]

    models = [m for m in MODELS if m in test_base["model"].values]
    x = np.arange(len(models))
    width = 0.25

    test_f1 = [test_base.loc[test_base["model"] == m, "f1_macro"].values[0] for m in models]
    sim_batch_f1 = [sim_base.loc[sim_base["model"] == m, "f1_macro_synth_batch"].values[0] for m in models]
    sim_stream_acc = [sim_base.loc[sim_base["model"] == m, "stream_accuracy"].values[0] for m in models]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    bars1 = ax.bar(x - width, test_f1, width, label="Test F1 (macro)", color=PALETTE["test"], edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x, sim_batch_f1, width, label="Sim Batch F1 (macro)", color=PALETTE["sim_batch"], edgecolor="white", linewidth=0.5)
    bars3 = ax.bar(x + width, sim_stream_acc, width, label="Sim Stream Accuracy", color=PALETTE["sim_stream"], edgecolor="white", linewidth=0.5)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_NAMES[m] for m in models], fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Baseline Model Performance: Test vs Simulation", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "baseline_performance.svg", format="svg")
    plt.close(fig)
    print("  Saved baseline_performance.svg")


def plot_resampling_comparison(df: pd.DataFrame) -> None:
    """Plot 2: For each evaluation context (test / sim batch / sim stream),
    compare none vs SMOTE vs ADASYN per model — 3 subplots."""

    test_rows = df[df["dataset"] == "test"].copy()
    sim_rows = df[df["dataset"] == "simulation"].copy()

    # Parse strategy
    test_rows[["base_model", "strategy"]] = test_rows["model"].apply(
        lambda m: pd.Series(_extract_base_and_strategy(m))
    )
    sim_rows[["base_model", "strategy"]] = sim_rows["model"].apply(
        lambda m: pd.Series(_extract_base_and_strategy(m))
    )

    # Drop baseline dummy
    test_rows = test_rows[test_rows["base_model"] != "Baseline (majority)"]
    sim_rows = sim_rows[sim_rows["base_model"] != "Baseline (majority)"]

    models = [m for m in MODELS if m in test_rows["base_model"].values]
    strategies = ["none", "SMOTE", "ADASYN"]

    # Collect scores
    def get_test_f1(model, strat):
        row = test_rows[(test_rows["base_model"] == model) & (test_rows["strategy"] == strat)]
        return row["f1_macro"].values[0] if len(row) > 0 else 0

    def get_sim_batch_f1(model, strat):
        row = sim_rows[(sim_rows["base_model"] == model) & (sim_rows["strategy"] == strat)]
        return row["f1_macro_synth_batch"].values[0] if len(row) > 0 else 0

    def get_sim_stream_acc(model, strat):
        row = sim_rows[(sim_rows["base_model"] == model) & (sim_rows["strategy"] == strat)]
        return row["stream_accuracy"].values[0] if len(row) > 0 else 0

    metrics = [
        ("Test Set — F1 Macro", get_test_f1),
        ("Simulation Batch — F1 Macro", get_sim_batch_f1),
        ("Simulation Stream — Accuracy", get_sim_stream_acc),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)

    x = np.arange(len(models))
    width = 0.25

    for ax, (title, getter) in zip(axes, metrics):
        for i, strat in enumerate(strategies):
            vals = [getter(m, strat) for m in models]
            offset = (i - 1) * width
            bars = ax.bar(x + offset, vals, width,
                          label=strat if strat != "none" else "Baseline",
                          color=STRATEGY_PALETTE[strat],
                          edgecolor="white", linewidth=0.5)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_NAMES[m] for m in models], fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    axes[0].set_ylabel("Score", fontsize=11)
    axes[0].legend(loc="upper left", fontsize=9)

    fig.suptitle("Resampling Strategy Comparison: Baseline vs SMOTE vs ADASYN",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "resampling_comparison.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    print("  Saved resampling_comparison.svg")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    print("Generating plots...")
    plot_baseline_performance(df)
    plot_resampling_comparison(df)
    print("Done.")


if __name__ == "__main__":
    main()
