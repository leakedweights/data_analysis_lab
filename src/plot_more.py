"""SVG comparison plots for the additional-features experiment.

Reads ``results/more_per_class.csv`` and writes:

* ``v2_more_macro_f1.svg``   — macro F1 across A+B / A+B+extras /
                                A+B+extras+OOD on test + genericity.
* ``v2_more_per_class.svg``  — per-class F1 (Suspicious + DDoS) for
                                test + genericity, all three configs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
PLOTS_DIR = Path(__file__).resolve().parents[1] / "plots" / "svg"

MODELS = [
    "Logistic Regression", "Decision Tree", "Random Forest",
    "KNN (k=5)", "Gradient Boosting",
]
SHORT = {
    "Logistic Regression": "LogReg", "Decision Tree": "DTree",
    "Random Forest": "RF", "KNN (k=5)": "KNN", "Gradient Boosting": "GBT",
}

COLOR_AB = "#10b981"        # emerald — A+B baseline
COLOR_EXTRAS = "#f59e0b"    # amber — A+B + extras (count + robust stats)
COLOR_OOD = "#dc2626"       # red — A+B + extras + GMM OOD signals

CONFIG_ORDER = ["ab", "ab_extras", "ab_extras_ood"]
CONFIG_LABEL = {
    "ab":             "A+B (21 feat)",
    "ab_extras":      "+ component-count & robust stats (28 feat)",
    "ab_extras_ood":  "+ GMM OOD signals (30 feat)",
}
CONFIG_COLOR = {
    "ab": COLOR_AB, "ab_extras": COLOR_EXTRAS, "ab_extras_ood": COLOR_OOD,
}


def _style_ax(ax) -> None:
    ax.set_ylim(0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")


def _annotate(ax, bars, fs: int = 7, fmt: str = "{:.3f}") -> None:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                fmt.format(h), ha="center", va="bottom",
                fontsize=fs, fontweight="bold")


def _value(df: pd.DataFrame, model: str, config: str, dataset: str,
           cls: str = "__macro__", metric: str = "f1") -> float:
    row = df[(df["model"] == model) & (df["config"] == config)
             & (df["dataset"] == dataset) & (df["class"] == cls)]
    return float(row[metric].iloc[0]) if not row.empty else 0.0


def plot_macro(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    x = np.arange(len(MODELS))
    width = 0.27

    for ax, dataset, title in [
        (axes[0], "test", "Test split (SetC)"),
        (axes[1], "genericity", "Genericity split (SetD)"),
    ]:
        for i, cfg in enumerate(CONFIG_ORDER):
            vals = [_value(df, m, cfg, dataset) for m in MODELS]
            offset = (i - 1) * width
            bars = ax.bar(x + offset, vals, width, label=CONFIG_LABEL[cfg],
                          color=CONFIG_COLOR[cfg], edgecolor="white",
                          linewidth=0.5)
            _annotate(ax, bars, fs=6.5)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in MODELS], fontsize=10)
        _style_ax(ax)

    axes[0].set_ylabel("F1 macro", fontsize=11)
    axes[0].legend(loc="upper left", fontsize=8.5, framealpha=0.95)
    fig.suptitle(
        "Additional derived features — macro F1 effect",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    path = PLOTS_DIR / "v2_more_macro_f1.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


def plot_per_class(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    x = np.arange(len(MODELS))
    width = 0.27

    cells = [
        (axes[0, 0], "test", "Suspicious traffic", "Test — Suspicious F1"),
        (axes[0, 1], "test", "DDoS attack",        "Test — DDoS F1"),
        (axes[1, 0], "genericity", "Suspicious traffic",
            "Genericity — Suspicious F1"),
        (axes[1, 1], "genericity", "DDoS attack",
            "Genericity — DDoS F1"),
    ]
    for ax, ds, cls, title in cells:
        for i, cfg in enumerate(CONFIG_ORDER):
            vals = [_value(df, m, cfg, ds, cls) for m in MODELS]
            offset = (i - 1) * width
            bars = ax.bar(x + offset, vals, width, label=CONFIG_LABEL[cfg],
                          color=CONFIG_COLOR[cfg], edgecolor="white",
                          linewidth=0.5)
            _annotate(ax, bars, fs=6, fmt="{:.2f}")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in MODELS], fontsize=10)
        _style_ax(ax)

    axes[0, 0].set_ylabel("F1 (per class)", fontsize=11)
    axes[1, 0].set_ylabel("F1 (per class)", fontsize=11)
    axes[0, 0].legend(loc="upper left", fontsize=8.5, framealpha=0.95)
    fig.suptitle(
        "Per-class F1 — Suspicious and DDoS minority classes\n"
        "A+B vs +extras vs +extras+OOD",
        fontsize=13, fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    path = PLOTS_DIR / "v2_more_per_class.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS_DIR / "more_per_class.csv")
    print("Generating MORE-feature comparison plots...")
    plot_macro(df)
    plot_per_class(df)
    print("Done.")


if __name__ == "__main__":
    main()
