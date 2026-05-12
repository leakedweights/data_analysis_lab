"""Generate SVG plots comparing v2 baseline, Fix A+B, and Fix A+B+C.

Reads ``results/abc_per_class.csv`` (produced by ``src.bench_abc``) and
writes:

* ``v2_abc_macro_f1.svg``    — headline macro F1 across the three configs
                                on test + genericity, all five models.
* ``v2_abc_per_class.svg``   — per-class F1 (Suspicious + DDoS) on test
                                + genericity for v2 vs A+B+C; this is
                                where the wins / regressions live.
* ``v2_abc_susp_collapse.svg`` — Suspicious-class recall on genericity
                                for all three configs across all five
                                models. Makes the RF/DT/GBT regression
                                under Fix C visually obvious.
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

# Re-using the existing v2 palette where possible.
COLOR_V2 = "#2563eb"   # blue — original v2
COLOR_AB = "#10b981"   # emerald — Fix A+B (features only)
COLOR_ABC = "#8b5cf6"  # violet — Fix A+B+C (features + gen-prior weights)

CONFIG_ORDER = ["v2", "ab", "abc"]
CONFIG_LABEL = {
    "v2":  "v2 baseline (balanced)",
    "ab":  "+ Fix A+B (features)",
    "abc": "+ Fix A+B+C (gen-prior wts)",
}
CONFIG_COLOR = {"v2": COLOR_V2, "ab": COLOR_AB, "abc": COLOR_ABC}


def _style_ax(ax, ymax: float = 1.0) -> None:
    ax.set_ylim(0, ymax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")


def _annotate(ax, bars, fs: int = 7, fmt: str = "{:.3f}") -> None:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                fmt.format(h), ha="center", va="bottom",
                fontsize=fs, fontweight="bold")


def _macro_value(df: pd.DataFrame, model: str, config: str, dataset: str) -> float:
    row = df[(df["model"] == model) & (df["config"] == config)
             & (df["dataset"] == dataset) & (df["class"] == "__macro__")]
    return float(row["f1"].iloc[0]) if not row.empty else 0.0


def _class_value(df: pd.DataFrame, model: str, config: str, dataset: str,
                 cls: str, metric: str = "f1") -> float:
    row = df[(df["model"] == model) & (df["config"] == config)
             & (df["dataset"] == dataset) & (df["class"] == cls)]
    return float(row[metric].iloc[0]) if not row.empty else 0.0


# ---------------------------------------------------------------------------
# Plot 1 — headline macro F1 across all three configs
# ---------------------------------------------------------------------------

def plot_macro_f1(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    x = np.arange(len(MODELS))
    width = 0.27

    for ax, dataset, title in [
        (axes[0], "test", "Test split (SetC, n=130k)"),
        (axes[1], "genericity", "Genericity split (SetD, n=438k)"),
    ]:
        for i, cfg in enumerate(CONFIG_ORDER):
            vals = [_macro_value(df, m, cfg, dataset) for m in MODELS]
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
    axes[0].legend(loc="upper left", fontsize=9, framealpha=0.95)
    fig.suptitle("Failure-mode fixes — macro F1 across configurations",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = PLOTS_DIR / "v2_abc_macro_f1.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ---------------------------------------------------------------------------
# Plot 2 — per-class F1 for the two minority classes
# ---------------------------------------------------------------------------

def plot_per_class(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    x = np.arange(len(MODELS))
    width = 0.27

    cells = [
        (axes[0, 0], "test", "Suspicious traffic", "Test — Suspicious class F1"),
        (axes[0, 1], "test", "DDoS attack",        "Test — DDoS class F1"),
        (axes[1, 0], "genericity", "Suspicious traffic",
            "Genericity — Suspicious class F1"),
        (axes[1, 1], "genericity", "DDoS attack",
            "Genericity — DDoS class F1"),
    ]

    for ax, ds, cls, title in cells:
        for i, cfg in enumerate(CONFIG_ORDER):
            vals = [_class_value(df, m, cfg, ds, cls) for m in MODELS]
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
    axes[0, 0].legend(loc="upper left", fontsize=9, framealpha=0.95)
    fig.suptitle(
        "Per-class F1 — Suspicious and DDoS minority classes\n"
        "(Normal omitted; always > 0.93)",
        fontsize=13, fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    path = PLOTS_DIR / "v2_abc_per_class.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ---------------------------------------------------------------------------
# Plot 3 — Suspicious-class recall on genericity (the regression story)
# ---------------------------------------------------------------------------

def plot_susp_collapse(df: pd.DataFrame) -> None:
    """The Suspicious-recall collapse on genericity under Fix C is the
    single most important regression to show — RF / DT / GBT lose 30+
    percentage points of recall on the Suspicious class because the
    gen-prior weights de-emphasize it. This is the trade-off behind the
    test-precision wins."""
    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(MODELS))
    width = 0.27

    for i, cfg in enumerate(CONFIG_ORDER):
        vals = [_class_value(df, m, cfg, "genericity",
                             "Suspicious traffic", "recall")
                for m in MODELS]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=CONFIG_LABEL[cfg],
                      color=CONFIG_COLOR[cfg], edgecolor="white",
                      linewidth=0.5)
        _annotate(ax, bars, fs=8, fmt="{:.2f}")

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in MODELS], fontsize=11)
    ax.set_ylabel("Suspicious-class recall (genericity split)", fontsize=11)
    ax.set_title(
        "The trade-off behind Fix C — Suspicious recall on genericity\n"
        "Tree-based models lose 30+ pp of recall when the class weight "
        "stops over-weighting Suspicious",
        fontsize=12, fontweight="bold",
    )
    _style_ax(ax)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    path = PLOTS_DIR / "v2_abc_susp_collapse.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS_DIR / "abc_per_class.csv")
    print("Generating ABC comparison plots...")
    plot_macro_f1(df)
    plot_per_class(df)
    plot_susp_collapse(df)
    print("Done.")


if __name__ == "__main__":
    main()
