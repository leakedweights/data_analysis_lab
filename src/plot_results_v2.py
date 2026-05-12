"""Generate SVG comparison plots + a Figma-friendly text summary from
``results/model_comparison_v2.csv``.

Plots
-----
1. ``v2_features_vs_v1.svg`` — v1 none vs v2 none on test + genericity
2. ``v2_resampling_comparison.svg`` — v2 none vs SMOTE vs ADASYN on
   test + genericity
3. ``v2_full_matrix.svg`` — the whole 4 × 2 comparison in one figure

Text
----
``results/v2_summary.txt`` — a plain-text block sized for a Figma
text frame (monospace, ≤ 80 cols).
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

COLOR_V1 = "#9ca3af"        # slate grey — the old way
COLOR_V2_NONE = "#2563eb"   # blue — the new baseline
COLOR_V2_SMOTE = "#f59e0b"  # amber — SMOTE
COLOR_V2_ADASYN = "#ef4444" # red   — ADASYN


def load_df() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "model_comparison_v2.csv")


def _lookup(df: pd.DataFrame, model: str, features: str, resample: str,
            dataset: str) -> float:
    row = df[
        (df["model"] == model)
        & (df["features"] == features)
        & (df["resample"] == resample)
        & (df["dataset"] == dataset)
    ]
    if row.empty:
        return 0.0
    return float(row["f1_macro"].iloc[0])


def _style_ax(ax) -> None:
    ax.set_ylim(0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")


def _annotate(ax, bars, fs: int = 7) -> None:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                f"{h:.3f}", ha="center", va="bottom",
                fontsize=fs, fontweight="bold")


# ---------------------------------------------------------------------------
# Plot 1 — v1 none vs v2 none
# ---------------------------------------------------------------------------

def plot_v2_vs_v1(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    x = np.arange(len(MODELS))
    width = 0.38

    for ax, dataset, title in [
        (axes[0], "test", "Test split (SetC)"),
        (axes[1], "genericity", "Genericity split (SetD)"),
    ]:
        v1 = [_lookup(df, m, "v1", "none", dataset) for m in MODELS]
        v2 = [_lookup(df, m, "v2", "none", dataset) for m in MODELS]

        b1 = ax.bar(x - width / 2, v1, width, label="v1 (event-only, 6 feat)",
                    color=COLOR_V1, edgecolor="white", linewidth=0.5)
        b2 = ax.bar(x + width / 2, v2, width, label="v2 (component agg, 15 feat)",
                    color=COLOR_V2_NONE, edgecolor="white", linewidth=0.5)
        _annotate(ax, b1)
        _annotate(ax, b2)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in MODELS], fontsize=10)
        _style_ax(ax)

    axes[0].set_ylabel("F1 macro", fontsize=11)
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("v1 vs v2 features — same models, no resampling",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = PLOTS_DIR / "v2_features_vs_v1.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ---------------------------------------------------------------------------
# Plot 2 — v2 resampling comparison
# ---------------------------------------------------------------------------

def plot_v2_resampling(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    x = np.arange(len(MODELS))
    width = 0.26

    for ax, dataset, title in [
        (axes[0], "test", "Test split (SetC)"),
        (axes[1], "genericity", "Genericity split (SetD)"),
    ]:
        none = [_lookup(df, m, "v2", "none", dataset) for m in MODELS]
        smote = [_lookup(df, m, "v2", "smote", dataset) for m in MODELS]
        adasyn = [_lookup(df, m, "v2", "adasyn", dataset) for m in MODELS]

        b1 = ax.bar(x - width, none, width, label="v2 none",
                    color=COLOR_V2_NONE, edgecolor="white", linewidth=0.5)
        b2 = ax.bar(x, smote, width, label="v2 SMOTE",
                    color=COLOR_V2_SMOTE, edgecolor="white", linewidth=0.5)
        b3 = ax.bar(x + width, adasyn, width, label="v2 ADASYN",
                    color=COLOR_V2_ADASYN, edgecolor="white", linewidth=0.5)
        _annotate(ax, b1, fs=6)
        _annotate(ax, b2, fs=6)
        _annotate(ax, b3, fs=6)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in MODELS], fontsize=10)
        _style_ax(ax)

    axes[0].set_ylabel("F1 macro", fontsize=11)
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("v2 feature set: data augmentation comparison",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = PLOTS_DIR / "v2_resampling_comparison.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ---------------------------------------------------------------------------
# Plot 3 — full matrix
# ---------------------------------------------------------------------------

def plot_full_matrix(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.0), sharey=True)
    x = np.arange(len(MODELS))
    width = 0.2

    variants = [
        ("v1", "none", "v1 none", COLOR_V1),
        ("v2", "none", "v2 none", COLOR_V2_NONE),
        ("v2", "smote", "v2 SMOTE", COLOR_V2_SMOTE),
        ("v2", "adasyn", "v2 ADASYN", COLOR_V2_ADASYN),
    ]

    for ax, dataset, title in [
        (axes[0], "test", "Test split (SetC)"),
        (axes[1], "genericity", "Genericity split (SetD)"),
    ]:
        for i, (feat, strat, label, color) in enumerate(variants):
            vals = [_lookup(df, m, feat, strat, dataset) for m in MODELS]
            offset = (i - 1.5) * width
            bars = ax.bar(x + offset, vals, width, label=label,
                          color=color, edgecolor="white", linewidth=0.5)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                        f"{h:.2f}", ha="center", va="bottom",
                        fontsize=5.5, fontweight="bold")

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in MODELS], fontsize=10)
        _style_ax(ax)

    axes[0].set_ylabel("F1 macro", fontsize=11)
    axes[0].legend(loc="upper left", fontsize=8, ncol=2)
    fig.suptitle("Full comparison — v1 vs v2 × (none, SMOTE, ADASYN)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = PLOTS_DIR / "v2_full_matrix.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ---------------------------------------------------------------------------
# Plain text summary — ready to paste into Figma
# ---------------------------------------------------------------------------

def _fmt_row(label: str, vals: list[float], widths: list[int]) -> str:
    cells = [f"{label:<22}"]
    for v, w in zip(vals, widths):
        cells.append(f"{v:>{w}.4f}")
    return " ".join(cells)


def write_text_summary(df: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("DDoS DETECTION — MODEL v2 COMPARISON")
    lines.append("=" * 68)
    lines.append("")
    lines.append("v1  : 6 raw event columns, no components, no genericity eval")
    lines.append("v2  : 15 features from components aggregated per event,")
    lines.append("      log1p source-IP compression, held-out genericity eval")
    lines.append("")
    lines.append("F1 macro — higher is better, 3 classes")
    lines.append("-" * 68)
    lines.append("")

    configs = [
        ("v1", "none", "v1 none"),
        ("v2", "none", "v2 none"),
        ("v2", "smote", "v2 SMOTE"),
        ("v2", "adasyn", "v2 ADASYN"),
    ]

    # Test split
    lines.append("TEST SPLIT (SetC, 130,000 events)")
    lines.append("                       LogReg   DTree      RF     KNN     GBT")
    for feat, strat, label in configs:
        vals = [_lookup(df, m, feat, strat, "test") for m in MODELS]
        lines.append(_fmt_row(label, vals, [7, 7, 7, 7, 7]))
    lines.append("")

    # Genericity split
    lines.append("GENERICITY SPLIT (SetD, 437,657 events — never seen by v1)")
    lines.append("                       LogReg   DTree      RF     KNN     GBT")
    for feat, strat, label in configs:
        vals = [_lookup(df, m, feat, strat, "genericity") for m in MODELS]
        lines.append(_fmt_row(label, vals, [7, 7, 7, 7, 7]))
    lines.append("")
    lines.append("-" * 68)
    lines.append("")
    lines.append("KEY FINDINGS")
    lines.append("")
    lines.append("1. v2 matches v1 on the test split (within +-0.07 for all")
    lines.append("   5 models), without relying on features that cannot be")
    lines.append("   measured at inference time.")
    lines.append("")
    lines.append("2. v2 wins on the genericity split for 4 of 5 models, biggest")
    lines.append("   gains on Logistic Regression (+0.20) and Gradient Boosting")
    lines.append("   (+0.02).  v1 never reported these numbers.")
    lines.append("")
    lines.append("3. Data augmentation on v2 features is model-specific:")
    lines.append("   * SMOTE boosts Gradient Boosting on test (+0.12)")
    lines.append("   * Resampling hurts Decision Tree / RF / KNN on both splits")
    lines.append("   * Logistic Regression is largely unaffected")
    lines.append("")
    lines.append("4. Best single configuration per objective:")
    lines.append("   * Best test F1:         v1 KNN         (0.7841)")
    lines.append("   * Best genericity F1:   v2 none GBT    (0.6864)")
    lines.append("   * Best balanced:        v2 SMOTE GBT   (0.66 / 0.66)")
    lines.append("")
    lines.append("5. Soundness win: v2 uses one featurizer across training,")
    lines.append("   synthetic eval, and live capture -- no train/serve skew.")
    lines.append("   v1 had different feature semantics in the live path")
    lines.append("   (invented Source IPs, broken Detect count).")

    txt = "\n".join(lines) + "\n"
    out = RESULTS_DIR / "v2_summary.txt"
    out.write_text(txt)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------

def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_df()
    print("Generating v2 comparison plots...")
    plot_v2_vs_v1(df)
    plot_v2_resampling(df)
    plot_full_matrix(df)
    write_text_summary(df)
    print("Done.")


if __name__ == "__main__":
    main()
