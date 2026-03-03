"""Exploratory data analysis on the SCLDDoS2024 dataset.

Structured around key questions for the DDoS detection/classification task.
Run with: uv run python -m src.eda
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.data_pipeline import load_components, load_events

PLOTS_DIR = Path(__file__).resolve().parents[1] / "plots"
PNG_DIR = PLOTS_DIR / "png"
SVG_DIR = PLOTS_DIR / "svg"
TYPE_COLORS = {"Normal traffic": "#2ecc71", "Suspicious traffic": "#f39c12", "DDoS attack": "#e74c3c"}
TYPE_ORDER = ["Normal traffic", "Suspicious traffic", "DDoS attack"]


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(PNG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(SVG_DIR / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


# --- Q1: How severe is the class imbalance? ---

def plot_class_imbalance(events: pd.DataFrame) -> None:
    counts = events["Type"].value_counts().reindex(TYPE_ORDER)
    pcts = counts / counts.sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    bars = axes[0].bar(counts.index, counts.values,
                       color=[TYPE_COLORS[t] for t in counts.index])
    for bar, pct in zip(bars, pcts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{pct:.1f}%", ha="center", va="bottom", fontweight="bold")
    axes[0].set_title("Event Type Distribution")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=15)

    bars = axes[1].bar(counts.index, counts.values,
                       color=[TYPE_COLORS[t] for t in counts.index], log=True)
    for bar, n in zip(bars, counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"n={n:,}", ha="center", va="bottom", fontsize=9)
    axes[1].set_title("Event Type Distribution (log scale)")
    axes[1].set_ylabel("Count (log)")
    axes[1].tick_params(axis="x", rotation=15)

    fig.suptitle("Q1: Class Imbalance", fontsize=14, fontweight="bold")
    save(fig, "01_class_imbalance")


# --- Q2: Which features distinguish DDoS from normal traffic? ---

def plot_feature_comparison(events: pd.DataFrame) -> None:
    cols = ["Avg packet len", "Avg source IP count", "Detect count", "Packet speed"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, col in zip(axes.flat, cols):
        medians = events.groupby("Type")[col].median().reindex(TYPE_ORDER)
        bars = ax.bar(medians.index, medians.values,
                      color=[TYPE_COLORS[t] for t in medians.index])
        for bar, val in zip(bars, medians):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:,.0f}", ha="center", va="bottom", fontweight="bold")
        ax.set_title(col)
        ax.set_ylabel("Median value")
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Q2: Feature Medians by Event Type", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "02_feature_medians")


def plot_feature_distributions(events: pd.DataFrame) -> None:
    cols = ["Avg packet len", "Avg source IP count", "Detect count", "Packet speed"]
    # Use log-spaced bins for heavily skewed features
    log_cols = {"Avg source IP count", "Detect count"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, col in zip(axes.flat, cols):
        if col in log_cols:
            # Filter to > 0 for log scale, use log-spaced bins
            for t in TYPE_ORDER:
                subset = events.loc[events["Type"] == t, col]
                subset = subset[subset > 0]
                clip = subset.clip(upper=subset.quantile(0.99))
                bins = np.logspace(np.log10(clip.min()), np.log10(clip.max()), 40)
                ax.hist(clip, bins=bins, alpha=0.5, label=t, color=TYPE_COLORS[t], density=True)
            ax.set_xscale("log")
        else:
            for t in TYPE_ORDER:
                subset = events.loc[events["Type"] == t, col]
                clip = subset.clip(upper=subset.quantile(0.99))
                ax.hist(clip, bins=50, alpha=0.5, label=t, color=TYPE_COLORS[t], density=True)
        ax.set_title(col)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    fig.suptitle("Q2: Feature Distributions by Type (clipped at 99th pct)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "03_feature_distributions")


# --- Q3: What are the attack types and how do they differ? ---

def plot_attack_types(events: pd.DataFrame, components: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Events: attack code by Type (top 12)
    ddos = events[events["Type"] == "DDoS attack"]
    top_codes = ddos["Attack code"].value_counts().head(12)
    bars = axes[0].barh(range(len(top_codes)), top_codes.values, color="#e74c3c")
    axes[0].set_yticks(range(len(top_codes)))
    axes[0].set_yticklabels(top_codes.index)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Count")
    axes[0].set_title("Top Attack Codes in DDoS Events")

    # Components: top individual attack tokens
    comp_counts = components["Attack code"].value_counts()
    top = comp_counts.head(10)
    other = comp_counts.iloc[10:].sum()
    top = pd.concat([top, pd.Series({"Other": other})])
    bars = axes[1].barh(range(len(top)), top.values)
    axes[1].set_yticks(range(len(top)))
    axes[1].set_yticklabels(top.index)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Count")
    axes[1].set_xscale("log")
    axes[1].set_title("Component Attack Codes (top 10 + other)")

    fig.suptitle("Q3: Attack Type Breakdown", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "04_attack_types")


# --- Q4: What temporal patterns exist? ---

def plot_temporal_patterns(events: pd.DataFrame) -> None:
    df = events.dropna(subset=["Start time"]).copy()
    df["date"] = df["Start time"].dt.date

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    for ax, typ in zip(axes, TYPE_ORDER):
        daily = df[df["Type"] == typ].groupby("date").size()
        ax.fill_between(daily.index, daily.values, alpha=0.4, color=TYPE_COLORS[typ])
        ax.plot(daily.index, daily.values, color=TYPE_COLORS[typ], linewidth=0.8)
        ax.set_ylabel("Events / day")
        ax.set_title(typ, fontweight="bold", color=TYPE_COLORS[typ])
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Q4: Daily Event Volume by Type", fontsize=14, fontweight="bold")
    fig.autofmt_xdate()
    fig.tight_layout()
    save(fig, "05_temporal_patterns")


def plot_event_duration(events: pd.DataFrame) -> None:
    df = events.dropna(subset=["Start time", "End time"]).copy()
    df["duration_s"] = (df["End time"] - df["Start time"]).dt.total_seconds()
    df = df[df["duration_s"] >= 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for t in TYPE_ORDER:
        subset = df.loc[df["Type"] == t, "duration_s"]
        subset = subset[subset > 0].clip(upper=subset.quantile(0.99))
        bins = np.logspace(np.log10(max(subset.min(), 0.1)), np.log10(subset.max()), 40)
        axes[0].hist(subset, bins=bins, alpha=0.5, label=t, color=TYPE_COLORS[t], density=True)
    axes[0].set_xscale("log")
    axes[0].set_title("Duration Distribution (> 0s, clipped at 99th pct)")
    axes[0].set_xlabel("Duration (seconds, log scale)")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    medians = df.groupby("Type")["duration_s"].median().reindex(TYPE_ORDER)
    bars = axes[1].bar(medians.index, medians.values,
                       color=[TYPE_COLORS[t] for t in medians.index])
    for bar, val in zip(bars, medians):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.0f}s", ha="center", va="bottom", fontweight="bold")
    axes[1].set_title("Median Event Duration by Type")
    axes[1].set_ylabel("Seconds")
    axes[1].tick_params(axis="x", rotation=15)

    fig.suptitle("Q4: Event Duration Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "06_event_duration")


# --- Q5: How are components related to events? ---

def plot_components_per_event(events: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for t in TYPE_ORDER:
        subset = events.loc[events["Type"] == t, "Detect count"]
        subset = subset[subset > 0].clip(upper=subset.quantile(0.99))
        bins = np.logspace(np.log10(subset.min()), np.log10(max(subset.max(), 2)), 40)
        axes[0].hist(subset, bins=bins, alpha=0.5, label=t, color=TYPE_COLORS[t], density=True)
    axes[0].set_xscale("log")
    axes[0].set_title("Components per Event (clipped at 99th pct)")
    axes[0].set_xlabel("Detect count (log scale)")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    means = events.groupby("Type")["Detect count"].mean().reindex(TYPE_ORDER)
    bars = axes[1].bar(means.index, means.values,
                       color=[TYPE_COLORS[t] for t in means.index])
    for bar, val in zip(bars, means):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.1f}", ha="center", va="bottom", fontweight="bold")
    axes[1].set_title("Mean Components per Event by Type")
    axes[1].set_ylabel("Detect count (mean)")
    axes[1].tick_params(axis="x", rotation=15)

    fig.suptitle("Q5: Event Complexity (Components per Event)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "07_components_per_event")


# --- Q6: Correlation structure ---

def plot_correlation(events: pd.DataFrame) -> None:
    numeric_cols = ["Packet speed", "Data speed", "Avg packet len",
                    "Avg source IP count", "Detect count"]
    corr = events[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                square=True, linewidths=0.5)
    ax.set_title("Q6: Feature Correlations (Events)", fontsize=14, fontweight="bold")
    save(fig, "08_correlation")


# --- Q7: Target port patterns ---

def plot_port_analysis(events: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Top ports overall
    top_ports = events["Port number"].value_counts().head(10)
    axes[0].barh(range(len(top_ports)), top_ports.values)
    axes[0].set_yticks(range(len(top_ports)))
    axes[0].set_yticklabels([str(p) for p in top_ports.index])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Count")
    axes[0].set_title("Top 10 Target Ports (all events)")

    # Top ports for DDoS specifically
    ddos = events[events["Type"] == "DDoS attack"]
    ddos_ports = ddos["Port number"].value_counts().head(10)
    axes[1].barh(range(len(ddos_ports)), ddos_ports.values, color="#e74c3c")
    axes[1].set_yticks(range(len(ddos_ports)))
    axes[1].set_yticklabels([str(p) for p in ddos_ports.index])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Count")
    axes[1].set_title("Top 10 Target Ports (DDoS only)")

    fig.suptitle("Q7: Target Port Patterns", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "09_port_analysis")


# --- Q8: Victim IP concentration ---

def plot_victim_concentration(events: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall: cumulative event share by victim IP
    ip_counts = events["Victim IP"].value_counts()
    cumshare = ip_counts.cumsum() / ip_counts.sum() * 100
    axes[0].plot(range(1, len(cumshare) + 1), cumshare.values)
    axes[0].axhline(80, color="red", linestyle="--", alpha=0.5, label="80%")
    n80 = (cumshare <= 80).sum()
    axes[0].axvline(n80, color="red", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("Number of Victim IPs (ranked)")
    axes[0].set_ylabel("Cumulative % of events")
    axes[0].set_title(f"IP Concentration ({n80} IPs cover 80% of events)")
    axes[0].legend()

    # DDoS: top victim IPs
    ddos = events[events["Type"] == "DDoS attack"]
    top_ips = ddos["Victim IP"].value_counts().head(10)
    axes[1].barh(range(len(top_ips)), top_ips.values, color="#e74c3c")
    axes[1].set_yticks(range(len(top_ips)))
    axes[1].set_yticklabels(top_ips.index)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("DDoS event count")
    axes[1].set_title("Top 10 DDoS Victim IPs")

    fig.suptitle("Q8: Victim IP Concentration", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "10_victim_concentration")


def main() -> None:
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    SVG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    events = load_events("train")
    components = load_components("train")

    print(f"Events: {events.shape}, Components: {components.shape}\n")

    print("Generating plots...")
    plot_class_imbalance(events)
    plot_feature_comparison(events)
    plot_feature_distributions(events)
    plot_attack_types(events, components)
    plot_temporal_patterns(events)
    plot_event_duration(events)
    plot_components_per_event(events)
    plot_correlation(events)
    plot_port_analysis(events)
    plot_victim_concentration(events)

    print("\nDone. All plots saved to plots/")


if __name__ == "__main__":
    main()
