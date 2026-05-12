"""Publication-quality figures for DDoS detection data analysis.

Three narrative sections:
  1. Adathalmaz jellemzése — Dataset composition & structure
  2. Támadások karakterizálása — Attack characterization
  3. Támadások szegmentálása — Attack segmentation & separability

Figures are styled for Nature-level legibility: clean spines, muted
palette, monospaced annotations, panel labels in bold.

Usage:
    uv run python -m src.eda_figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils.data_pipeline import load_components, load_events

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUT = Path(__file__).resolve().parents[1] / "results" / "figures"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
TYPE_ORDER = ["Normal traffic", "Suspicious traffic", "DDoS attack"]
TYPE_SHORT = {"Normal traffic": "Normal", "Suspicious traffic": "Suspicious", "DDoS attack": "DDoS"}
PAL = {
    "Normal traffic": "#3C7EA6",
    "Suspicious traffic": "#D4943A",
    "DDoS attack": "#B83A3A",
}
PAL_LIGHT = {
    "Normal traffic": "#3C7EA620",
    "Suspicious traffic": "#D4943A20",
    "DDoS attack": "#B83A3A20",
}

FEATURE_COLS = [
    "Packet speed", "Data speed", "Avg packet len",
    "Avg source IP count", "Detect count",
]
FEATURE_NICE = {
    "Packet speed": "Packet rate",
    "Data speed": "Data rate",
    "Avg packet len": "Avg packet length",
    "Avg source IP count": "Source IP count",
    "Detect count": "Component count",
}


def _setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "axes.grid": False,
    })


def _panel_label(ax, label, x=-0.08, y=1.06):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")


def _save(fig, name):
    for fmt in ("png", "svg", "pdf"):
        d = OUT / fmt
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.{fmt}")
    plt.close(fig)
    print(f"  ✓ {name}")


def _thousands(x, _):
    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if x >= 1_000:
        return f"{x / 1_000:.0f}k"
    return f"{int(x)}"


# ===================================================================
# SECTION 1 — Adathalmaz jellemzése
# ===================================================================

def fig_01_composition(splits: dict[str, pd.DataFrame],
                       comp_splits: dict[str, pd.DataFrame] | None = None):
    """Class distribution across splits — events and components, absolute + proportional."""
    has_comp = comp_splits is not None
    n_rows = 2 if has_comp else 1
    fig = plt.figure(figsize=(7.5, 3.5 * n_rows + 0.5))
    gs = gridspec.GridSpec(n_rows, 2, width_ratios=[1.1, 1], wspace=0.35,
                           hspace=0.45)

    split_names = ["Train", "Test", "Genericity"]
    split_keys = ["train", "test", "genericity"]
    y_pos = np.arange(len(split_names))

    def _get_type_counts(data_dict, sk):
        """Return per-class counts. Components need attack_code→type mapping."""
        df = data_dict[sk]
        if "Type" in df.columns:
            return {t: int((df["Type"] == t).sum()) for t in TYPE_ORDER}, len(df)
        # Components: map via attack code from events
        code_type = splits[sk].groupby("Attack code")["Type"].first()
        mapped = df["Attack code"].map(code_type)
        return {t: int((mapped == t).sum()) for t in TYPE_ORDER}, len(df)

    def _draw_row(row, data_dict, label_prefix, panel_a, panel_b):
        ax_abs = fig.add_subplot(gs[row, 0])
        ax_pct = fig.add_subplot(gs[row, 1])

        for i, sk in enumerate(split_keys):
            counts, total = _get_type_counts(data_dict, sk)
            # Absolute
            left = 0
            for t in TYPE_ORDER:
                ax_abs.barh(i, counts[t], left=left, color=PAL[t], height=0.6,
                            edgecolor="white", linewidth=0.4)
                left += counts[t]
            # Annotations to the right
            for j, t in enumerate(TYPE_ORDER):
                offset = 0.18 - j * 0.18
                ax_abs.text(total * 1.02, i + offset, f"{counts[t]:,}",
                            va="center", fontsize=5.5, color=PAL[t])

            # Proportional
            left_pct = 0
            for t in TYPE_ORDER:
                pct = counts[t] / total * 100
                ax_pct.barh(i, pct, left=left_pct, color=PAL[t], height=0.6,
                            edgecolor="white", linewidth=0.4)
                if pct > 3:
                    ax_pct.text(left_pct + pct / 2, i, f"{pct:.1f}%",
                                ha="center", va="center", fontsize=6,
                                color="white", fontweight="bold")
                left_pct += pct

        ax_abs.set_yticks(y_pos)
        ax_abs.set_yticklabels(split_names)
        ax_abs.set_xlabel(f"Number of {label_prefix}")
        ax_abs.xaxis.set_major_formatter(mticker.FuncFormatter(_thousands))
        ax_abs.set_xlim(right=ax_abs.get_xlim()[1] * 1.22)
        ax_abs.set_title(f"{label_prefix.capitalize()} — absolute counts")
        _panel_label(ax_abs, panel_a)

        ax_pct.set_yticks(y_pos)
        ax_pct.set_yticklabels(split_names)
        ax_pct.set_xlabel("Proportion (%)")
        ax_pct.set_xlim(0, 100)
        ax_pct.set_title(f"{label_prefix.capitalize()} — class proportions")
        _panel_label(ax_pct, panel_b)

    # Row 0: events
    _draw_row(0, splits, "events", "a", "b")

    # Row 1: components
    if has_comp:
        _draw_row(1, comp_splits, "components", "c", "d")

    # Legend
    handles = [mpl.patches.Patch(facecolor=PAL[t], label=TYPE_SHORT[t]) for t in TYPE_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01), fontsize=7.5)

    fig.suptitle("Dataset composition across splits", fontsize=10, fontweight="bold", y=1.0)
    _save(fig, "01_composition")


def fig_01b_pie_charts(splits: dict[str, pd.DataFrame],
                       comp_splits: dict[str, pd.DataFrame]):
    """Pie charts showing class distribution per split — events and components."""
    split_names = ["Train", "Test", "Genericity"]
    split_keys = ["train", "test", "genericity"]

    fig, axes = plt.subplots(2, 3, figsize=(9, 6.0))
    fig.subplots_adjust(wspace=0.12, hspace=0.35)

    def _draw_pie(ax, counts, title_line):
        total = counts.sum()
        colors = [PAL[t] for t in TYPE_ORDER]
        wedges, texts, autotexts = ax.pie(
            counts.values,
            colors=colors,
            autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
            startangle=90,
            pctdistance=0.55,
            wedgeprops=dict(edgecolor="white", linewidth=1.2),
        )
        for at in autotexts:
            at.set_fontsize(7)
            at.set_fontweight("bold")
            at.set_color("white")
        # Annotate small slices outside
        for wedge, t, n in zip(wedges, TYPE_ORDER, counts.values):
            pct = n / total * 100
            if pct <= 2:
                ang = (wedge.theta2 + wedge.theta1) / 2
                x = 1.35 * np.cos(np.radians(ang))
                y = 1.35 * np.sin(np.radians(ang))
                ax.annotate(
                    f"{TYPE_SHORT[t]}\n{pct:.1f}%",
                    xy=(0.9 * np.cos(np.radians(ang)), 0.9 * np.sin(np.radians(ang))),
                    xytext=(x, y),
                    fontsize=5.5, ha="center", va="center", color=PAL[t],
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=PAL[t], lw=0.6),
                )
        ax.set_title(title_line, fontsize=7.5, fontweight="bold")

    # Row 0: events
    for col, sk, name in zip(range(3), split_keys, split_names):
        ev = splits[sk]
        counts = ev["Type"].value_counts().reindex(TYPE_ORDER)
        _draw_pie(axes[0, col], counts, f"{name} — events\n({len(ev):,})")

    # Row 1: components (components don't have "Type", need to join via Attack ID or use Attack code)
    # Components share Attack code with events; map Attack code → Type from events
    for col, sk, name in zip(range(3), split_keys, split_names):
        ev = splits[sk]
        co = comp_splits[sk]
        # Build attack_code → type mapping from events
        code_type = ev.groupby("Attack code")["Type"].first()
        co_type = co["Attack code"].map(code_type)
        counts = co_type.value_counts().reindex(TYPE_ORDER).fillna(0).astype(int)
        _draw_pie(axes[1, col], counts, f"{name} — components\n({len(co):,})")

    handles = [mpl.patches.Patch(facecolor=PAL[t], label=TYPE_SHORT[t]) for t in TYPE_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01), fontsize=7.5)
    fig.suptitle("Class distribution — events vs components", fontsize=10,
                 fontweight="bold", y=1.0)
    _save(fig, "01b_pie_charts")


def fig_02_feature_distributions(ev: pd.DataFrame):
    """Ridge-plot showing feature distributions by traffic type."""
    fig, axes = plt.subplots(len(FEATURE_COLS), 1, figsize=(7.2, 7.5),
                             sharex=False)
    fig.subplots_adjust(hspace=0.55)

    for row, col in enumerate(FEATURE_COLS):
        ax = axes[row]
        vals = ev[col].dropna()
        # Use log1p for highly skewed features
        use_log = vals.max() > 10 * vals.median() and vals.median() > 0

        for t in TYPE_ORDER:
            subset = ev.loc[ev["Type"] == t, col].dropna()
            if use_log:
                subset = np.log1p(subset)
            clipped = subset.clip(upper=subset.quantile(0.995))
            ax.hist(clipped, bins=80, density=True, alpha=0.35,
                    color=PAL[t], label=TYPE_SHORT[t])
            # KDE overlay
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(clipped, bw_method=0.15)
                xs = np.linspace(clipped.min(), clipped.max(), 300)
                ax.plot(xs, kde(xs), color=PAL[t], linewidth=1.2)
            except Exception:
                pass

        nice = FEATURE_NICE.get(col, col)
        suffix = " (log₁₊ₓ)" if use_log else ""
        ax.set_ylabel("Density", fontsize=7)
        ax.set_title(f"{nice}{suffix}", fontsize=8, fontweight="bold", loc="left")
        ax.tick_params(axis="y", labelsize=6)
        if row == 0:
            ax.legend(loc="upper right", fontsize=6.5)

    fig.suptitle("Feature distributions by traffic type", fontsize=10,
                 fontweight="bold", y=1.0)
    _save(fig, "02_feature_distributions")


def fig_03_correlation_structure(ev: pd.DataFrame):
    """Correlation heatmap with hierarchical clustering dendrogram."""
    data = ev[FEATURE_COLS].dropna()
    corr = data.corr()
    nice_labels = [FEATURE_NICE.get(c, c) for c in FEATURE_COLS]

    fig = plt.figure(figsize=(5.5, 5.0))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 5], width_ratios=[5, 1],
                           hspace=0.02, wspace=0.02)
    ax_dend = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[1, 0])
    ax_cbar = fig.add_subplot(gs[1, 1])

    # Dendrogram
    Z = linkage(corr, method="ward")
    dn = dendrogram(Z, ax=ax_dend, labels=nice_labels, no_labels=True,
                    color_threshold=0, above_threshold_color="#3C7EA6")
    ax_dend.set_xticks([])
    ax_dend.set_yticks([])
    for spine in ax_dend.spines.values():
        spine.set_visible(False)

    # Reorder correlation matrix by dendrogram
    order = dn["leaves"]
    corr_ord = corr.iloc[order, order]
    labels_ord = [nice_labels[i] for i in order]

    # Heatmap
    im = ax_heat.imshow(corr_ord.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax_heat.set_xticks(range(len(labels_ord)))
    ax_heat.set_xticklabels(labels_ord, rotation=35, ha="right", fontsize=7)
    ax_heat.set_yticks(range(len(labels_ord)))
    ax_heat.set_yticklabels(labels_ord, fontsize=7)

    # Annotate cells
    for i in range(len(labels_ord)):
        for j in range(len(labels_ord)):
            val = corr_ord.iloc[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax_heat.text(j, i, f"{val:.2f}", ha="center", va="center",
                         fontsize=6.5, color=color)

    plt.colorbar(im, cax=ax_cbar, label="Pearson r")
    _panel_label(ax_heat, "", x=-0.15)
    fig.suptitle("Feature correlation structure", fontsize=10, fontweight="bold")
    _save(fig, "03_correlation")


# ===================================================================
# SECTION 2 — Támadások karakterizálása
# ===================================================================

def fig_04_attack_taxonomy(ev: pd.DataFrame):
    """Attack code frequency as a grouped lollipop chart."""
    type_frames = []
    for t in TYPE_ORDER:
        sub = ev[ev["Type"] == t]
        counts = sub["Attack code"].value_counts()
        counts = counts[counts > 0].head(8)
        for code, n in counts.items():
            type_frames.append({"type": t, "code": str(code), "count": n})
    df = pd.DataFrame(type_frames)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.5), sharey=False)
    fig.subplots_adjust(wspace=0.7)

    for ax, t in zip(axes, TYPE_ORDER):
        sub = df[df["type"] == t].sort_values("count")
        y = list(range(len(sub)))
        counts = sub["count"].values
        labels = []
        for _, row in sub.iterrows():
            lb = row["code"]
            if len(lb) > 28:
                lb = lb[:26] + "…"
            labels.append(f"{lb}  ({row['count']:,})")

        ax.barh(y, counts, color=PAL[t], height=0.55,
                edgecolor="white", linewidth=0.3, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=5.5)
        ax.set_xlabel("Events")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_thousands))
        ax.set_title(TYPE_SHORT[t], color=PAL[t], fontsize=9, fontweight="bold")

    _panel_label(axes[0], "a", x=-0.7)
    _panel_label(axes[1], "b", x=-0.35)
    _panel_label(axes[2], "c", x=-0.35)

    fig.suptitle("Attack code taxonomy by traffic type", fontsize=10,
                 fontweight="bold", y=1.0)
    _save(fig, "04_attack_taxonomy")


def fig_05_attack_fingerprints(ev: pd.DataFrame):
    """Z-score heatmap of feature means per attack code, clustered."""
    # Top attack codes by count (excluding the dominant "High volume traffic")
    codes = ev["Attack code"].value_counts()
    top_codes = codes.head(15).index.tolist()

    rows = []
    for code in top_codes:
        sub = ev[ev["Attack code"] == code]
        means = sub[FEATURE_COLS].mean()
        means["_type"] = sub["Type"].mode().iloc[0]
        means["_code"] = code
        means["_n"] = len(sub)
        rows.append(means)
    df = pd.DataFrame(rows)

    # Z-score normalize feature columns
    feat_vals = df[FEATURE_COLS].values.astype(float)
    zscores = (feat_vals - feat_vals.mean(axis=0)) / (feat_vals.std(axis=0) + 1e-9)
    nice_features = [FEATURE_NICE.get(c, c) for c in FEATURE_COLS]

    # Cluster rows
    Z = linkage(zscores, method="ward")
    dn = dendrogram(Z, no_plot=True)
    order = dn["leaves"]

    fig = plt.figure(figsize=(7.2, 5.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[0.8, 5, 0.3], wspace=0.05)
    ax_type = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])
    ax_cbar = fig.add_subplot(gs[2])

    z_ordered = zscores[order]
    labels_ordered = [df.iloc[i]["_code"] for i in order]
    types_ordered = [df.iloc[i]["_type"] for i in order]
    counts_ordered = [df.iloc[i]["_n"] for i in order]

    # Shorten long labels
    short_labels = []
    for lb, n in zip(labels_ordered, counts_ordered):
        s = lb if len(lb) <= 32 else lb[:30] + "…"
        short_labels.append(f"{s}  (n={int(n):,})")

    # Type color strip
    for i, t in enumerate(types_ordered):
        ax_type.barh(i, 1, color=PAL[t], height=0.9, edgecolor="white", linewidth=0.3)
    ax_type.set_yticks(range(len(short_labels)))
    ax_type.set_yticklabels(short_labels, fontsize=6.5)
    ax_type.set_xticks([])
    ax_type.set_xlim(0, 1)
    ax_type.invert_yaxis()
    for spine in ax_type.spines.values():
        spine.set_visible(False)

    # Heatmap
    vmax = max(abs(z_ordered.min()), abs(z_ordered.max()), 2.5)
    im = ax_heat.imshow(z_ordered, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax_heat.set_xticks(range(len(nice_features)))
    ax_heat.set_xticklabels(nice_features, rotation=35, ha="right", fontsize=7)
    ax_heat.set_yticks([])
    ax_heat.invert_yaxis()

    # Annotate cells
    for i in range(z_ordered.shape[0]):
        for j in range(z_ordered.shape[1]):
            val = z_ordered[i, j]
            color = "white" if abs(val) > 1.5 else "black"
            ax_heat.text(j, i, f"{val:.1f}", ha="center", va="center",
                         fontsize=6, color=color)

    plt.colorbar(im, cax=ax_cbar, label="Z-score")

    # Legend for type strip
    handles = [mpl.patches.Patch(facecolor=PAL[t], label=TYPE_SHORT[t]) for t in TYPE_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.03), fontsize=7)

    fig.suptitle("Attack fingerprints — feature profiles by attack code",
                 fontsize=10, fontweight="bold", y=1.0)
    _save(fig, "05_fingerprints")


def fig_06_temporal_targeting(ev: pd.DataFrame):
    """Duration distributions + port targeting heatmap."""
    fig = plt.figure(figsize=(10, 5.0))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.3], wspace=0.35)
    ax_dur = fig.add_subplot(gs[0])
    ax_port = fig.add_subplot(gs[1])

    # --- Panel a: Duration distribution (violin) ---
    df = ev.dropna(subset=["Start time", "End time"]).copy()
    df["duration_s"] = (df["End time"] - df["Start time"]).dt.total_seconds()
    df = df[df["duration_s"] > 0]

    parts = ax_dur.violinplot(
        [np.log10(df.loc[df["Type"] == t, "duration_s"].clip(lower=0.1).values)
         for t in TYPE_ORDER],
        positions=range(3), showmeans=True, showmedians=True, widths=0.7,
    )
    for i, (body, t) in enumerate(zip(parts["bodies"], TYPE_ORDER)):
        body.set_facecolor(PAL[t])
        body.set_alpha(0.6)
    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("#333")
            parts[key].set_linewidth(0.8)

    ax_dur.set_xticks(range(3))
    ax_dur.set_xticklabels([TYPE_SHORT[t] for t in TYPE_ORDER], fontsize=7)
    ax_dur.set_ylabel("Duration (log₁₀ seconds)")
    ax_dur.set_title("Event duration", fontsize=9, fontweight="bold")
    _panel_label(ax_dur, "a")

    # --- Panel b: Port targeting heatmap (type × top ports) ---
    top_ports = ev["Port number"].value_counts().head(10).index.tolist()
    heatdata = []
    for t in TYPE_ORDER:
        sub = ev[ev["Type"] == t]
        total = len(sub)
        row = []
        for p in top_ports:
            pct = (sub["Port number"] == p).sum() / total * 100
            row.append(pct)
        heatdata.append(row)

    hm = np.array(heatdata)
    im = ax_port.imshow(hm, cmap="YlOrRd", aspect="auto")
    ax_port.set_xticks(range(len(top_ports)))
    ax_port.set_xticklabels([str(p) for p in top_ports], rotation=45, ha="right", fontsize=7)
    ax_port.set_yticks(range(3))
    ax_port.set_yticklabels([TYPE_SHORT[t] for t in TYPE_ORDER], fontsize=7)
    ax_port.set_xlabel("Port number")
    ax_port.set_title("Port targeting (% of events per type)", fontsize=9, fontweight="bold")

    for i in range(3):
        for j in range(len(top_ports)):
            val = hm[i, j]
            color = "white" if val > hm.max() * 0.6 else "black"
            ax_port.text(j, i, f"{val:.1f}", ha="center", va="center",
                         fontsize=6.5, color=color)

    plt.colorbar(im, ax=ax_port, shrink=0.6, label="%", pad=0.02)
    _panel_label(ax_port, "b")

    fig.suptitle("Temporal dynamics and target selection",
                 fontsize=10, fontweight="bold", y=1.0)
    _save(fig, "06_temporal_targeting")


# ===================================================================
# SECTION 3 — Támadások szegmentálása
# ===================================================================

def fig_07_feature_space(ev: pd.DataFrame):
    """PCA projection with marginal densities and confidence ellipses."""
    from matplotlib.patches import Ellipse
    np.random.seed(42)

    # Subsample for visual clarity
    n_sample = min(15_000, len(ev))
    sample = ev.sample(n_sample, random_state=42).copy()

    # Clip features at 99th percentile before scaling to tame outliers
    X_raw = sample[FEATURE_COLS].values.astype(float)
    p99 = np.percentile(X_raw, 99, axis=0)
    X_clipped = np.minimum(X_raw, p99)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clipped)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    sample["PC1"] = coords[:, 0]
    sample["PC2"] = coords[:, 1]

    # Further trim view to 1st–99th percentile of PC space
    pc1_lo, pc1_hi = np.percentile(coords[:, 0], [1, 99])
    pc2_lo, pc2_hi = np.percentile(coords[:, 1], [1, 99])
    pad1 = (pc1_hi - pc1_lo) * 0.1
    pad2 = (pc2_hi - pc2_lo) * 0.1

    fig = plt.figure(figsize=(7.2, 6.5))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 4], width_ratios=[4, 1],
                           hspace=0.04, wspace=0.04)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_inset = fig.add_subplot(gs[0, 1])

    # --- Main scatter ---
    for t in TYPE_ORDER:
        mask = sample["Type"] == t
        ax_main.scatter(sample.loc[mask, "PC1"], sample.loc[mask, "PC2"],
                        c=PAL[t], s=4, alpha=0.35, label=TYPE_SHORT[t],
                        rasterized=True, edgecolors="none")

    # 95% confidence ellipses (use trimmed data for robust covariance)
    for t in TYPE_ORDER:
        mask = sample["Type"] == t
        xd = sample.loc[mask, "PC1"].values
        yd = sample.loc[mask, "PC2"].values
        if len(xd) < 10:
            continue
        # Trim to IQR for robust ellipse
        qx = np.percentile(xd, [5, 95])
        qy = np.percentile(yd, [5, 95])
        trim = (xd >= qx[0]) & (xd <= qx[1]) & (yd >= qy[0]) & (yd <= qy[1])
        xt, yt = xd[trim], yd[trim]
        if len(xt) < 10:
            continue
        cov = np.cov(xt, yt)
        eigvals, eigvecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
        w, h = 2 * np.sqrt(np.maximum(eigvals, 0) * 5.991)
        ell = Ellipse(xy=(xt.mean(), yt.mean()), width=w, height=h, angle=angle,
                      edgecolor=PAL[t], facecolor=PAL[t], linewidth=1.5,
                      linestyle="--", alpha=0.08)
        ax_main.add_patch(ell)

    ax_main.set_xlim(pc1_lo - pad1, pc1_hi + pad1)
    ax_main.set_ylim(pc2_lo - pad2, pc2_hi + pad2)

    var1, var2 = pca.explained_variance_ratio_[:2] * 100
    ax_main.set_xlabel(f"PC 1 ({var1:.1f}% variance)")
    ax_main.set_ylabel(f"PC 2 ({var2:.1f}% variance)")
    ax_main.legend(loc="lower right", fontsize=7, markerscale=3)

    # --- Marginal KDEs ---
    for t in TYPE_ORDER:
        mask = sample["Type"] == t
        ax_top.hist(sample.loc[mask, "PC1"], bins=60, density=True,
                    alpha=0.4, color=PAL[t], edgecolor="none")
        ax_right.hist(sample.loc[mask, "PC2"], bins=60, density=True,
                      alpha=0.4, color=PAL[t], orientation="horizontal",
                      edgecolor="none")

    ax_top.set_yticks([])
    ax_top.tick_params(labelbottom=False)
    for spine in ax_top.spines.values():
        spine.set_visible(False)
    ax_right.set_xticks([])
    ax_right.tick_params(labelleft=False)
    for spine in ax_right.spines.values():
        spine.set_visible(False)

    # --- Inset: explained variance ---
    n_comp = min(5, len(FEATURE_COLS))
    pca_full = PCA(n_components=n_comp).fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100
    ax_inset.bar(range(1, n_comp + 1), pca_full.explained_variance_ratio_ * 100,
                 color="#3C7EA6", alpha=0.7, edgecolor="white", linewidth=0.3)
    ax_inset.plot(range(1, n_comp + 1), cumvar, "o-", color="#B83A3A",
                  markersize=3, linewidth=1)
    ax_inset.set_xlabel("PC", fontsize=6)
    ax_inset.set_ylabel("Var %", fontsize=6)
    ax_inset.tick_params(labelsize=6)
    ax_inset.set_title("Explained var.", fontsize=7)

    fig.suptitle("Feature-space structure — PCA projection",
                 fontsize=10, fontweight="bold", y=0.98)
    _save(fig, "07_feature_space")


def fig_08_pairwise_separability(ev: pd.DataFrame):
    """Best separating feature pairs as scatter + contour plots."""
    np.random.seed(42)
    # Compute per-feature discriminative power (ratio of between/within class variance)
    scores = {}
    for col in FEATURE_COLS:
        groups = [ev.loc[ev["Type"] == t, col].dropna().values for t in TYPE_ORDER]
        grand_mean = ev[col].mean()
        between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        within = sum(g.var() * len(g) for g in groups)
        scores[col] = between / (within + 1e-9)

    ranked = sorted(scores, key=scores.get, reverse=True)
    # Top 3 pairs from top 4 features
    pairs = [(ranked[0], ranked[1]), (ranked[0], ranked[2]), (ranked[1], ranked[3])]

    n_sample = min(10_000, len(ev))
    sample = ev.sample(n_sample, random_state=42).copy()

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    fig.subplots_adjust(wspace=0.38)

    for idx, (ax, (fx, fy)) in enumerate(zip(axes, pairs)):
        # Clip both features to 99th percentile for clean view
        xvals = sample[fx].clip(upper=sample[fx].quantile(0.99))
        yvals = sample[fy].clip(upper=sample[fy].quantile(0.99))

        for t in TYPE_ORDER:
            mask = sample["Type"] == t
            ax.scatter(xvals[mask], yvals[mask],
                       c=PAL[t], s=5, alpha=0.3, label=TYPE_SHORT[t],
                       rasterized=True, edgecolors="none")

        ax.set_xlabel(FEATURE_NICE.get(fx, fx), fontsize=7)
        ax.set_ylabel(FEATURE_NICE.get(fy, fy), fontsize=7)
        if idx == 2:
            ax.legend(loc="best", fontsize=6, markerscale=3)
        _panel_label(ax, chr(ord("a") + idx), x=-0.14)

        # Annotate discriminative score
        sx = scores[fx]
        sy = scores[fy]
        ax.text(0.97, 0.03, f"F={sx:.2f} × {sy:.2f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=5.5, color="#64748b")

    fig.suptitle("Pairwise feature separability — best discriminating pairs",
                 fontsize=10, fontweight="bold", y=1.02)
    _save(fig, "08_pairwise_separability")


def _ddos_families(ddos: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Assign DDoS events to attack families. Returns (ddos_with_family, family_order, palette)."""
    def _family(code: str) -> str:
        c = str(code).lower()
        if "syn" in c:
            return "SYN Flood"
        if "dns" in c:
            return "DNS Amplification"
        if "ntp" in c:
            return "NTP Amplification"
        if "udp" in c:
            return "UDP Flood"
        return "Other"

    ddos = ddos.copy()
    ddos["family"] = ddos["Attack code"].apply(_family)
    families = ["SYN Flood", "DNS Amplification", "NTP Amplification", "UDP Flood", "Other"]
    fam_pal = {
        "SYN Flood": "#e63946",
        "DNS Amplification": "#457b9d",
        "NTP Amplification": "#e9c46a",
        "UDP Flood": "#2a9d8f",
        "Other": "#94a3b8",
    }
    return ddos, families, fam_pal


def fig_09a_ddos_families(ev: pd.DataFrame):
    """DDoS attack family distribution."""
    ddos = ev[ev["Type"] == "DDoS attack"]
    if len(ddos) < 50:
        return
    ddos, families, fam_pal = _ddos_families(ddos)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    fam_counts = ddos["family"].value_counts().reindex(families).fillna(0).astype(int)
    y = list(range(len(families)))
    ax.barh(y, fam_counts.values, color=[fam_pal[f] for f in families],
            height=0.6, edgecolor="white", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(families, fontsize=8)
    ax.invert_yaxis()
    for yi, (fam, n) in enumerate(fam_counts.items()):
        pct = n / len(ddos) * 100
        ax.text(n + 12, yi, f"{n:,}  ({pct:.0f}%)", va="center",
                fontsize=7, fontweight="bold", color=fam_pal[fam])
    ax.set_xlabel("Events")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.25)
    fig.suptitle("DDoS attack families — distribution",
                 fontsize=10, fontweight="bold")
    _save(fig, "09a_ddos_families")


def fig_09b_ddos_signatures(ev: pd.DataFrame):
    """Feature fingerprint heatmap per DDoS family."""
    ddos = ev[ev["Type"] == "DDoS attack"]
    if len(ddos) < 50:
        return
    ddos, families, fam_pal = _ddos_families(ddos)

    feat_means = ddos.groupby("family")[FEATURE_COLS].mean().reindex(families)
    zscores = (feat_means - feat_means.mean()) / (feat_means.std() + 1e-9)
    nice_features = [FEATURE_NICE.get(c, c) for c in FEATURE_COLS]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    vmax = max(abs(zscores.values.min()), abs(zscores.values.max()), 2.0)
    im = ax.imshow(zscores.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto")
    ax.set_xticks(range(len(nice_features)))
    ax.set_xticklabels(nice_features, rotation=30, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(families, fontsize=8)
    for i in range(len(families)):
        for j in range(len(FEATURE_COLS)):
            val = zscores.iloc[i, j]
            color = "white" if abs(val) > 1.2 else "black"
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                    fontsize=7.5, color=color, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Z-score", pad=0.03)
    fig.suptitle("DDoS family feature signatures",
                 fontsize=10, fontweight="bold")
    _save(fig, "09b_ddos_signatures")


def fig_09c_ddos_scatter(ev: pd.DataFrame):
    """DDoS families on two key separating features."""
    ddos = ev[ev["Type"] == "DDoS attack"]
    if len(ddos) < 50:
        return
    ddos, families, fam_pal = _ddos_families(ddos)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    fig.subplots_adjust(wspace=0.3)

    pairs = [("Avg packet len", "Data speed"), ("Avg source IP count", "Detect count")]
    for ax, (xf, yf) in zip(axes, pairs):
        for fam in families:
            sub = ddos[ddos["family"] == fam]
            ax.scatter(
                sub[xf].clip(upper=sub[xf].quantile(0.98)),
                sub[yf].clip(upper=sub[yf].quantile(0.98)),
                c=fam_pal[fam], s=14, alpha=0.5, label=fam,
                edgecolors="white", linewidth=0.2, rasterized=True,
            )
        ax.set_xlabel(FEATURE_NICE.get(xf, xf), fontsize=7.5)
        ax.set_ylabel(FEATURE_NICE.get(yf, yf), fontsize=7.5)

    axes[1].legend(fontsize=6, loc="upper right", markerscale=1.5,
                   labelspacing=0.3)
    _panel_label(axes[0], "a")
    _panel_label(axes[1], "b")
    fig.suptitle("DDoS family separation on key features",
                 fontsize=10, fontweight="bold", y=1.02)
    _save(fig, "09c_ddos_scatter")


# ===================================================================
# SECTION 4 — Live vs Synthetic comparison
# ===================================================================

SOURCE_PAL = {"synthetic": "#3C7EA6", "hping3": "#B83A3A"}


def fig_10_live_vs_synthetic(
    synthetic: pd.DataFrame,
    live: pd.DataFrame,
    feature_cols: list[str] | None = None,
):
    """Compare feature distributions between synthetic and live (hping3) traffic.

    Call with two DataFrames that share the standard event schema.
    Produces a multi-panel figure with overlaid histograms for each feature
    plus a scatter panel of (Packet speed, Data speed) coloured by source.

    Usage::

        from src.synthetic import TrafficGenerator
        from src.live_capture import LiveTrafficGenerator, LIVE_SCENARIOS

        gen_syn = TrafficGenerator(seed=0)
        syn = gen_syn.generate_events(n=2000, ddos_ratio=0.3)

        gen_live = LiveTrafficGenerator()
        gen_live.generate_stream_live(LIVE_SCENARIOS["live_syn_flood"])
        live = pd.DataFrame(gen_live.pop_events())

        fig_10_live_vs_synthetic(syn, live)
    """
    _setup_style()
    cols = feature_cols or FEATURE_COLS
    n_feat = len(cols)

    fig, axes = plt.subplots(2, (n_feat + 1) // 2 + 1, figsize=(14, 6))
    axes = axes.flatten()

    # Histogram panels
    for i, col in enumerate(cols):
        ax = axes[i]
        for label, df, color in [
            ("Synthetic", synthetic, SOURCE_PAL["synthetic"]),
            ("hping3", live, SOURCE_PAL["hping3"]),
        ]:
            vals = df[col].dropna().values.astype(float)
            if len(vals) == 0:
                continue
            clipped = np.clip(vals, 0, np.percentile(vals, 99.5))
            ax.hist(
                clipped, bins=40, alpha=0.55, color=color,
                density=True, label=label, edgecolor="none",
            )
        ax.set_title(FEATURE_NICE.get(col, col), fontsize=8)
        ax.set_ylabel("Density", fontsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    # Scatter panel: Packet speed vs Data speed
    ax_sc = axes[n_feat]
    for label, df, color, marker in [
        ("Synthetic", synthetic, SOURCE_PAL["synthetic"], "o"),
        ("hping3", live, SOURCE_PAL["hping3"], "x"),
    ]:
        if "Packet speed" in df.columns and "Data speed" in df.columns:
            ax_sc.scatter(
                df["Packet speed"].values[:500],
                df["Data speed"].values[:500],
                c=color, alpha=0.4, s=10, marker=marker, label=label,
            )
    ax_sc.set_xlabel("Packet speed", fontsize=7)
    ax_sc.set_ylabel("Data speed", fontsize=7)
    ax_sc.set_title("Feature space (sample)", fontsize=8)
    ax_sc.legend(fontsize=7)

    # Hide unused axes
    for j in range(n_feat + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Live (hping3) vs Synthetic traffic — feature comparison",
        fontsize=10, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    _save(fig, "fig_10_live_vs_synthetic")


# ===================================================================
# MAIN
# ===================================================================

def main():
    _setup_style()
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    splits = {}
    comp_splits = {}
    for s in ("train", "test", "genericity"):
        splits[s] = load_events(s)
        comp_splits[s] = load_components(s)
        print(f"  {s}: {len(splits[s]):,} events, {len(comp_splits[s]):,} components")

    ev = splits["train"]

    print("\n── Section 1: Adathalmaz jellemzése ──")
    fig_01_composition(splits, comp_splits)
    fig_01b_pie_charts(splits, comp_splits)
    fig_02_feature_distributions(ev)
    fig_03_correlation_structure(ev)

    print("\n── Section 2: Támadások karakterizálása ──")
    fig_04_attack_taxonomy(ev)
    fig_05_attack_fingerprints(ev)
    fig_06_temporal_targeting(ev)

    print("\n── Section 3: Támadások szegmentálása ──")
    fig_07_feature_space(ev)
    fig_08_pairwise_separability(ev)
    fig_09a_ddos_families(ev)
    fig_09b_ddos_signatures(ev)
    fig_09c_ddos_scatter(ev)

    print(f"\nDone — figures saved to {OUT}/")


if __name__ == "__main__":
    main()
