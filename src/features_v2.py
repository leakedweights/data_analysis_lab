"""v2 featurizer — component statistics aggregated to event level.

This module is the single source of truth for v2 features. The training
script, the synthetic evaluation, and the live capture all call
``featurize`` so the feature semantics are identical across paths.

Design notes
------------
v1 trained on six raw event columns and used a different formula for
``Detect count`` and ``Avg source IP count`` at inference time, producing
train/serve skew. v2 fixes that by:

1. Computing all features from the *components* of each event (statistics
   over within-event detections), so the model learns rate distributions
   instead of single-point summaries.
2. Dropping features that cannot be measured identically in the live
   ``/proc/net/dev`` path: source-IP counts, attack-code metadata, and
   the original ``Detect count`` (which means "components in this event"
   in training but was reinterpreted as ``packets / 1000`` in live v1).
3. Keeping only features that survive the train ↔ live boundary with
   the same semantics.

The contract: any caller that produces an ``events`` table and a
``components`` table joined by ``Attack ID`` can call ``featurize`` and
get a model-ready ``(X, y, feature_names)`` triple. Live capture is
written to satisfy this contract by treating each window as an event
and each interface-counter sample as a component.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.simulator import TYPE_TO_INT

FEATURE_NAMES: list[str] = [
    # Absolute rate features — log1p-compressed (Fix B). Compressing all
    # multiplicative-scale features symmetrically with src_ip_* keeps
    # distance-based and linear models from being dominated by raw pps
    # magnitudes and shrinks the train-vs-genericity rate-shift gap.
    "pps_mean",
    "pps_max",
    "pps_p95",
    "pps_std",
    # Scale-invariant ratio features (Fix A) — robust to the ~22% pps
    # shift between train DDoS and genericity DDoS that drove the
    # DDoS→Normal misclassifications.
    "pps_burstiness",
    "pps_cv",
    "pps_skew_proxy",
    "bytes_per_pkt_mean",
    "bytes_per_pkt_max",
    "bytes_per_pkt_min",
    "bytes_per_pkt_range",
    "bytes_to_pkts_ratio",
    "src_ip_mean",
    "src_ip_max",
    "src_ip_std",
    "is_port_zero",
    "is_port_dns",
    "is_amplification_port",
    "is_http_port",
    "is_https_port",
    "is_ephemeral_port",
]

# Port groups. DNS is split out because it has massive legitimate
# traffic in addition to being an amplification vector; the remaining
# amplification ports are grouped since any of them flagged alone is
# rare and the model benefits from the pooled signal.
_DNS_PORT = 53
_AMPLIFICATION_PORTS = frozenset({
    123,    # NTP
    389,    # CLDAP
    1900,   # SSDP
    11211,  # memcached
    19,     # CHARGEN
    161,    # SNMP
    5683,   # CoAP
    3702,   # WSD
})
_HTTP_PORTS = frozenset({80, 8080})
_HTTPS_PORTS = frozenset({443, 8443})
_EPHEMERAL_PORT_MIN = 32768


@dataclass(frozen=True)
class FeatureMatrix:
    X: np.ndarray
    y: np.ndarray | None
    feature_names: list[str]
    attack_ids: np.ndarray


def _component_aggregates(components: pd.DataFrame) -> pd.DataFrame:
    """Roll components up into per-event statistics keyed by Attack ID."""
    pps = components["Packet speed"].astype(np.float64)
    plen = components["Avg packet len"].astype(np.float64)
    # Source IP count column is named differently in the two tables;
    # the live path uses ``Source IP count`` to match the components schema.
    if "Source IP count" in components.columns:
        sip = components["Source IP count"].astype(np.float64)
    elif "Avg source IP count" in components.columns:
        sip = components["Avg source IP count"].astype(np.float64)
    else:
        sip = pd.Series(np.zeros(len(components)), index=components.index)

    # Drop non-positive packet lengths so they don't poison min/mean.
    # Zero-length packets in this dataset are measurement artefacts
    # (every real packet has at least a header).
    plen_nonzero = plen.where(plen > 0)

    grouped = pd.DataFrame({
        "Attack ID": components["Attack ID"].values,
        "_pps": pps.values,
        "_plen": plen.values,
        "_plen_nz": plen_nonzero.values,
        "_sip": sip.values,
    }).groupby("Attack ID", sort=False)

    out = grouped.agg(
        pps_mean=("_pps", "mean"),
        pps_max=("_pps", "max"),
        pps_std=("_pps", "std"),
        pps_p95=("_pps", lambda s: float(np.percentile(s, 95)) if len(s) else 0.0),
        bytes_per_pkt_mean=("_plen_nz", "mean"),
        bytes_per_pkt_max=("_plen_nz", "max"),
        bytes_per_pkt_min=("_plen_nz", "min"),
        src_ip_mean=("_sip", "mean"),
        src_ip_max=("_sip", "max"),
        src_ip_std=("_sip", "std"),
    )

    out = out.fillna(0.0)

    # Scale-invariant ratio features (Fix A). Computed from raw absolute
    # values BEFORE the log1p transform below so the ratios stay in their
    # natural scale (peak/mean, std/mean) and aren't deformed by the log.
    out["pps_burstiness"] = out["pps_max"] / out["pps_mean"].clip(lower=1.0)
    out["pps_cv"] = out["pps_std"] / out["pps_mean"].clip(lower=1.0)
    # Tail-vs-bulk skew. Numerator: gap between absolute peak and the
    # 95th-percentile peak (large only when a few extreme outliers sit
    # above the bulk). Denominator: total spread between peak and mean.
    # Clipped to 1 because perfectly steady traffic gives 0/0; we want the
    # feature to be 0 in that case, not NaN.
    out["pps_skew_proxy"] = (
        (out["pps_max"] - out["pps_p95"])
        / (out["pps_max"] - out["pps_mean"]).clip(lower=1.0)
    )
    out["bytes_per_pkt_range"] = (
        (out["bytes_per_pkt_max"] - out["bytes_per_pkt_min"])
        / out["bytes_per_pkt_mean"].clip(lower=1.0)
    )
    # Small-packet-flood signal: bytes/pkt over packets/sec. DDoS has tiny
    # frames at high pps (low ratio); benign traffic has larger frames at
    # moderate pps (higher ratio). Stays informative under the gen DDoS
    # rate drop because both numerator and denominator scale together.
    out["bytes_to_pkts_ratio"] = (
        out["bytes_per_pkt_mean"] / out["pps_mean"].clip(lower=1.0)
    )

    # Log-compress absolute rate features (Fix B). pps_* spans
    # ~10^4..10^7 in raw values; KNN's StandardScaled distance and LR's
    # decision boundary are dominated by the raw magnitude, and the gen
    # DDoS distribution is ~22% lower than train — a multiplicative shift
    # that log1p halves into an additive one the model can absorb. The
    # bps_estimate column from v2 is dropped: it was pps_mean *
    # bytes_per_pkt_mean and the model can recover it from the two factor
    # features without giving it a 10^8-scale free pass on KNN distance.
    for c in ("pps_mean", "pps_max", "pps_p95", "pps_std"):
        out[c] = np.log1p(out[c].clip(lower=0))

    # Source IP count is heavy-tailed in training (DDoS up to ~18k) and
    # extreme in live ``--rand-source`` floods (~200k in 5 s). log1p
    # compresses both tails to the same scale so the model trained on
    # SCLDDoS2024 generalizes to live captures without retraining.
    for c in ("src_ip_mean", "src_ip_max", "src_ip_std"):
        out[c] = np.log1p(out[c].clip(lower=0))
    return out.reset_index()


def featurize(
    events: pd.DataFrame,
    components: pd.DataFrame,
    label_col: str | None = "Type",
) -> FeatureMatrix:
    """Build the v2 feature matrix from a paired events / components table.

    Parameters
    ----------
    events:
        One row per Attack ID with at least ``Attack ID`` and ``Port number``
        columns (and ``label_col`` if labels are wanted).
    components:
        Many rows per Attack ID with ``Packet speed`` and ``Avg packet len``.
        Joined to ``events`` on ``Attack ID``.
    label_col:
        Column in ``events`` containing the categorical traffic type.
        Pass ``None`` to skip label extraction (live inference).
    """
    if "Attack ID" not in events.columns:
        raise ValueError("events must contain 'Attack ID'")
    if "Attack ID" not in components.columns:
        raise ValueError("components must contain 'Attack ID'")

    aggs = _component_aggregates(components)

    # Left join so that events without components still produce a row.
    merged = events[["Attack ID", "Port number"] + ([label_col] if label_col else [])]
    merged = merged.merge(aggs, on="Attack ID", how="left")

    # Events with zero components fall back to event-level Packet speed
    # (a degenerate single-sample "distribution"). The aggregates path
    # log1p-transforms pps_*, so the fallback values must be log1p'd too
    # — otherwise events-without-components carry raw 10^5-scale values
    # while events-with-components carry log-scale ~10^1 values, and the
    # model sees two different feature semantics in one column.
    if "Packet speed" in events.columns:
        fallback_pps = np.log1p(
            events.set_index("Attack ID")["Packet speed"]
            .astype(np.float64).clip(lower=0)
        )
        for col in ("pps_mean", "pps_max", "pps_p95"):
            mask = merged[col].isna()
            if mask.any():
                merged.loc[mask, col] = merged.loc[mask, "Attack ID"].map(fallback_pps)
    if "Avg packet len" in events.columns:
        fallback_len = events.set_index("Attack ID")["Avg packet len"].astype(np.float64)
        for col in ("bytes_per_pkt_mean", "bytes_per_pkt_max", "bytes_per_pkt_min"):
            mask = merged[col].isna()
            if mask.any():
                merged.loc[mask, col] = merged.loc[mask, "Attack ID"].map(fallback_len)

    # Single-component fallbacks for the variability/ratio features:
    # zero variance, ratio = 1 (max == mean), no skew, zero range, and
    # bytes_to_pkts_ratio falls back to bytes_per_pkt_mean / event-level
    # Packet speed (still raw — the ratio is computed on raw scales).
    merged["pps_std"] = merged["pps_std"].fillna(0.0)
    merged["pps_burstiness"] = merged["pps_burstiness"].fillna(1.0)
    merged["pps_cv"] = merged["pps_cv"].fillna(0.0)
    merged["pps_skew_proxy"] = merged["pps_skew_proxy"].fillna(0.0)
    merged["bytes_per_pkt_range"] = merged["bytes_per_pkt_range"].fillna(0.0)
    if "Packet speed" in events.columns:
        raw_pps = events.set_index("Attack ID")["Packet speed"].astype(np.float64)
        fallback_b2p = (
            merged["bytes_per_pkt_mean"]
            / merged["Attack ID"].map(raw_pps).clip(lower=1.0)
        )
        mask = merged["bytes_to_pkts_ratio"].isna()
        if mask.any():
            merged.loc[mask, "bytes_to_pkts_ratio"] = fallback_b2p[mask]

    # Source IP fallback: events without components get the event-level
    # Avg source IP count as a single-sample stand-in. log1p to match
    # the transform applied inside ``_component_aggregates``.
    if "Avg source IP count" in events.columns:
        fallback_sip = np.log1p(
            events.set_index("Attack ID")["Avg source IP count"]
            .astype(np.float64).clip(lower=0)
        )
        for col in ("src_ip_mean", "src_ip_max"):
            mask = merged[col].isna()
            if mask.any():
                merged.loc[mask, col] = merged.loc[mask, "Attack ID"].map(fallback_sip)
    merged["src_ip_std"] = merged["src_ip_std"].fillna(0.0)

    # Fill numeric columns only — leave the categorical label column alone
    # (pandas 2.x rejects fillna(0.0) on a Categorical without an existing
    # category for 0.0).
    numeric_cols = [c for c in merged.columns
                    if c != "Attack ID" and c != label_col]
    for c in numeric_cols:
        if merged[c].isna().any():
            merged[c] = merged[c].fillna(0.0)

    # Port gets encoded as a set of service-specific binary flags.
    # Raw port number is not a feature — port 52 and port 53 are
    # different services, not neighbours on a linear scale, so
    # distance-based and linear models can't use the raw value.
    port = merged["Port number"].astype(np.int64)
    merged["is_port_zero"] = (port == 0).astype(np.int8)
    merged["is_port_dns"] = (port == _DNS_PORT).astype(np.int8)
    merged["is_amplification_port"] = port.isin(_AMPLIFICATION_PORTS).astype(np.int8)
    merged["is_http_port"] = port.isin(_HTTP_PORTS).astype(np.int8)
    merged["is_https_port"] = port.isin(_HTTPS_PORTS).astype(np.int8)
    merged["is_ephemeral_port"] = (port >= _EPHEMERAL_PORT_MIN).astype(np.int8)

    X = merged[FEATURE_NAMES].to_numpy(dtype=np.float64)

    if label_col and label_col in merged.columns:
        y = merged[label_col].astype(str).map(TYPE_TO_INT).to_numpy(dtype=np.int64)
    else:
        y = None

    return FeatureMatrix(
        X=X,
        y=y,
        feature_names=list(FEATURE_NAMES),
        attack_ids=merged["Attack ID"].to_numpy(),
    )
