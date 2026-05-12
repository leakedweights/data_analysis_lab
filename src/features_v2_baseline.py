"""Frozen snapshot of the v2 featurizer **before** Fix A+B.

Kept around so we can reproduce the original baseline numbers for
side-by-side plots and benchmarks. Do not edit alongside ``features_v2``;
this is a fixed reference that intentionally lacks:

  * the four scale-invariant ratio features (Fix A): ``pps_cv``,
    ``pps_skew_proxy``, ``bytes_per_pkt_range``, ``bytes_to_pkts_ratio``
  * the log1p compression on the absolute pps features (Fix B)

It still keeps the dropped ``bps_estimate`` column so the resulting
matrix matches the original 18-feature v2 design.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features_v2 import FeatureMatrix, _AMPLIFICATION_PORTS, _DNS_PORT, \
    _EPHEMERAL_PORT_MIN, _HTTP_PORTS, _HTTPS_PORTS
from src.simulator import TYPE_TO_INT

FEATURE_NAMES_BASELINE: list[str] = [
    "pps_mean",
    "pps_max",
    "pps_p95",
    "pps_std",
    "pps_burstiness",
    "bytes_per_pkt_mean",
    "bytes_per_pkt_max",
    "bytes_per_pkt_min",
    "bps_estimate",
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


def _component_aggregates_baseline(components: pd.DataFrame) -> pd.DataFrame:
    pps = components["Packet speed"].astype(np.float64)
    plen = components["Avg packet len"].astype(np.float64)
    if "Source IP count" in components.columns:
        sip = components["Source IP count"].astype(np.float64)
    elif "Avg source IP count" in components.columns:
        sip = components["Avg source IP count"].astype(np.float64)
    else:
        sip = pd.Series(np.zeros(len(components)), index=components.index)

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
    out["pps_burstiness"] = out["pps_max"] / out["pps_mean"].clip(lower=1.0)
    out["bps_estimate"] = out["pps_mean"] * out["bytes_per_pkt_mean"]

    for c in ("src_ip_mean", "src_ip_max", "src_ip_std"):
        out[c] = np.log1p(out[c].clip(lower=0))
    return out.reset_index()


def featurize_baseline(
    events: pd.DataFrame,
    components: pd.DataFrame,
    label_col: str | None = "Type",
) -> FeatureMatrix:
    if "Attack ID" not in events.columns:
        raise ValueError("events must contain 'Attack ID'")
    if "Attack ID" not in components.columns:
        raise ValueError("components must contain 'Attack ID'")

    aggs = _component_aggregates_baseline(components)
    merged = events[["Attack ID", "Port number"] + ([label_col] if label_col else [])]
    merged = merged.merge(aggs, on="Attack ID", how="left")

    if "Packet speed" in events.columns:
        fallback_pps = events.set_index("Attack ID")["Packet speed"].astype(np.float64)
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

    merged["pps_std"] = merged["pps_std"].fillna(0.0)
    merged["pps_burstiness"] = merged["pps_burstiness"].fillna(1.0)
    merged["bps_estimate"] = merged["bps_estimate"].fillna(
        merged["pps_mean"] * merged["bytes_per_pkt_mean"]
    )

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

    numeric_cols = [c for c in merged.columns
                    if c != "Attack ID" and c != label_col]
    for c in numeric_cols:
        if merged[c].isna().any():
            merged[c] = merged[c].fillna(0.0)

    port = merged["Port number"].astype(np.int64)
    merged["is_port_zero"] = (port == 0).astype(np.int8)
    merged["is_port_dns"] = (port == _DNS_PORT).astype(np.int8)
    merged["is_amplification_port"] = port.isin(_AMPLIFICATION_PORTS).astype(np.int8)
    merged["is_http_port"] = port.isin(_HTTP_PORTS).astype(np.int8)
    merged["is_https_port"] = port.isin(_HTTPS_PORTS).astype(np.int8)
    merged["is_ephemeral_port"] = (port >= _EPHEMERAL_PORT_MIN).astype(np.int8)

    X = merged[FEATURE_NAMES_BASELINE].to_numpy(dtype=np.float64)
    if label_col and label_col in merged.columns:
        y = merged[label_col].astype(str).map(TYPE_TO_INT).to_numpy(dtype=np.int64)
    else:
        y = None
    return FeatureMatrix(
        X=X, y=y,
        feature_names=list(FEATURE_NAMES_BASELINE),
        attack_ids=merged["Attack ID"].to_numpy(),
    )
