"""Extra derived features stacked alongside the v2 A+B featurizer.

Returns a (N, 7) array aligned with the row order of ``events`` (by
``Attack ID``). The output is meant to be ``np.concatenate`` -d with
the matrix from ``features_v2.featurize``.

Columns
-------
1. ``n_components``           — component count per event (raw int).
2. ``log1p_n_components``     — log-compressed component count.
3. ``is_single_sample``       — 1 if exactly one component (the
                                 degenerate-stats regime); else 0.
4. ``pps_median_log``         — log1p of the median Packet speed
                                 across components.
5. ``pps_mad_log``            — log1p of the median absolute deviation
                                 of Packet speed across components.
6. ``bytes_per_pkt_median``   — median Avg packet len (excluding zeros).
7. ``bytes_per_pkt_mad``      — MAD of Avg packet len (excluding zeros).

Why these
---------
* **Component count** addresses the genericity finding that 79 % of
  DDoS events on SetD have ≤2 components, vs 40 % on SetC. When the
  count is low the std/p95/burstiness stats degenerate toward zero
  and the model cannot tell "a steady-rate event" from "a single
  sample with no variance". The count itself, plus the
  ``is_single_sample`` flag, give the model a way to gate trust in
  the variance-based v2 features.
* **Robust statistics (median, MAD)** are far less sensitive to
  outliers than mean/std when there are 2–3 components. They also
  give the model a non-zero signal where ``pps_std`` collapses to
  zero. The two statistics target the same underlying problem from
  a different angle than ``n_components``: where component count
  helps the model *gate* trust in the variance features, the robust
  stats give it an *alternative* signal it can lean on regardless.

Live capture compatibility
--------------------------
All seven columns are computable from the live capture path: each
window's components are the ``/proc/net/dev`` samples within it, so
``n_components`` ≤ 50 (5 s window / 100 ms sample period) and the
medians/MADs come from the same per-sample series. No new
dependencies, no train ↔ inference drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


EXTRA_FEATURE_NAMES: list[str] = [
    "n_components",
    "log1p_n_components",
    "is_single_sample",
    "pps_median_log",
    "pps_mad_log",
    "bytes_per_pkt_median",
    "bytes_per_pkt_mad",
]


def _mad(s: pd.Series) -> float:
    """Median absolute deviation. Returns 0.0 on empty / NaN-only input."""
    s = s.dropna()
    if s.empty:
        return 0.0
    return float((s - s.median()).abs().median())


def extra_features(
    events: pd.DataFrame, components: pd.DataFrame,
) -> np.ndarray:
    """Compute the 7 extra columns aligned with ``events`` row order."""
    if "Attack ID" not in events.columns:
        raise ValueError("events must contain 'Attack ID'")
    if "Attack ID" not in components.columns:
        raise ValueError("components must contain 'Attack ID'")

    aid = events["Attack ID"].to_numpy()

    # n_components — count rows per Attack ID; events with no
    # components in the table get 0.
    counts = (
        components.groupby("Attack ID", sort=False).size()
        .reindex(aid, fill_value=0)
        .to_numpy(dtype=np.float64)
    )
    log1p_counts = np.log1p(counts)
    is_single = (counts == 1).astype(np.float64)

    # Packet-speed median + MAD per event. Use log1p to match the
    # log-compression of pps_mean/pps_max in features_v2.
    pps_grp = components.groupby("Attack ID", sort=False)["Packet speed"]
    pps_median_raw = (
        pps_grp.median().reindex(aid, fill_value=0.0).to_numpy(dtype=np.float64)
    )
    # MAD via apply — slower than agg but robust on small groups.
    pps_mad_raw = (
        pps_grp.apply(_mad).reindex(aid, fill_value=0.0)
        .to_numpy(dtype=np.float64)
    )
    pps_median_log = np.log1p(np.clip(pps_median_raw, 0, None)).copy()
    pps_mad_log = np.log1p(np.clip(pps_mad_raw, 0, None)).copy()

    # Packet length median + MAD, excluding zero-length artefacts (same
    # rule as features_v2._component_aggregates).
    plen = components["Avg packet len"].astype(np.float64)
    plen_nz_components = components.assign(
        _plen_nz=plen.where(plen > 0)
    ).dropna(subset=["_plen_nz"])
    plen_grp = plen_nz_components.groupby("Attack ID", sort=False)["_plen_nz"]
    plen_median = np.array(
        plen_grp.median().reindex(aid, fill_value=0.0).to_numpy(dtype=np.float64),
        copy=True,
    )
    plen_mad = np.array(
        plen_grp.apply(_mad).reindex(aid, fill_value=0.0).to_numpy(dtype=np.float64),
        copy=True,
    )

    # Fallback for events with no components: use event-level Packet
    # speed / Avg packet len if available, so single-sample events have
    # a non-zero median (zero medians falsely signal "no traffic").
    if "Packet speed" in events.columns:
        ev_pps = events["Packet speed"].to_numpy(dtype=np.float64)
        zero = pps_median_raw == 0
        pps_median_log[zero] = np.log1p(np.clip(ev_pps[zero], 0, None))
    if "Avg packet len" in events.columns:
        ev_plen = events["Avg packet len"].to_numpy(dtype=np.float64)
        zero = plen_median == 0
        plen_median[zero] = np.clip(ev_plen[zero], 0, None)

    return np.column_stack([
        counts, log1p_counts, is_single,
        pps_median_log, pps_mad_log,
        plen_median, plen_mad,
    ])
