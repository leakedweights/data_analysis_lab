"""Continuous host-network sampler producing v2-shaped windows.

Watches ``/proc/net/dev`` and (optionally) sniffs an AF_PACKET raw socket
for source-IP counts, the same way ``live_capture_v2.LiveTrafficGeneratorV2``
does — but without the hping3 scenario driver. Yields one
``(events_df, components_df)`` tuple per closed wall-clock window in the
schema ``features_v2.featurize`` expects, indefinitely until a stop
event is set.

This is the building block for the real-host monitor: feed its output
straight into ``ModelRegistry.featurize_window`` and you get live
predictions on whatever traffic is on the interface.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from src.live_capture_v2 import (
    CapturedSample,
    CapturedWindow,
    SourceIPSniffer,
    _read_counters,
)

logger = logging.getLogger(__name__)


def auto_detect_interface() -> str:
    """Pick the first non-loopback interface from ``/proc/net/dev``.

    Falls back to ``lo`` if nothing else is available (e.g. a minimal
    container) so the monitor still runs and can be exercised against
    loopback traffic.
    """
    from pathlib import Path

    candidates: list[str] = []
    for line in Path("/proc/net/dev").read_text().splitlines():
        stripped = line.strip()
        if ":" not in stripped:
            continue
        name = stripped.split(":", 1)[0].strip()
        if name in ("face", ""):
            continue
        candidates.append(name)

    for name in candidates:
        if name != "lo":
            return name
    return "lo"


@dataclass
class WindowReading:
    """Light wrapper around the raw window for the monitor's view layer.
    The featurizer only needs the DataFrames, but the TUI also wants the
    summary stats without re-deriving them from the frames.
    """
    attack_id: int
    start_wall: pd.Timestamp
    end_wall: pd.Timestamp
    n_samples: int
    pps_mean: float
    pps_max: float
    bytes_per_pkt_mean: float
    distinct_src_ips: int
    events_df: pd.DataFrame
    components_df: pd.DataFrame


class HostSampler:
    """Continuously samples a real network interface and yields
    (events_df, components_df) per closed window.

    Parameters
    ----------
    interface:
        NIC name as it appears in ``/proc/net/dev`` (e.g. ``lo``, ``wlp3s0``).
    window_s:
        Wall-clock window length. Each yielded tuple covers one window.
    sample_s:
        Sub-second sample cadence. Each sample becomes one row in the
        components frame.
    enable_src_ip_sniffer:
        Whether to spin up the AF_PACKET source-IP sniffer thread. If
        the process lacks ``CAP_NET_RAW`` the sniffer logs a warning and
        reports zero distinct IPs; the monitor still functions.
    """

    def __init__(
        self,
        interface: str,
        window_s: float = 5.0,
        sample_s: float = 0.1,
        enable_src_ip_sniffer: bool = True,
        port: int = 443,
    ):
        self.interface = interface
        self.window_s = float(window_s)
        self.sample_s = float(sample_s)
        self.enable_src_ip_sniffer = enable_src_ip_sniffer
        # Port number feeds the v2 ``is_port_*`` binary features. Real
        # host traffic spans many ports so any single choice is a lie;
        # 443 puts ``is_https_port=1`` which is the closest match for
        # general benign egress (HTTPS dominates a laptop's traffic).
        # 0 → ``is_port_zero=1`` which in training is heavily attack-
        # correlated, so the default 443 keeps the model honest. Override
        # via ``--port`` if monitoring a specific service.
        self.port = int(port)

    def stream_windows(
        self, stop_event: threading.Event,
    ) -> Iterator[WindowReading]:
        """Yield one :class:`WindowReading` per closed window until
        ``stop_event`` is set."""
        sniffer = (
            SourceIPSniffer(self.interface, window_s=self.window_s)
            if self.enable_src_ip_sniffer
            else None
        )
        if sniffer is not None:
            sniffer.start()

        attack_id = 0
        base_wall = pd.Timestamp.now()
        t0 = time.monotonic()

        try:
            while not stop_event.is_set():
                win_start_mono = time.monotonic()
                win_end_mono = win_start_mono + self.window_s

                window = CapturedWindow(
                    attack_id=attack_id + 1,
                    label="Normal traffic",  # placeholder; predict overrides
                    port=self.port,
                    start_t=win_start_mono,
                    end_t=win_end_mono,
                    spec_name="host_monitor",
                )
                attack_id += 1

                prev_pkts, prev_bytes = _read_counters(self.interface)
                prev_t = time.monotonic()

                while not stop_event.is_set():
                    now = time.monotonic()
                    if now >= win_end_mono:
                        break
                    sleep_s = min(self.sample_s, win_end_mono - now)
                    if sleep_s > 0:
                        stop_event.wait(sleep_s)
                    now = time.monotonic()
                    cur_pkts, cur_bytes = _read_counters(self.interface)
                    d_pkts = max(0, cur_pkts - prev_pkts)
                    d_bytes = max(0, cur_bytes - prev_bytes)
                    dt = max(0.001, now - prev_t)
                    if d_pkts > 0:
                        window.samples.append(CapturedSample(
                            t=now,
                            pps=d_pkts / dt,
                            bps=d_bytes / dt,
                            avg_pkt_len=d_bytes / d_pkts,
                        ))
                    prev_pkts, prev_bytes = cur_pkts, cur_bytes
                    prev_t = now

                if stop_event.is_set():
                    break

                if sniffer is not None:
                    window.distinct_src_ips = sniffer.distinct_in_range(
                        win_start_mono, win_end_mono,
                    )

                yield self._to_reading(window, base_wall, t0)
        finally:
            if sniffer is not None:
                sniffer.stop()

    def _to_reading(
        self,
        w: CapturedWindow,
        base_wall: pd.Timestamp,
        t0: float,
    ) -> WindowReading:
        # Same conversion logic as LiveTrafficGeneratorV2._to_dataframes,
        # specialized for a single window.
        wall_start = base_wall + pd.Timedelta(seconds=w.start_t - t0)
        wall_end = base_wall + pd.Timedelta(seconds=w.end_t - t0)

        if w.samples:
            pps_arr = np.array([s.pps for s in w.samples], dtype=np.float64)
            len_arr = np.array([s.avg_pkt_len for s in w.samples], dtype=np.float64)
        else:
            pps_arr = np.array([0.0])
            len_arr = np.array([0.0])

        events_df = pd.DataFrame([{
            "Attack ID": w.attack_id,
            "Port number": int(w.port),
            "Packet speed": int(pps_arr.mean()),
            "Avg packet len": int(len_arr.mean()),
            "Avg source IP count": int(w.distinct_src_ips),
            "Start time": wall_start,
            "End time": wall_end,
            # No Type column — featurize is called with label_col=None.
            "_source": "host_monitor",
        }])

        if w.samples:
            comp_rows = [
                {
                    "Attack ID": w.attack_id,
                    "Detect count": i + 1,
                    "Port number": int(w.port),
                    "Packet speed": int(s.pps),
                    "Avg packet len": int(s.avg_pkt_len),
                    "Source IP count": int(w.distinct_src_ips),
                    "Time": base_wall + pd.Timedelta(seconds=s.t - t0),
                }
                for i, s in enumerate(w.samples)
            ]
        else:
            comp_rows = [{
                "Attack ID": w.attack_id,
                "Detect count": 1,
                "Port number": int(w.port),
                "Packet speed": 0,
                "Avg packet len": 0,
                "Source IP count": int(w.distinct_src_ips),
                "Time": wall_start,
            }]
        components_df = pd.DataFrame(comp_rows)

        return WindowReading(
            attack_id=w.attack_id,
            start_wall=wall_start,
            end_wall=wall_end,
            n_samples=len(w.samples),
            pps_mean=float(pps_arr.mean()),
            pps_max=float(pps_arr.max()),
            bytes_per_pkt_mean=float(len_arr.mean()),
            distinct_src_ips=int(w.distinct_src_ips),
            events_df=events_df,
            components_df=components_df,
        )
