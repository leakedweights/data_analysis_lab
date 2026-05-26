"""v2 live capture — windowed, train/inference-aligned, no leaky features.

Differences from ``live_capture.py`` (v1):

* **Window-based granularity.** Each "event" is a fixed ``window_s``
  (default 5 s) wall-clock window. Within the window we sample
  ``/proc/net/dev`` every ``sample_s`` (default 0.1 s); each sample
  becomes one "component". This matches the multi-component event
  structure the v2 featurizer expects from training data.

* **Real source IP measurement.** A second thread sniffs an
  ``AF_PACKET`` raw socket on the same interface and parses the IPv4
  source address out of each frame. The number of distinct source IPs
  observed during a window is recorded in the components table — no
  more hard-coded ``estimated_src_ips`` from the attack spec, which in
  v1 was effectively a label leak.

* **No invented Detect count.** v1 set ``Detect count = d_pkts // 1000``
  which had nothing to do with the training-data semantic ("number of
  detection components in the event") and was wildly out of range for
  flood traffic. v2 doesn't use that feature at all — the model uses
  rate statistics aggregated from the within-window samples instead.

* **Symmetric featurization.** The captured events + components flow
  straight through ``features_v2.featurize`` — exactly the same code
  path the training script uses.

Requires:
    - hping3 to run scenarios (only available inside the docker eval target)
    - Linux raw socket access (CAP_NET_RAW) — the docker compose grants this
    - root or capability when running outside docker
"""

from __future__ import annotations

import logging
import shutil
import socket
import struct
import subprocess
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TrafficType = Literal["Normal traffic", "Suspicious traffic", "DDoS attack"]


def hping3_available() -> bool:
    return shutil.which("hping3") is not None


# ---------------------------------------------------------------------------
# Attack specs (reuse v1's list verbatim — same hping3 invocations)
# ---------------------------------------------------------------------------

from src.live_capture import HPING_ATTACKS, LIVE_SCENARIOS, HpingAttack, LiveScenario  # noqa: E402,F401


# ---------------------------------------------------------------------------
# Source IP sniffer (raw AF_PACKET, stdlib only)
# ---------------------------------------------------------------------------

ETH_P_ALL = 0x0003
ETH_P_IP = 0x0800


class SourceIPSniffer:
    """Background thread that counts distinct IPv4 source addresses per
    rolling window. Uses an AF_PACKET raw socket on the given interface
    and parses the IP source field out of the frame header — no scapy,
    no extra dependencies, just stdlib.
    """

    def __init__(self, interface: str, window_s: float = 5.0):
        self.interface = interface
        self.window_s = window_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._buckets: list[set[int]] = []
        self._bucket_starts: list[float] = []
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

    def start(self) -> None:
        try:
            sock = socket.socket(
                socket.AF_PACKET, socket.SOCK_RAW, socket.htons(ETH_P_ALL)
            )
            sock.bind((self.interface, ETH_P_ALL))
            sock.settimeout(0.2)
        except (OSError, AttributeError) as exc:
            # AF_PACKET is Linux-only; missing CAP_NET_RAW gives EPERM.
            self._error = exc
            logger.warning("Source IP sniffer disabled: %s", exc)
            return
        self._sock = sock
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass

    def _run(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                data = self._sock.recv(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(data) < 34:  # eth(14) + min ip(20)
                continue
            ethertype = struct.unpack("!H", data[12:14])[0]
            if ethertype != ETH_P_IP:
                continue
            # IPv4 src is 4 bytes at offset 26 (eth header 14 + ip src offset 12)
            src_int = struct.unpack("!I", data[26:30])[0]
            now = time.monotonic()
            with self._lock:
                if not self._buckets or now - self._bucket_starts[-1] >= self.window_s:
                    self._buckets.append(set())
                    self._bucket_starts.append(now)
                self._buckets[-1].add(src_int)

    def distinct_in_range(self, t_start: float, t_end: float) -> int:
        """Number of distinct source IPs observed in [t_start, t_end)."""
        if self._error is not None:
            return 0
        with self._lock:
            distinct: set[int] = set()
            for bs, bucket in zip(self._bucket_starts, self._buckets):
                if bs >= t_end:
                    break
                if bs + self.window_s <= t_start:
                    continue
                distinct |= bucket
            return len(distinct)


# ---------------------------------------------------------------------------
# /proc/net/dev counter reader
# ---------------------------------------------------------------------------

def _read_counters(interface: str) -> tuple[int, int]:
    text = Path("/proc/net/dev").read_text()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{interface}:"):
            parts = stripped.split()
            rx_bytes, rx_pkts = int(parts[1]), int(parts[2])
            tx_bytes, tx_pkts = int(parts[9]), int(parts[10])
            return (rx_pkts + tx_pkts, rx_bytes + tx_bytes)
    return (0, 0)


# ---------------------------------------------------------------------------
# Captured-window data structures
# ---------------------------------------------------------------------------

@dataclass
class CapturedSample:
    """One sub-second sample inside a window — becomes a 'component'."""
    t: float
    pps: float
    bps: float
    avg_pkt_len: float


@dataclass
class CapturedWindow:
    """One window — becomes an 'event' for the v2 featurizer."""
    attack_id: int
    label: TrafficType
    port: int
    start_t: float
    end_t: float
    samples: list[CapturedSample] = field(default_factory=list)
    distinct_src_ips: int = 0
    spec_name: str = ""


# ---------------------------------------------------------------------------
# Live generator
# ---------------------------------------------------------------------------

class LiveTrafficGeneratorV2:
    """Run an hping3 scenario and emit (events, components) DataFrames
    in the schema the v2 featurizer expects.

    A wall-clock window is opened for each scenario phase. Sub-second
    samples within the window are collected as components. Source IPs
    are measured live from a raw socket, not invented from the spec.
    """

    def __init__(
        self,
        target: str = "127.0.0.1",
        interface: str = "lo",
        window_s: float = 5.0,
        sample_s: float = 0.1,
    ):
        self.target = target
        self.interface = interface
        self.window_s = float(window_s)
        self.sample_s = float(sample_s)

    # -- hping3 ------------------------------------------------------------

    def _build_cmd(self, spec: HpingAttack) -> list[str]:
        cmd = ["hping3"]
        for arg in spec.hping_args:
            cmd.append(arg.format(port=spec.port))
        cmd.extend(["-q", self.target])
        return cmd

    def _start_hping(self, spec: HpingAttack) -> subprocess.Popen:
        cmd = self._build_cmd(spec)
        logger.info("hping3 start: %s", " ".join(cmd))
        return subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    @staticmethod
    def _stop_hping(proc: subprocess.Popen) -> None:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=1)

    # -- main capture loop -------------------------------------------------

    def run_scenario(self, scenario: LiveScenario) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run a scenario and return (events_df, components_df)."""
        sniffer = SourceIPSniffer(self.interface, window_s=self.window_s)
        sniffer.start()

        sorted_phases = sorted(scenario.phases, key=lambda p: p.start_offset_s)
        windows: list[CapturedWindow] = []
        attack_id = 0
        base_wall = pd.Timestamp.now()
        t0 = time.monotonic()

        try:
            for phase in sorted_phases:
                spec = HPING_ATTACKS[phase.attack_name]
                # Wait until phase start
                while time.monotonic() - t0 < phase.start_offset_s:
                    time.sleep(0.01)

                proc = self._start_hping(spec)
                phase_end = phase.start_offset_s + phase.duration_s

                # Open windows of length window_s within the phase.
                window_idx = 0
                while True:
                    win_start_offset = phase.start_offset_s + window_idx * self.window_s
                    win_end_offset = min(win_start_offset + self.window_s, phase_end)
                    if win_start_offset >= phase_end:
                        break
                    if win_end_offset - win_start_offset < self.sample_s * 2:
                        break  # window too small to be meaningful

                    win_start_mono = t0 + win_start_offset
                    win_end_mono = t0 + win_end_offset

                    attack_id += 1
                    window = CapturedWindow(
                        attack_id=attack_id,
                        label=spec.traffic_type,
                        port=spec.port,
                        start_t=win_start_mono,
                        end_t=win_end_mono,
                        spec_name=spec.name,
                    )

                    prev_pkts, prev_bytes = _read_counters(self.interface)
                    prev_t = time.monotonic()

                    # Sample sub-window components.
                    while True:
                        now = time.monotonic()
                        if now >= win_end_mono:
                            break
                        sleep_s = min(self.sample_s, win_end_mono - now)
                        time.sleep(max(0.0, sleep_s))
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

                    window.distinct_src_ips = sniffer.distinct_in_range(
                        win_start_mono, win_end_mono,
                    )
                    windows.append(window)
                    window_idx += 1

                self._stop_hping(proc)
        finally:
            sniffer.stop()

        return self._to_dataframes(windows, base_wall, t0)

    def stream_scenario(
        self, scenario: LiveScenario, stop_event: threading.Event | None = None,
    ) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """Yield ``(events_df, components_df)`` per closed window as the
        scenario runs. Same window construction as :meth:`run_scenario`;
        the only difference is the per-window flush so the dashboard can
        render results progressively instead of waiting for the whole
        scenario.

        Pass ``stop_event`` to abort mid-scenario from another thread —
        each window is checked before the next phase starts and after
        each sample.
        """
        sniffer = SourceIPSniffer(self.interface, window_s=self.window_s)
        sniffer.start()

        sorted_phases = sorted(scenario.phases, key=lambda p: p.start_offset_s)
        attack_id = 0
        base_wall = pd.Timestamp.now()
        t0 = time.monotonic()

        def _aborted() -> bool:
            return stop_event is not None and stop_event.is_set()

        try:
            for phase in sorted_phases:
                if _aborted():
                    break
                spec = HPING_ATTACKS[phase.attack_name]
                while time.monotonic() - t0 < phase.start_offset_s:
                    if _aborted():
                        return
                    time.sleep(0.01)

                proc = self._start_hping(spec)
                phase_end = phase.start_offset_s + phase.duration_s

                window_idx = 0
                while True:
                    if _aborted():
                        self._stop_hping(proc)
                        return
                    win_start_offset = phase.start_offset_s + window_idx * self.window_s
                    win_end_offset = min(win_start_offset + self.window_s, phase_end)
                    if win_start_offset >= phase_end:
                        break
                    if win_end_offset - win_start_offset < self.sample_s * 2:
                        break

                    win_start_mono = t0 + win_start_offset
                    win_end_mono = t0 + win_end_offset

                    attack_id += 1
                    window = CapturedWindow(
                        attack_id=attack_id,
                        label=spec.traffic_type,
                        port=spec.port,
                        start_t=win_start_mono,
                        end_t=win_end_mono,
                        spec_name=spec.name,
                    )

                    prev_pkts, prev_bytes = _read_counters(self.interface)
                    prev_t = time.monotonic()

                    while True:
                        if _aborted():
                            self._stop_hping(proc)
                            return
                        now = time.monotonic()
                        if now >= win_end_mono:
                            break
                        sleep_s = min(self.sample_s, win_end_mono - now)
                        time.sleep(max(0.0, sleep_s))
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

                    window.distinct_src_ips = sniffer.distinct_in_range(
                        win_start_mono, win_end_mono,
                    )
                    yield self._to_dataframes([window], base_wall, t0)
                    window_idx += 1

                self._stop_hping(proc)
        finally:
            sniffer.stop()

    # -- conversion to v2-shaped DataFrames -------------------------------

    def _to_dataframes(
        self,
        windows: list[CapturedWindow],
        base_wall: pd.Timestamp,
        t0: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        event_rows: list[dict] = []
        comp_rows: list[dict] = []

        for w in windows:
            n = max(1, len(w.samples))
            pps_arr = np.array([s.pps for s in w.samples], dtype=np.float64) if w.samples else np.array([0.0])
            len_arr = np.array([s.avg_pkt_len for s in w.samples], dtype=np.float64) if w.samples else np.array([0.0])

            wall_start = base_wall + pd.Timedelta(seconds=w.start_t - t0)
            wall_end = base_wall + pd.Timedelta(seconds=w.end_t - t0)

            event_rows.append({
                "Attack ID": w.attack_id,
                "Port number": int(w.port),
                "Packet speed": int(pps_arr.mean()),
                "Avg packet len": int(len_arr.mean()) if len_arr.size else 0,
                "Avg source IP count": int(w.distinct_src_ips),
                "Start time": wall_start,
                "End time": wall_end,
                "Type": w.label,
                "_spec": w.spec_name,
                "_source": "hping3_v2",
            })

            for i, s in enumerate(w.samples or [CapturedSample(w.start_t, 0.0, 0.0, 0.0)]):
                comp_rows.append({
                    "Attack ID": w.attack_id,
                    "Detect count": i + 1,
                    "Port number": int(w.port),
                    "Packet speed": int(s.pps),
                    "Avg packet len": int(s.avg_pkt_len),
                    "Source IP count": int(w.distinct_src_ips),
                    "Time": base_wall + pd.Timedelta(seconds=s.t - t0),
                })

        events_df = pd.DataFrame(event_rows)
        components_df = pd.DataFrame(comp_rows)
        for col in ("Type",):
            if col in events_df.columns:
                events_df[col] = events_df[col].astype("category")
        return events_df, components_df
