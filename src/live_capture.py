"""Live network traffic generation and capture using hping3.

Generates real attack traffic with hping3, measures packet statistics
via /proc/net/dev interface counters, and produces event DataFrames
compatible with the synthetic TrafficGenerator.

Requirements:
    - hping3 (apt-get install hping3)
    - Root / CAP_NET_RAW + CAP_NET_ADMIN
    - Linux (/proc/net/dev interface counters)

Usage:
    from src.live_capture import LiveTrafficGenerator, LIVE_SCENARIOS

    gen = LiveTrafficGenerator()
    # Blocking — runs for scenario.duration_seconds of real time
    gen.generate_stream_live(LIVE_SCENARIOS["live_syn_flood"])
    events = gen.pop_events()
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TrafficType = Literal["Normal traffic", "Suspicious traffic", "DDoS attack"]


def hping3_available() -> bool:
    """Check whether hping3 is installed and accessible."""
    return shutil.which("hping3") is not None


# ---------------------------------------------------------------------------
# Attack specifications
# ---------------------------------------------------------------------------

@dataclass
class HpingAttack:
    """Specification for an hping3-based attack pattern."""

    name: str
    traffic_type: TrafficType
    attack_code: str
    hping_args: list[str]
    port: int
    estimated_src_ips: int
    description: str


_ATTACK_LIST: list[HpingAttack] = [
    # ---- DDoS attacks ----
    HpingAttack(
        name="syn_flood",
        traffic_type="DDoS attack",
        attack_code="SYN Attack",
        hping_args=["-S", "--flood", "-p", "{port}"],
        port=80,
        estimated_src_ips=1,
        description="TCP SYN flood",
    ),
    HpingAttack(
        name="syn_flood_spoof",
        traffic_type="DDoS attack",
        attack_code="SYN Attack",
        hping_args=["-S", "--flood", "-p", "{port}", "--rand-source"],
        port=80,
        estimated_src_ips=50,
        description="TCP SYN flood with spoofed source IPs",
    ),
    HpingAttack(
        name="udp_flood",
        traffic_type="DDoS attack",
        attack_code="Generic UDP",
        hping_args=["--udp", "--flood", "-p", "{port}"],
        port=53,
        estimated_src_ips=1,
        description="UDP flood targeting DNS port",
    ),
    HpingAttack(
        name="icmp_flood",
        traffic_type="DDoS attack",
        attack_code="ICMP",
        hping_args=["--icmp", "--flood"],
        port=0,
        estimated_src_ips=1,
        description="ICMP echo flood",
    ),
    HpingAttack(
        name="ack_flood",
        traffic_type="DDoS attack",
        attack_code="TCP Anomaly, ACK Attack",
        hping_args=["-A", "--flood", "-p", "{port}", "--rand-source"],
        port=443,
        estimated_src_ips=30,
        description="TCP ACK flood with spoofed sources",
    ),
    HpingAttack(
        name="xmas_flood",
        traffic_type="DDoS attack",
        attack_code="TCP Anomaly",
        hping_args=["-FPU", "--flood", "-p", "{port}"],
        port=80,
        estimated_src_ips=1,
        description="TCP Xmas tree flood (FIN+PSH+URG)",
    ),
    HpingAttack(
        name="fragment_flood",
        traffic_type="DDoS attack",
        attack_code="IPv4 fragmentation",
        hping_args=["-f", "--flood", "-p", "{port}", "-d", "1400"],
        port=80,
        estimated_src_ips=1,
        description="Fragmented IP packet flood",
    ),
    HpingAttack(
        name="dns_amp_sim",
        traffic_type="DDoS attack",
        attack_code="DNS",
        hping_args=["--udp", "--flood", "-p", "53", "-d", "512", "--rand-source"],
        port=53,
        estimated_src_ips=40,
        description="DNS amplification simulation (large UDP to port 53)",
    ),
    # ---- Normal traffic ----
    HpingAttack(
        name="normal_tcp",
        traffic_type="Normal traffic",
        attack_code="High volume traffic",
        hping_args=["-S", "-p", "{port}", "-i", "u10000"],
        port=443,
        estimated_src_ips=1,
        description="Normal TCP SYN at moderate rate (~100 pps)",
    ),
    HpingAttack(
        name="normal_icmp",
        traffic_type="Normal traffic",
        attack_code="High volume traffic",
        hping_args=["--icmp", "-i", "u50000"],
        port=0,
        estimated_src_ips=1,
        description="Normal ICMP ping traffic (~20 pps)",
    ),
    # ---- Suspicious traffic ----
    HpingAttack(
        name="suspicious_rst",
        traffic_type="Suspicious traffic",
        attack_code="Suspicious traffic",
        hping_args=["-R", "-p", "{port}", "-i", "u5000"],
        port=443,
        estimated_src_ips=1,
        description="Suspicious RST scan traffic (~200 pps)",
    ),
]

HPING_ATTACKS: dict[str, HpingAttack] = {a.name: a for a in _ATTACK_LIST}


# ---------------------------------------------------------------------------
# Live scenario definitions
# ---------------------------------------------------------------------------

@dataclass
class LivePhase:
    """One phase of a live traffic generation schedule."""

    start_offset_s: float
    duration_s: float
    attack_name: str  # key into HPING_ATTACKS


@dataclass
class LiveScenario:
    """Full schedule for a live traffic generation run."""

    phases: list[LivePhase]
    duration_seconds: int


LIVE_SCENARIOS: dict[str, LiveScenario] = {
    "live_syn_flood": LiveScenario(
        duration_seconds=90,
        phases=[
            LivePhase(0.0, 10.0, "normal_tcp"),
            LivePhase(10.0, 10.0, "syn_flood"),
            LivePhase(20.0, 10.0, "normal_tcp"),
            LivePhase(30.0, 10.0, "syn_flood_spoof"),
            LivePhase(40.0, 10.0, "normal_tcp"),
            LivePhase(50.0, 15.0, "syn_flood"),
            LivePhase(65.0, 10.0, "normal_tcp"),
            LivePhase(75.0, 15.0, "syn_flood_spoof"),
        ],
    ),
    "live_multi_vector": LiveScenario(
        duration_seconds=120,
        phases=[
            LivePhase(0.0, 10.0, "normal_tcp"),
            LivePhase(10.0, 8.0, "syn_flood"),
            LivePhase(18.0, 7.0, "normal_tcp"),
            LivePhase(25.0, 8.0, "udp_flood"),
            LivePhase(33.0, 7.0, "normal_tcp"),
            LivePhase(40.0, 8.0, "icmp_flood"),
            LivePhase(48.0, 7.0, "normal_tcp"),
            LivePhase(55.0, 8.0, "ack_flood"),
            LivePhase(63.0, 7.0, "normal_tcp"),
            LivePhase(70.0, 8.0, "xmas_flood"),
            LivePhase(78.0, 7.0, "normal_tcp"),
            LivePhase(85.0, 10.0, "fragment_flood"),
            LivePhase(95.0, 10.0, "normal_icmp"),
            LivePhase(105.0, 15.0, "dns_amp_sim"),
        ],
    ),
    "live_dns_amp": LiveScenario(
        duration_seconds=100,
        phases=[
            LivePhase(0.0, 15.0, "normal_tcp"),
            LivePhase(15.0, 15.0, "dns_amp_sim"),
            LivePhase(30.0, 10.0, "normal_tcp"),
            LivePhase(40.0, 15.0, "dns_amp_sim"),
            LivePhase(55.0, 10.0, "normal_tcp"),
            LivePhase(65.0, 15.0, "dns_amp_sim"),
            LivePhase(80.0, 20.0, "normal_tcp"),
        ],
    ),
    "live_escalating": LiveScenario(
        duration_seconds=120,
        phases=[
            LivePhase(0.0, 20.0, "normal_tcp"),
            LivePhase(20.0, 10.0, "suspicious_rst"),
            LivePhase(30.0, 10.0, "normal_tcp"),
            LivePhase(40.0, 15.0, "syn_flood"),
            LivePhase(55.0, 5.0, "normal_tcp"),
            LivePhase(60.0, 20.0, "syn_flood_spoof"),
            LivePhase(80.0, 5.0, "normal_tcp"),
            LivePhase(85.0, 25.0, "dns_amp_sim"),
            LivePhase(110.0, 10.0, "normal_tcp"),
        ],
    ),
}


# ---------------------------------------------------------------------------
# Interface counter reading
# ---------------------------------------------------------------------------

def _read_interface_counters(interface: str) -> tuple[int, int]:
    """Read total (rx+tx) packets and bytes from /proc/net/dev."""
    try:
        text = Path("/proc/net/dev").read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(f"{interface}:"):
                parts = stripped.split()
                # Format: iface: rx_bytes rx_pkts ... tx_bytes tx_pkts ...
                rx_bytes, rx_pkts = int(parts[1]), int(parts[2])
                tx_bytes, tx_pkts = int(parts[9]), int(parts[10])
                return (rx_pkts + tx_pkts, rx_bytes + tx_bytes)
    except Exception as e:
        logger.warning("Could not read /proc/net/dev: %s", e)
    return (0, 0)


# ---------------------------------------------------------------------------
# Live traffic generator
# ---------------------------------------------------------------------------

class LiveTrafficGenerator:
    """Generate real network traffic with hping3 and measure packet statistics.

    Runs hping3 processes according to a schedule, reads interface counters
    from /proc/net/dev to measure actual packet rates, and produces events
    in the same DataFrame schema as the synthetic TrafficGenerator.

    Events accumulate in an internal buffer.  Call :meth:`pop_events` from
    another thread (e.g. the async StreamEngine) to drain them.
    """

    def __init__(
        self,
        target: str = "127.0.0.1",
        interface: str = "lo",
        seed: int = 42,
    ):
        self.target = target
        self.interface = interface
        self.rng = np.random.default_rng(seed)
        self._lock = threading.Lock()
        self._events: list[dict] = []
        self._stop_flag = threading.Event()

    # -- hping3 process management ------------------------------------------

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

    # -- event buffer -------------------------------------------------------

    def pop_events(self) -> list[dict]:
        """Thread-safe drain of all accumulated events."""
        with self._lock:
            events = self._events.copy()
            self._events.clear()
        return events

    def stop(self) -> None:
        """Signal the generation loop to stop."""
        self._stop_flag.set()

    # -- main generation loop -----------------------------------------------

    def generate_stream_live(
        self,
        scenario: LiveScenario,
        sample_interval_s: float = 0.5,
    ) -> None:
        """Run hping3 attacks in real time and populate the event buffer.

        This method **blocks** for ``scenario.duration_seconds``.
        Call :meth:`pop_events` from another thread to consume results.
        """
        self._events.clear()
        self._stop_flag.clear()

        t0 = time.monotonic()
        base_ts = pd.Timestamp.now()
        event_id = 0
        card_pool = [f"NIC_{i}" for i in range(4)]

        sorted_phases = sorted(scenario.phases, key=lambda p: p.start_offset_s)
        phase_idx = 0
        current_proc: subprocess.Popen | None = None
        current_spec: HpingAttack | None = None
        current_phase: LivePhase | None = None

        prev_pkts, prev_bytes = _read_interface_counters(self.interface)
        prev_sample_time = time.monotonic()

        try:
            while not self._stop_flag.is_set():
                elapsed = time.monotonic() - t0
                if elapsed >= scenario.duration_seconds:
                    break

                # -- advance to the correct phase --
                while phase_idx < len(sorted_phases):
                    next_phase = sorted_phases[phase_idx]
                    if elapsed >= next_phase.start_offset_s:
                        # Stop current hping3
                        if current_proc is not None:
                            self._stop_hping(current_proc)
                            current_proc = None

                        spec = HPING_ATTACKS[next_phase.attack_name]
                        current_spec = spec
                        current_phase = next_phase
                        current_proc = self._start_hping(spec)

                        # Reset counter baseline for the new phase
                        prev_pkts, prev_bytes = _read_interface_counters(self.interface)
                        prev_sample_time = time.monotonic()
                        phase_idx += 1
                    else:
                        break

                # -- check if current phase has ended --
                if current_phase is not None:
                    phase_end = current_phase.start_offset_s + current_phase.duration_s
                    if elapsed >= phase_end:
                        if current_proc is not None:
                            self._stop_hping(current_proc)
                            current_proc = None
                        current_spec = None
                        current_phase = None

                # -- sample counters --
                time.sleep(sample_interval_s)
                if self._stop_flag.is_set():
                    break

                now = time.monotonic()
                curr_pkts, curr_bytes = _read_interface_counters(self.interface)
                d_pkts = max(0, curr_pkts - prev_pkts)
                d_bytes = max(0, curr_bytes - prev_bytes)
                dt = max(0.01, now - prev_sample_time)
                elapsed = now - t0

                if d_pkts > 0 and current_spec is not None:
                    event_id += 1
                    pkt_speed = int(d_pkts / dt)
                    data_speed = int(d_bytes / dt)
                    avg_pkt_len = int(d_bytes / d_pkts) if d_pkts > 0 else 0

                    src_ip_count = current_spec.estimated_src_ips
                    if src_ip_count > 1:
                        src_ip_count = max(
                            1,
                            int(self.rng.normal(src_ip_count, src_ip_count * 0.2)),
                        )

                    detect_count = max(1, d_pkts // 1000)
                    event_ts = base_ts + pd.Timedelta(seconds=elapsed)

                    event = {
                        "Attack ID": event_id,
                        "Port number": current_spec.port,
                        "Card": str(self.rng.choice(card_pool)),
                        "Victim IP": self.target,
                        "Attack code": current_spec.attack_code,
                        "Detect count": detect_count,
                        "Packet speed": pkt_speed,
                        "Data speed": data_speed,
                        "Avg packet len": avg_pkt_len,
                        "Avg source IP count": src_ip_count,
                        "Start time": event_ts,
                        "End time": event_ts + pd.Timedelta(seconds=dt),
                        "Type": current_spec.traffic_type,
                        "_profile": f"live_{current_spec.name}",
                        "_arrival_offset_s": elapsed,
                        "_source": "hping3",
                    }

                    with self._lock:
                        self._events.append(event)

                prev_pkts, prev_bytes = curr_pkts, curr_bytes
                prev_sample_time = now

        finally:
            if current_proc is not None:
                self._stop_hping(current_proc)
            logger.info(
                "Live generation finished: %d events in %.1fs",
                event_id,
                time.monotonic() - t0,
            )
