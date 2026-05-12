"""Synthetic DDoS network traffic generator.

Generates realistic network traffic data matching the SCLDDoS2024 schema.
Profiles are derived directly from real training data log-normal statistics,
covering all major attack codes plus novel/unseen attack variants.

Usage:
    from src.synthetic import TrafficGenerator
    gen = TrafficGenerator(seed=42)
    events = gen.generate_events(n=10000, ddos_ratio=0.05, suspicious_ratio=0.10)
    components = gen.generate_components(events)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

TrafficType = Literal["Normal traffic", "Suspicious traffic", "DDoS attack"]


@dataclass
class AttackProfile:
    """Statistical profile for one attack variant.

    All numeric fields are (log_mean, log_std) — values are sampled as
    exp(Normal(log_mean, log_std)).  Derived from real SCLDDoS2024 training
    data unless marked as *novel*.
    """
    name: str
    traffic_type: TrafficType
    attack_codes: list[str]
    weight: float  # relative sampling probability within its traffic_type
    ports: list[int]
    port_weights: list[float] | None  # if None, uniform
    packet_speed: tuple[float, float]
    data_speed: tuple[float, float]
    avg_packet_len: tuple[float, float]
    source_ip_count: tuple[float, float]
    detect_count: tuple[float, float]
    duration_seconds: tuple[float, float]


# ---------------------------------------------------------------------------
# Port pools (from real data)
# ---------------------------------------------------------------------------
NORMAL_PORTS = [0, 443, 4500, 80, 7777, 60645, 51821, 1200]
NORMAL_PORT_W = [83187, 21710, 19814, 6982, 4601, 3689, 2366, 1927]

SUSPICIOUS_PORTS = [0, 34863, 443, 4500, 51413, 1200, 20409, 27041]
SUSPICIOUS_PORT_W = [4147, 686, 396, 188, 120, 67, 33, 25]

DDOS_PORTS = [0, 443, 53, 10052, 80, 22, 22003, 51413]
DDOS_PORT_W = [619, 504, 189, 180, 177, 106, 104, 67]

# ---------------------------------------------------------------------------
# PROFILES — organised by traffic type, then by attack variant
# All (log_mean, log_std) tuples from real per-attack-code statistics.
# ---------------------------------------------------------------------------

PROFILES: list[AttackProfile] = [
    # =======================================================================
    # NORMAL TRAFFIC — single profile matching real "High volume traffic"
    # =======================================================================
    AttackProfile(
        name="normal_high_volume",
        traffic_type="Normal traffic",
        attack_codes=["High volume traffic"],
        weight=1.0,
        ports=NORMAL_PORTS,
        port_weights=NORMAL_PORT_W,
        packet_speed=(11.13, 0.27),
        data_speed=(4.37, 0.42),
        avg_packet_len=(7.09, 0.29),
        source_ip_count=(0.46, 0.78),
        detect_count=(0.52, 0.88),
        duration_seconds=(5.0, 1.5),
    ),

    # =======================================================================
    # SUSPICIOUS TRAFFIC — from real data
    # =======================================================================
    AttackProfile(
        name="suspicious_generic",
        traffic_type="Suspicious traffic",
        attack_codes=["Suspicious traffic"],
        weight=6609,
        ports=SUSPICIOUS_PORTS,
        port_weights=SUSPICIOUS_PORT_W,
        packet_speed=(11.15, 0.23),
        data_speed=(4.03, 0.97),
        avg_packet_len=(6.75, 0.88),
        source_ip_count=(1.74, 0.86),
        detect_count=(0.24, 0.56),
        duration_seconds=(5.0, 1.5),
    ),
    AttackProfile(
        name="suspicious_icmp",
        traffic_type="Suspicious traffic",
        attack_codes=["ICMP"],
        weight=3546,
        ports=[0, 443, 80],
        port_weights=[70, 20, 10],
        packet_speed=(10.93, 0.11),
        data_speed=(1.30, 0.17),
        avg_packet_len=(4.0, 0.5),  # median 0, many near-zero
        source_ip_count=(4.08, 0.24),
        detect_count=(0.0, 0.05),
        duration_seconds=(5.0, 1.0),
    ),
    AttackProfile(
        name="suspicious_high_volume",
        traffic_type="Suspicious traffic",
        attack_codes=["Suspicious traffic, High volume traffic"],
        weight=3225,
        ports=SUSPICIOUS_PORTS,
        port_weights=SUSPICIOUS_PORT_W,
        packet_speed=(11.21, 0.19),
        data_speed=(4.44, 0.36),
        avg_packet_len=(7.10, 0.36),
        source_ip_count=(1.73, 0.70),
        detect_count=(1.41, 0.91),
        duration_seconds=(5.5, 1.5),
    ),

    # =======================================================================
    # DDoS ATTACKS — real profiles from training data
    # =======================================================================

    # --- SYN Flood (largest DDoS category) ---
    AttackProfile(
        name="ddos_syn_attack",
        traffic_type="DDoS attack",
        attack_codes=["SYN Attack", "High volume traffic, SYN Attack"],
        weight=798,
        ports=DDOS_PORTS,
        port_weights=DDOS_PORT_W,
        packet_speed=(11.13, 0.51),
        data_speed=(3.48, 1.49),
        avg_packet_len=(4.0, 1.0),  # median ~0, small SYN packets
        source_ip_count=(0.67, 1.61),
        detect_count=(0.29, 0.77),
        duration_seconds=(5.5, 2.0),
    ),

    # --- DNS amplification ---
    AttackProfile(
        name="ddos_dns_amp",
        traffic_type="DDoS attack",
        attack_codes=["DNS, High volume traffic", "DNS"],
        weight=459,
        ports=[53, 0, 443, 80],
        port_weights=[50, 30, 10, 10],
        packet_speed=(11.34, 0.41),
        data_speed=(4.61, 0.55),
        avg_packet_len=(7.15, 0.39),  # large DNS responses
        source_ip_count=(0.42, 0.95),
        detect_count=(4.07, 3.11),
        duration_seconds=(6.0, 2.0),
    ),

    # --- NTP amplification ---
    AttackProfile(
        name="ddos_ntp_amp",
        traffic_type="DDoS attack",
        attack_codes=["NTP", "NTP, High volume traffic"],
        weight=192,
        ports=[123, 0, 443],
        port_weights=[60, 30, 10],
        packet_speed=(10.45, 0.64),
        data_speed=(3.42, 0.67),
        avg_packet_len=(6.64, 0.57),
        source_ip_count=(1.17, 1.58),
        detect_count=(0.90, 1.26),
        duration_seconds=(5.5, 1.8),
    ),

    # --- Generic UDP flood ---
    AttackProfile(
        name="ddos_generic_udp",
        traffic_type="DDoS attack",
        attack_codes=["Generic UDP", "Generic UDP, High volume traffic"],
        weight=228,
        ports=[0, 80, 443, 53],
        port_weights=[50, 20, 20, 10],
        packet_speed=(11.03, 0.22),
        data_speed=(1.86, 0.70),
        avg_packet_len=(4.82, 0.49),  # small UDP packets
        source_ip_count=(4.34, 1.85),  # heavily spoofed
        detect_count=(1.85, 2.09),
        duration_seconds=(6.0, 2.0),
    ),

    # --- Generic UDP + Suspicious (multi-vector) ---
    AttackProfile(
        name="ddos_udp_suspicious",
        traffic_type="DDoS attack",
        attack_codes=["Generic UDP, Suspicious traffic",
                      "Generic UDP, Suspicious traffic, High volume traffic"],
        weight=121,
        ports=[0, 443, 80],
        port_weights=[50, 30, 20],
        packet_speed=(11.18, 0.15),
        data_speed=(1.76, 0.39),
        avg_packet_len=(4.59, 0.26),
        source_ip_count=(2.75, 1.88),
        detect_count=(4.11, 1.63),
        duration_seconds=(6.5, 2.0),
    ),

    # --- CLDAP amplification ---
    AttackProfile(
        name="ddos_cldap",
        traffic_type="DDoS attack",
        attack_codes=["CLDAP, High volume traffic", "CLDAP"],
        weight=71,
        ports=[389, 0, 443],
        port_weights=[60, 30, 10],
        packet_speed=(10.93, 0.15),
        data_speed=(4.11, 0.13),
        avg_packet_len=(7.06, 0.09),  # large LDAP responses
        source_ip_count=(0.37, 0.39),
        detect_count=(2.87, 2.32),
        duration_seconds=(5.0, 1.5),
    ),

    # --- WSD (Web Services Discovery) ---
    AttackProfile(
        name="ddos_wsd",
        traffic_type="DDoS attack",
        attack_codes=["WSD", "High volume traffic, WSD"],
        weight=87,
        ports=[3702, 0, 443],
        port_weights=[50, 30, 20],
        packet_speed=(10.46, 0.24),
        data_speed=(3.69, 0.38),
        avg_packet_len=(7.03, 0.47),
        source_ip_count=(0.72, 1.55),
        detect_count=(0.33, 0.59),
        duration_seconds=(5.0, 1.5),
    ),

    # --- CoAP amplification ---
    AttackProfile(
        name="ddos_coap",
        traffic_type="DDoS attack",
        attack_codes=["CoAP", "CoAP, High volume traffic"],
        weight=44,
        ports=[5683, 0, 443],
        port_weights=[60, 30, 10],
        packet_speed=(10.38, 0.35),
        data_speed=(3.13, 0.79),
        avg_packet_len=(6.68, 0.61),
        source_ip_count=(0.44, 0.75),
        detect_count=(0.11, 0.32),
        duration_seconds=(5.0, 1.5),
    ),

    # --- Multi-vector: Suspicious+DNS+High volume (high intensity) ---
    AttackProfile(
        name="ddos_multi_dns_high",
        traffic_type="DDoS attack",
        attack_codes=["Suspicious traffic, DNS, High volume traffic"],
        weight=30,
        ports=[53, 0, 443, 80],
        port_weights=[40, 30, 20, 10],
        packet_speed=(11.93, 0.83),  # very high pps
        data_speed=(4.97, 0.97),     # very high bps
        avg_packet_len=(7.03, 0.52),
        source_ip_count=(5.77, 1.73),  # massive spoofing
        detect_count=(4.67, 0.97),
        duration_seconds=(7.0, 2.0),
    ),

    # --- Sentinel / low-and-slow ---
    AttackProfile(
        name="ddos_sentinel",
        traffic_type="DDoS attack",
        attack_codes=["Sentinel"],
        weight=23,
        ports=[0, 443, 80, 22],
        port_weights=[40, 30, 20, 10],
        packet_speed=(9.82, 0.32),  # low pps
        data_speed=(0.54, 0.36),    # very low bps
        avg_packet_len=(4.92, 0.29),
        source_ip_count=(0.14, 0.32),
        detect_count=(0.0, 0.01),
        duration_seconds=(6.0, 2.0),
    ),

    # --- TCP Anomaly ---
    AttackProfile(
        name="ddos_tcp_anomaly",
        traffic_type="DDoS attack",
        attack_codes=["TCP Anomaly", "TCP Anomaly, ACK Attack",
                      "TCP Anomaly, SYN Attack"],
        weight=12,
        ports=[0, 443, 80, 22],
        port_weights=[40, 30, 20, 10],
        packet_speed=(10.5, 0.5),
        data_speed=(2.5, 1.0),
        avg_packet_len=(5.0, 0.8),
        source_ip_count=(1.0, 1.5),
        detect_count=(0.3, 0.5),
        duration_seconds=(5.0, 1.5),
    ),

    # --- IPv4 fragmentation ---
    AttackProfile(
        name="ddos_ipv4_frag",
        traffic_type="DDoS attack",
        attack_codes=["IPv4 fragmentation",
                      "IPv4 fragmentation, High volume traffic"],
        weight=37,
        ports=[0, 443, 80],
        port_weights=[50, 30, 20],
        packet_speed=(10.8, 0.4),
        data_speed=(3.5, 0.8),
        avg_packet_len=(6.5, 0.6),
        source_ip_count=(1.0, 1.2),
        detect_count=(1.5, 1.5),
        duration_seconds=(5.5, 1.8),
    ),

    # --- Memcached amplification ---
    AttackProfile(
        name="ddos_memcached",
        traffic_type="DDoS attack",
        attack_codes=["Memcached, High volume traffic", "Memcached"],
        weight=9,
        ports=[11211, 0, 443],
        port_weights=[60, 30, 10],
        packet_speed=(11.5, 0.6),
        data_speed=(5.0, 0.8),   # extremely high amplification
        avg_packet_len=(7.2, 0.3),  # large payloads
        source_ip_count=(0.5, 0.8),
        detect_count=(2.0, 1.5),
        duration_seconds=(4.5, 1.5),
    ),

    # --- SSDP amplification ---
    AttackProfile(
        name="ddos_ssdp",
        traffic_type="DDoS attack",
        attack_codes=["SSDP", "SSDP, High volume traffic"],
        weight=7,
        ports=[1900, 0, 443],
        port_weights=[60, 30, 10],
        packet_speed=(10.6, 0.4),
        data_speed=(3.5, 0.6),
        avg_packet_len=(6.8, 0.4),
        source_ip_count=(0.8, 1.0),
        detect_count=(0.5, 0.8),
        duration_seconds=(5.0, 1.5),
    ),

    # --- RPC flood ---
    AttackProfile(
        name="ddos_rpc",
        traffic_type="DDoS attack",
        attack_codes=["RPC", "RPC, High volume traffic"],
        weight=14,
        ports=[135, 0, 443],
        port_weights=[50, 30, 20],
        packet_speed=(10.7, 0.4),
        data_speed=(3.3, 0.6),
        avg_packet_len=(6.2, 0.5),
        source_ip_count=(0.6, 1.0),
        detect_count=(0.8, 1.0),
        duration_seconds=(5.0, 1.5),
    ),

    # =======================================================================
    # NOVEL / UNSEEN ATTACK VARIANTS — not in training data
    # These test generalization: models must detect DDoS patterns they
    # haven't been trained on.
    # =======================================================================

    # --- HTTP/2 rapid-reset (CVE-2023-44487 style) ---
    AttackProfile(
        name="novel_http2_rapid_reset",
        traffic_type="DDoS attack",
        attack_codes=["HTTP2_RAPID_RESET"],
        weight=40,
        ports=[443, 8443, 80, 8080],
        port_weights=[50, 20, 20, 10],
        packet_speed=(11.8, 0.4),   # very high pps from RST frames
        data_speed=(3.0, 0.5),      # low data (tiny RST frames)
        avg_packet_len=(4.2, 0.3),  # ~66 bytes per RST
        source_ip_count=(2.0, 1.0),
        detect_count=(3.0, 1.5),
        duration_seconds=(5.0, 1.0),
    ),

    # --- DNS water-torture (random subdomain) ---
    AttackProfile(
        name="novel_dns_water_torture",
        traffic_type="DDoS attack",
        attack_codes=["DNS_WATER_TORTURE"],
        weight=30,
        ports=[53, 0],
        port_weights=[80, 20],
        packet_speed=(10.8, 0.5),
        data_speed=(3.8, 0.6),
        avg_packet_len=(5.5, 0.4),  # ~250 byte queries
        source_ip_count=(3.5, 1.5),  # distributed botnet
        detect_count=(2.0, 1.8),
        duration_seconds=(7.0, 1.5),  # sustained
    ),

    # --- QUIC flood (UDP 443) ---
    AttackProfile(
        name="novel_quic_flood",
        traffic_type="DDoS attack",
        attack_codes=["QUIC_FLOOD"],
        weight=25,
        ports=[443, 8443],
        port_weights=[80, 20],
        packet_speed=(11.5, 0.5),
        data_speed=(4.5, 0.7),
        avg_packet_len=(7.0, 0.3),   # max-size QUIC initial packets
        source_ip_count=(3.0, 1.8),
        detect_count=(1.5, 1.2),
        duration_seconds=(5.5, 1.5),
    ),

    # --- Carpet-bombing (spread across many victim IPs) ---
    AttackProfile(
        name="novel_carpet_bomb",
        traffic_type="DDoS attack",
        attack_codes=["CARPET_BOMB"],
        weight=20,
        ports=[0, 80, 443, 53],
        port_weights=[40, 25, 25, 10],
        packet_speed=(11.0, 0.3),
        data_speed=(4.0, 0.5),
        avg_packet_len=(6.5, 0.8),
        source_ip_count=(5.0, 1.5),  # massive source IPs
        detect_count=(3.5, 1.5),
        duration_seconds=(6.5, 2.0),
    ),

    # --- Slow HTTP POST (application-layer) ---
    AttackProfile(
        name="novel_slow_post",
        traffic_type="DDoS attack",
        attack_codes=["SLOW_POST"],
        weight=15,
        ports=[80, 443, 8080],
        port_weights=[40, 40, 20],
        packet_speed=(8.5, 0.5),    # low pps — that's the point
        data_speed=(2.0, 0.8),      # trickle of data
        avg_packet_len=(6.0, 0.5),
        source_ip_count=(3.0, 1.0),
        detect_count=(1.0, 0.8),
        duration_seconds=(8.0, 1.5),  # long-lived
    ),

    # --- GRE flood (tunneled) ---
    AttackProfile(
        name="novel_gre_flood",
        traffic_type="DDoS attack",
        attack_codes=["GRE_FLOOD"],
        weight=15,
        ports=[0, 443, 80],
        port_weights=[60, 25, 15],
        packet_speed=(11.4, 0.5),
        data_speed=(4.2, 0.6),
        avg_packet_len=(6.8, 0.4),
        source_ip_count=(2.5, 1.5),
        detect_count=(1.5, 1.2),
        duration_seconds=(5.5, 1.8),
    ),

    # --- CHARGEN amplification ---
    AttackProfile(
        name="novel_chargen",
        traffic_type="DDoS attack",
        attack_codes=["CHARGEN_AMP"],
        weight=10,
        ports=[19, 0],
        port_weights=[70, 30],
        packet_speed=(10.3, 0.5),
        data_speed=(3.8, 0.7),
        avg_packet_len=(6.5, 0.5),
        source_ip_count=(0.8, 1.0),
        detect_count=(1.0, 1.0),
        duration_seconds=(5.0, 1.5),
    ),

    # --- SNMP amplification ---
    AttackProfile(
        name="novel_snmp_amp",
        traffic_type="DDoS attack",
        attack_codes=["SNMP_AMP"],
        weight=10,
        ports=[161, 0],
        port_weights=[70, 30],
        packet_speed=(10.5, 0.4),
        data_speed=(4.0, 0.5),
        avg_packet_len=(6.8, 0.3),
        source_ip_count=(0.5, 0.8),
        detect_count=(1.2, 1.0),
        duration_seconds=(5.0, 1.5),
    ),

    # --- Pulse-wave (on-off high-volume bursts) ---
    AttackProfile(
        name="novel_pulse_wave",
        traffic_type="DDoS attack",
        attack_codes=["PULSE_WAVE"],
        weight=20,
        ports=[0, 443, 80, 53],
        port_weights=[35, 30, 25, 10],
        packet_speed=(12.0, 0.8),   # extreme peaks
        data_speed=(5.0, 1.0),
        avg_packet_len=(6.0, 1.0),
        source_ip_count=(4.0, 2.0),
        detect_count=(3.0, 2.0),
        duration_seconds=(4.0, 1.5),  # short but intense
    ),

    # --- Botnet HTTP GET flood ---
    AttackProfile(
        name="novel_http_get_flood",
        traffic_type="DDoS attack",
        attack_codes=["HTTP_GET_FLOOD"],
        weight=25,
        ports=[80, 443, 8080, 8443],
        port_weights=[35, 35, 15, 15],
        packet_speed=(11.0, 0.4),
        data_speed=(4.3, 0.5),
        avg_packet_len=(6.8, 0.3),   # full HTTP requests
        source_ip_count=(4.5, 1.0),  # large botnet
        detect_count=(2.0, 1.5),
        duration_seconds=(6.0, 2.0),
    ),
]


# ---------------------------------------------------------------------------
# Index profiles by type for fast lookup
# ---------------------------------------------------------------------------
def _profiles_by_type() -> dict[TrafficType, list[AttackProfile]]:
    result: dict[TrafficType, list[AttackProfile]] = {
        "Normal traffic": [],
        "Suspicious traffic": [],
        "DDoS attack": [],
    }
    for p in PROFILES:
        result[p.traffic_type].append(p)
    return result


_PROFILES_BY_TYPE = _profiles_by_type()


def _pick_profile(rng: np.random.Generator, traffic_type: TrafficType) -> AttackProfile:
    """Weighted random selection of an attack profile for the given type."""
    profiles = _PROFILES_BY_TYPE[traffic_type]
    weights = np.array([p.weight for p in profiles], dtype=np.float64)
    weights /= weights.sum()
    idx = rng.choice(len(profiles), p=weights)
    return profiles[idx]


def _pick_port(rng: np.random.Generator, prof: AttackProfile) -> int:
    if prof.port_weights:
        w = np.array(prof.port_weights, dtype=np.float64)
        w /= w.sum()
        return int(rng.choice(prof.ports, p=w))
    return int(rng.choice(prof.ports))


class TrafficGenerator:
    """Generate synthetic network traffic events and components."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self._caps = self._load_caps()

    def _load_caps(self) -> dict | None:
        path = DATA_DIR / "cleaning_config.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def _generate_ips(self, n: int, prefix: str = "10.0") -> np.ndarray:
        return np.array([
            f"{prefix}.{self.rng.integers(0, 256)}.{self.rng.integers(1, 255)}"
            for _ in range(n)
        ])

    def _sample_event(self, prof: AttackProfile, event_id: int,
                      t0: pd.Timestamp, span_seconds: int,
                      victim_pool: np.ndarray, card_pool: list[str]) -> dict:
        """Sample a single event from a profile."""
        offset_s = self.rng.integers(0, max(span_seconds, 1))
        start = t0 + pd.Timedelta(seconds=int(offset_s))
        dur = max(1, int(np.exp(self.rng.normal(*prof.duration_seconds))))
        end = start + pd.Timedelta(seconds=dur)

        pkt_speed = max(0, int(np.exp(self.rng.normal(*prof.packet_speed))))
        data_speed = max(0, int(np.exp(self.rng.normal(*prof.data_speed))))
        avg_pkt_len = int(np.clip(
            np.exp(self.rng.normal(*prof.avg_packet_len)), 0, 1518
        ))
        src_ip_cnt = max(1, int(np.exp(self.rng.normal(*prof.source_ip_count))))
        det_cnt = max(1, int(np.exp(self.rng.normal(*prof.detect_count))))

        return {
            "Attack ID": event_id,
            "Port number": _pick_port(self.rng, prof),
            "Card": self.rng.choice(card_pool),
            "Victim IP": self.rng.choice(victim_pool),
            "Attack code": self.rng.choice(prof.attack_codes),
            "Detect count": det_cnt,
            "Packet speed": pkt_speed,
            "Data speed": data_speed,
            "Avg packet len": avg_pkt_len,
            "Avg source IP count": src_ip_cnt,
            "Start time": start,
            "End time": end,
            "Type": prof.traffic_type,
            "_profile": prof.name,
        }

    def generate_events(
        self,
        n: int = 10000,
        ddos_ratio: float = 0.05,
        suspicious_ratio: float = 0.10,
        start_time: str = "2024-01-01",
        span_days: int = 30,
    ) -> pd.DataFrame:
        """Generate synthetic event-level traffic data.

        Args:
            n: Total number of events.
            ddos_ratio: Fraction of DDoS attack events.
            suspicious_ratio: Fraction of suspicious traffic events.
            start_time: Start of the time window.
            span_days: Duration of the time window in days.
        """
        n_ddos = int(n * ddos_ratio)
        n_suspicious = int(n * suspicious_ratio)
        n_normal = n - n_ddos - n_suspicious

        types: list[TrafficType] = (
            ["Normal traffic"] * n_normal
            + ["Suspicious traffic"] * n_suspicious
            + ["DDoS attack"] * n_ddos
        )
        self.rng.shuffle(types)  # type: ignore[arg-type]

        t0 = pd.Timestamp(start_time)
        span_seconds = span_days * 86400
        victim_pool = self._generate_ips(50)
        card_pool = [f"NIC_{i}" for i in range(4)]

        records = []
        for i, typ in enumerate(types):
            prof = _pick_profile(self.rng, typ)
            rec = self._sample_event(prof, i + 1, t0, span_seconds,
                                     victim_pool, card_pool)
            records.append(rec)

        df = pd.DataFrame(records)
        for col in ["Card", "Victim IP", "Attack code", "Type"]:
            df[col] = df[col].astype("category")
        return df

    def generate_components(self, events: pd.DataFrame) -> pd.DataFrame:
        """Generate component-level data from events."""
        rows = []
        for _, ev in events.iterrows():
            n_comp = int(ev["Detect count"])
            base_time = ev["Start time"]
            dur_s = (ev["End time"] - ev["Start time"]).total_seconds()

            for j in range(n_comp):
                pkt_speed = max(0, int(ev["Packet speed"] * self.rng.lognormal(0, 0.3)))
                data_speed = max(0, int(ev["Data speed"] * self.rng.lognormal(0, 0.3)))
                avg_pkt_len = int(np.clip(
                    ev["Avg packet len"] * self.rng.lognormal(0, 0.1), 0, 1518
                ))
                src_ip_cnt = max(1, int(
                    ev["Avg source IP count"] * self.rng.lognormal(0, 0.5)
                ))

                if self._caps:
                    cc = self._caps["component_caps"]
                    pkt_speed = min(pkt_speed, int(cc["Packet speed"]))
                    data_speed = min(data_speed, int(cc["Data speed"]))
                    avg_pkt_len = min(avg_pkt_len, int(cc["Avg packet len"]))
                    src_ip_cnt = min(src_ip_cnt, int(cc["Source IP count"]))

                comp_offset = self.rng.uniform(0, max(dur_s, 1))
                comp_time = base_time + pd.Timedelta(seconds=comp_offset)

                rows.append({
                    "Attack ID": ev["Attack ID"],
                    "Detect count": j + 1,
                    "Card": ev["Card"],
                    "Victim IP": ev["Victim IP"],
                    "Port number": ev["Port number"],
                    "Attack code": ev["Attack code"],
                    "Significant flag": self.rng.integers(0, 2),
                    "Packet speed": pkt_speed,
                    "Data speed": data_speed,
                    "Avg packet len": avg_pkt_len,
                    "Source IP count": src_ip_cnt,
                    "Time": comp_time,
                })

        df = pd.DataFrame(rows)
        for col in ["Card", "Victim IP", "Attack code"]:
            df[col] = df[col].astype("category")
        return df

    def generate_stream(
        self,
        duration_seconds: int = 300,
        events_per_second: float = 10.0,
        ddos_ratio: float = 0.05,
        suspicious_ratio: float = 0.10,
        burst_attacks: bool = False,
        burst_interval_s: int = 60,
        burst_duration_s: int = 10,
        burst_ddos_ratio: float = 0.50,
    ) -> pd.DataFrame:
        """Generate a time-ordered event stream for simulation.

        Optionally includes periodic DDoS bursts to test detection latency.
        """
        n_total = int(duration_seconds * events_per_second)
        t0 = pd.Timestamp.now()

        inter_arrivals = self.rng.exponential(1.0 / events_per_second, n_total)
        arrival_offsets = np.cumsum(inter_arrivals)

        victim_pool = self._generate_ips(30)
        card_pool = [f"NIC_{i}" for i in range(4)]

        records = []
        for i, offset in enumerate(arrival_offsets):
            # Determine traffic type
            if burst_attacks and (offset % burst_interval_s) < burst_duration_s:
                r = self.rng.random()
                if r < burst_ddos_ratio:
                    typ: TrafficType = "DDoS attack"
                elif r < burst_ddos_ratio + suspicious_ratio:
                    typ = "Suspicious traffic"
                else:
                    typ = "Normal traffic"
            else:
                r = self.rng.random()
                if r < ddos_ratio:
                    typ = "DDoS attack"
                elif r < ddos_ratio + suspicious_ratio:
                    typ = "Suspicious traffic"
                else:
                    typ = "Normal traffic"

            prof = _pick_profile(self.rng, typ)
            start = t0 + pd.Timedelta(seconds=float(offset))
            dur = max(1, int(np.exp(self.rng.normal(*prof.duration_seconds))))

            pkt_speed = max(0, int(np.exp(self.rng.normal(*prof.packet_speed))))
            data_speed = max(0, int(np.exp(self.rng.normal(*prof.data_speed))))
            avg_pkt_len = int(np.clip(
                np.exp(self.rng.normal(*prof.avg_packet_len)), 0, 1518
            ))
            src_ip_cnt = max(1, int(np.exp(self.rng.normal(*prof.source_ip_count))))
            det_cnt = max(1, int(np.exp(self.rng.normal(*prof.detect_count))))

            records.append({
                "Attack ID": i + 1,
                "Port number": _pick_port(self.rng, prof),
                "Card": self.rng.choice(card_pool),
                "Victim IP": self.rng.choice(victim_pool),
                "Attack code": self.rng.choice(prof.attack_codes),
                "Detect count": det_cnt,
                "Packet speed": pkt_speed,
                "Data speed": data_speed,
                "Avg packet len": avg_pkt_len,
                "Avg source IP count": src_ip_cnt,
                "Start time": start,
                "End time": start + pd.Timedelta(seconds=dur),
                "Type": typ,
                "_arrival_offset_s": float(offset),
                "_profile": prof.name,
            })

        df = pd.DataFrame(records)
        for col in ["Card", "Victim IP", "Attack code", "Type"]:
            df[col] = df[col].astype("category")
        return df.sort_values("Start time").reset_index(drop=True)
