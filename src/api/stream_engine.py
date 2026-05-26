"""Background simulation loop that publishes window results to Redis."""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import redis.asyncio as aioredis

from src.api.model_registry import ModelRegistry
from src.live_capture import LIVE_SCENARIOS, hping3_available
from src.live_capture_v2 import LiveTrafficGeneratorV2
from src.simulator import TYPE_TO_INT, INT_TO_TYPE
from src.synthetic import TrafficGenerator

logger = logging.getLogger(__name__)

REDIS_CHANNEL = "ddos:windows"

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

@dataclass
class ScenarioDef:
    id: str
    name: str
    description: str
    duration_seconds: int
    events_per_second: float
    ddos_ratio: float
    suspicious_ratio: float
    burst_attacks: bool
    burst_interval_s: int
    burst_duration_s: int
    burst_ddos_ratio: float
    novel_only: bool = False
    is_live: bool = False


SCENARIOS: dict[str, ScenarioDef] = {s.id: s for s in [
    # --- Synthetic scenarios (unchanged) ---
    ScenarioDef(
        id="normal_baseline",
        name="Normal Baseline",
        description="Pure normal traffic — tests false positive rate (0% DDoS)",
        duration_seconds=120,
        events_per_second=12.0,
        ddos_ratio=0.0,
        suspicious_ratio=0.05,
        burst_attacks=False,
        burst_interval_s=0,
        burst_duration_s=0,
        burst_ddos_ratio=0.0,
    ),
    ScenarioDef(
        id="low_rate_ddos",
        name="Low-Rate DDoS",
        description="Steady 5% DDoS background — tests sustained low-rate detection",
        duration_seconds=120,
        events_per_second=12.0,
        ddos_ratio=0.05,
        suspicious_ratio=0.10,
        burst_attacks=False,
        burst_interval_s=0,
        burst_duration_s=0,
        burst_ddos_ratio=0.0,
    ),
    ScenarioDef(
        id="syn_flood_burst",
        name="SYN Flood Burst",
        description="Periodic 50% DDoS bursts every 30s — tests burst detection + latency",
        duration_seconds=180,
        events_per_second=15.0,
        ddos_ratio=0.03,
        suspicious_ratio=0.08,
        burst_attacks=True,
        burst_interval_s=30,
        burst_duration_s=8,
        burst_ddos_ratio=0.50,
    ),
    ScenarioDef(
        id="dns_amplification",
        name="DNS Amplification",
        description="High-intensity DNS amplification attack with 70% DDoS bursts",
        duration_seconds=150,
        events_per_second=20.0,
        ddos_ratio=0.05,
        suspicious_ratio=0.08,
        burst_attacks=True,
        burst_interval_s=25,
        burst_duration_s=10,
        burst_ddos_ratio=0.70,
    ),
    ScenarioDef(
        id="novel_attacks",
        name="Novel/Zero-Day",
        description="HTTP/2 rapid-reset, QUIC flood, carpet bomb — tests generalization",
        duration_seconds=150,
        events_per_second=15.0,
        ddos_ratio=0.08,
        suspicious_ratio=0.08,
        burst_attacks=True,
        burst_interval_s=30,
        burst_duration_s=10,
        burst_ddos_ratio=0.60,
        novel_only=True,
    ),
    ScenarioDef(
        id="escalating",
        name="Escalating Attack",
        description="DDoS ratio ramps from 0% to 60% over the simulation duration",
        duration_seconds=180,
        events_per_second=15.0,
        ddos_ratio=0.0,
        suspicious_ratio=0.08,
        burst_attacks=False,
        burst_interval_s=0,
        burst_duration_s=0,
        burst_ddos_ratio=0.0,
    ),
    ScenarioDef(
        id="multi_vector",
        name="Multi-Vector Assault",
        description="Multiple simultaneous attack types with high-intensity bursts",
        duration_seconds=180,
        events_per_second=20.0,
        ddos_ratio=0.10,
        suspicious_ratio=0.12,
        burst_attacks=True,
        burst_interval_s=20,
        burst_duration_s=12,
        burst_ddos_ratio=0.65,
    ),
    # --- Live (hping3) scenarios ---
    ScenarioDef(
        id="live_syn_flood",
        name="Live: SYN Flood",
        description="Real TCP SYN flood via hping3 on loopback — alternates normal/attack phases",
        duration_seconds=90,
        events_per_second=0, ddos_ratio=0, suspicious_ratio=0,
        burst_attacks=False, burst_interval_s=0, burst_duration_s=0, burst_ddos_ratio=0,
        is_live=True,
    ),
    ScenarioDef(
        id="live_multi_vector",
        name="Live: Multi-Vector",
        description="Real SYN/UDP/ICMP/ACK/Xmas/frag/DNS floods via hping3 — cycles through attack types",
        duration_seconds=120,
        events_per_second=0, ddos_ratio=0, suspicious_ratio=0,
        burst_attacks=False, burst_interval_s=0, burst_duration_s=0, burst_ddos_ratio=0,
        is_live=True,
    ),
    ScenarioDef(
        id="live_dns_amp",
        name="Live: DNS Amplification",
        description="Real DNS amplification simulation via hping3 — large UDP payloads to port 53",
        duration_seconds=100,
        events_per_second=0, ddos_ratio=0, suspicious_ratio=0,
        burst_attacks=False, burst_interval_s=0, burst_duration_s=0, burst_ddos_ratio=0,
        is_live=True,
    ),
    ScenarioDef(
        id="live_escalating",
        name="Live: Escalating",
        description="Real traffic ramps from normal → suspicious → SYN → UDP → spoofed SYN → DNS amp",
        duration_seconds=120,
        events_per_second=0, ddos_ratio=0, suspicious_ratio=0,
        burst_attacks=False, burst_interval_s=0, burst_duration_s=0, burst_ddos_ratio=0,
        is_live=True,
    ),
]}


class StreamEngine:
    """Generates traffic windows in a background task, publishes results to Redis."""

    def __init__(self, registry: ModelRegistry, redis_url: str = "redis://redis:6379"):
        self._registry = registry
        self._redis_url = redis_url
        self._task: asyncio.Task | None = None
        self._running = False
        self._scenario_id: str | None = None

        # Live counters
        self.windows_processed = 0
        self.total_events = 0
        self.total_ddos_detected = 0
        self.total_ddos_actual = 0
        self.alert_count = 0
        self.false_alert_count = 0

    @property
    def running(self) -> bool:
        return self._running

    @property
    def scenario_id(self) -> str | None:
        return self._scenario_id

    def _reset_counters(self) -> None:
        self.windows_processed = 0
        self.total_events = 0
        self.total_ddos_detected = 0
        self.total_ddos_actual = 0
        self.alert_count = 0
        self.false_alert_count = 0

    async def start(self, scenario_id: str) -> None:
        if self._running:
            await self.stop()
        self._reset_counters()
        self._scenario_id = scenario_id
        self._running = True
        self._task = asyncio.create_task(self._run(scenario_id))

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    # -----------------------------------------------------------------------
    # Dispatch
    # -----------------------------------------------------------------------

    async def _run(self, scenario_id: str) -> None:
        scenario = SCENARIOS[scenario_id]
        redis = aioredis.from_url(self._redis_url)
        await redis.delete(REDIS_CHANNEL)

        try:
            if scenario.is_live:
                await self._run_live(scenario_id, scenario, redis)
            else:
                await self._run_synthetic(scenario_id, scenario, redis)
        except asyncio.CancelledError:
            pass
        finally:
            await redis.rpush(
                REDIS_CHANNEL,
                json.dumps({"done": True, "scenario_id": scenario_id}),
            )
            await redis.aclose()
            self._running = False

    # -----------------------------------------------------------------------
    # Synthetic mode (existing logic)
    # -----------------------------------------------------------------------

    async def _run_synthetic(
        self, scenario_id: str, scenario: ScenarioDef, redis: aioredis.Redis,
    ) -> None:
        gen = TrafficGenerator(seed=int(time.time()) % 10000)

        if scenario_id == "escalating":
            events = self._generate_escalating(gen, scenario)
        else:
            events = gen.generate_stream(
                duration_seconds=scenario.duration_seconds,
                events_per_second=scenario.events_per_second,
                ddos_ratio=scenario.ddos_ratio,
                suspicious_ratio=scenario.suspicious_ratio,
                burst_attacks=scenario.burst_attacks,
                burst_interval_s=scenario.burst_interval_s,
                burst_duration_s=scenario.burst_duration_s,
                burst_ddos_ratio=scenario.burst_ddos_ratio,
            )

        # v2 featurizer needs both events and components. Build the
        # component frame once up front; per-window slicing happens via
        # Attack ID inside _process_windows.
        components = gen.generate_components(events)

        await self._process_windows(events, components, scenario, redis, source="synthetic")

    def _generate_escalating(
        self, gen: TrafficGenerator, scenario: ScenarioDef,
    ) -> pd.DataFrame:
        n_chunks = 6
        chunk_dur = scenario.duration_seconds // n_chunks
        frames = []
        for i in range(n_chunks):
            ratio = (i / (n_chunks - 1)) * 0.60
            chunk = gen.generate_stream(
                duration_seconds=chunk_dur,
                events_per_second=scenario.events_per_second,
                ddos_ratio=ratio,
                suspicious_ratio=scenario.suspicious_ratio,
                burst_attacks=False,
                burst_interval_s=0,
                burst_duration_s=0,
                burst_ddos_ratio=0.0,
            )
            if frames:
                offset = pd.Timedelta(seconds=chunk_dur * i)
                chunk["Start time"] = chunk["Start time"] + offset
                chunk["End time"] = chunk["End time"] + offset
            frames.append(chunk)
        return (
            pd.concat(frames, ignore_index=True)
            .sort_values("Start time")
            .reset_index(drop=True)
        )

    async def _process_windows(
        self,
        events: pd.DataFrame,
        components: pd.DataFrame,
        scenario: ScenarioDef,
        redis: aioredis.Redis,
        source: str = "synthetic",
    ) -> None:
        t_start = events["Start time"].min()
        t_end = events["Start time"].max()
        window_delta = pd.Timedelta(seconds=5.0)

        current = t_start
        window_idx = 0

        while current < t_end and self._running:
            w_end = current + window_delta
            mask = (events["Start time"] >= current) & (events["Start time"] < w_end)
            window_events = events[mask]

            if len(window_events) == 0:
                current = w_end
                continue

            window_components = components[
                components["Attack ID"].isin(window_events["Attack ID"])
            ]

            await self._process_and_publish_window(
                window_events, window_components, window_idx, current, w_end,
                scenario.id, redis, source,
            )

            window_idx += 1
            current = w_end
            await asyncio.sleep(0.5)

    # -----------------------------------------------------------------------
    # Live (hping3) mode
    # -----------------------------------------------------------------------

    async def _run_live(
        self, scenario_id: str, scenario: ScenarioDef, redis: aioredis.Redis,
    ) -> None:
        live_scenario = LIVE_SCENARIOS[scenario_id]
        gen = LiveTrafficGeneratorV2()

        # The v2 generator yields one (events_df, components_df) tuple
        # per closed window. We push each tuple onto a thread-safe queue
        # and consume them from the asyncio loop.
        q: queue.Queue = queue.Queue(maxsize=8)
        stop_event = threading.Event()
        SENTINEL = object()

        def _produce() -> None:
            try:
                for ev_df, comp_df in gen.stream_scenario(live_scenario, stop_event):
                    q.put((ev_df, comp_df))
            except Exception:
                logger.exception("live capture failed")
            finally:
                q.put(SENTINEL)

        gen_thread = threading.Thread(target=_produce, daemon=True)
        gen_thread.start()

        window_idx = 0
        try:
            while self._running:
                try:
                    item = await asyncio.get_event_loop().run_in_executor(
                        None, q.get, True, 1.0,
                    )
                except queue.Empty:
                    if not gen_thread.is_alive():
                        break
                    continue
                if item is SENTINEL:
                    break

                ev_df, comp_df = item
                if len(ev_df) == 0:
                    continue
                window_start = ev_df["Start time"].iloc[0]
                window_end = ev_df["End time"].iloc[-1]
                await self._process_and_publish_window(
                    ev_df, comp_df, window_idx,
                    window_start, window_end,
                    scenario_id, redis, source="hping3",
                )
                window_idx += 1
        finally:
            stop_event.set()
            gen_thread.join(timeout=5)

    # -----------------------------------------------------------------------
    # Shared per-window processing
    # -----------------------------------------------------------------------

    async def _process_and_publish_window(
        self,
        window_events: pd.DataFrame,
        window_components: pd.DataFrame,
        window_idx: int,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
        scenario_id: str,
        redis: aioredis.Redis,
        source: str = "synthetic",
    ) -> None:
        """Run model prediction on one window and publish results to Redis."""
        model_name = self._registry.current_name or "unknown"
        if self._registry.current_model is None:
            logger.error("No model selected")
            return

        ddos_label = TYPE_TO_INT["DDoS attack"]
        normal_label = TYPE_TO_INT["Normal traffic"]
        suspicious_label = TYPE_TO_INT["Suspicious traffic"]

        # v2+cluster featurization: one row per event in the window.
        X_aug = self._registry.featurize_window(window_events, window_components)

        y_true = window_events["Type"].astype(str).map(TYPE_TO_INT).values

        t0 = time.perf_counter()
        y_pred = self._registry.predict(X_aug)
        latency_ms = (time.perf_counter() - t0) * 1000

        n_pred_ddos = int((y_pred == ddos_label).sum())
        n_pred_normal = int((y_pred == normal_label).sum())
        n_pred_suspicious = int((y_pred == suspicious_label).sum())
        n_actual_ddos = int((y_true == ddos_label).sum())
        n_actual_normal = int((y_true == normal_label).sum())
        n_actual_suspicious = int((y_true == suspicious_label).sum())
        accuracy = float(np.mean(y_pred == y_true))
        alert = n_pred_ddos >= 3

        # Confusion matrix (3x3)
        cm = np.zeros((3, 3), dtype=int)
        for true_label, pred_label in zip(y_true, y_pred):
            cm[int(true_label)][int(pred_label)] += 1

        # Traffic stats
        pkt_speeds = window_events["Packet speed"].values
        # Live-v2 capture omits Data speed (bps) from the events schema;
        # estimate from packet speed * avg packet length when missing.
        if "Data speed" in window_events.columns:
            data_speeds = window_events["Data speed"].values
        else:
            data_speeds = (
                window_events["Packet speed"].values
                * window_events.get("Avg packet len", pd.Series(0, index=window_events.index)).values
            )
        avg_pkt_rate = float(np.mean(pkt_speeds)) if len(pkt_speeds) else 0.0
        avg_data_rate = float(np.mean(data_speeds)) if len(data_speeds) else 0.0
        peak_pkt_rate = float(np.max(pkt_speeds)) if len(pkt_speeds) else 0.0
        port_series = window_events["Port number"]
        top_ports = port_series.value_counts().head(3)
        top_ports_list = [
            {"port": int(p), "count": int(c)} for p, c in top_ports.items()
        ]

        # Event samples (last N for live feed)
        sample_size = min(8, len(window_events))
        sample_rows = window_events.tail(sample_size)
        sample_preds = y_pred[-sample_size:]
        event_samples = []
        for (_, row), pred_label in zip(sample_rows.iterrows(), sample_preds):
            actual_type = row["Type"]
            actual_int = (
                TYPE_TO_INT.get(actual_type, 0)
                if isinstance(actual_type, str)
                else int(actual_type.map(TYPE_TO_INT))
                if hasattr(actual_type, "map")
                else TYPE_TO_INT.get(str(actual_type), 0)
            )
            pkt_speed = int(row["Packet speed"])
            if "Data speed" in window_events.columns:
                data_speed = int(row["Data speed"])
            else:
                data_speed = int(pkt_speed * row.get("Avg packet len", 0))
            event_samples.append({
                "src_ip_count": int(row.get("Avg source IP count", 0)),
                "pkt_speed": pkt_speed,
                "data_speed": data_speed,
                "port": int(row["Port number"]),
                "actual": INT_TO_TYPE[actual_int],
                "predicted": INT_TO_TYPE[int(pred_label)],
                "correct": bool(actual_int == int(pred_label)),
            })

        # Update counters
        self.windows_processed += 1
        self.total_events += len(window_events)
        self.total_ddos_detected += n_pred_ddos
        self.total_ddos_actual += n_actual_ddos
        if alert:
            self.alert_count += 1
            if n_actual_ddos == 0:
                self.false_alert_count += 1

        event = {
            "window_index": window_idx,
            "window_start": str(window_start),
            "window_end": str(window_end),
            "n_events": len(window_events),
            "n_predicted_normal": n_pred_normal,
            "n_predicted_suspicious": n_pred_suspicious,
            "n_predicted_ddos": n_pred_ddos,
            "n_actual_normal": n_actual_normal,
            "n_actual_suspicious": n_actual_suspicious,
            "n_actual_ddos": n_actual_ddos,
            "accuracy": round(accuracy, 4),
            "detection_latency_ms": round(latency_ms, 2),
            "alert_raised": alert,
            "model_name": model_name,
            "scenario_id": scenario_id,
            "source": source,
            "avg_pkt_rate": round(avg_pkt_rate, 1),
            "avg_data_rate": round(avg_data_rate, 1),
            "peak_pkt_rate": round(peak_pkt_rate, 1),
            "top_ports": top_ports_list,
            "confusion": cm.tolist(),
            "event_samples": event_samples,
        }

        await redis.rpush(REDIS_CHANNEL, json.dumps(event))
