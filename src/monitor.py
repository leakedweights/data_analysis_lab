"""Live host-network DDoS monitor — v2+cluster model on a real NIC.

Usage:
    uv run python -m src.monitor                # auto-detect interface
    uv run python -m src.monitor --interface lo
    uv run python -m src.monitor --model "Gradient Boosting" --alert-threshold 2

Ctrl+C to quit. First run trains the v2+cluster pipeline from ``data/``
and caches the result under ``--model-dir`` (default ``./models``);
subsequent runs are warm.

Source-IP counting needs CAP_NET_RAW (run with ``sudo`` or grant the
capability); without it the sniffer reports zero distinct IPs and the
``src_ip_*`` features fall back to zero. The monitor still runs.
"""

from __future__ import annotations

import argparse
import logging
import queue
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.api.model_registry import ModelRegistry
from src.host_sampler import HostSampler, WindowReading, auto_detect_interface
from src.simulator import INT_TO_TYPE, TYPE_TO_INT

logger = logging.getLogger(__name__)

# Class display config
_CLASS_COLOR = {
    "Normal traffic": "green",
    "Suspicious traffic": "yellow",
    "DDoS attack": "bold red",
}
_CLASS_GLYPH = {
    "Normal traffic": "N",
    "Suspicious traffic": "S",
    "DDoS attack": "D",
}
_TYPE_ORDER = ["Normal traffic", "Suspicious traffic", "DDoS attack"]

_SPARK_CHARS = "▁▂▃▄▅▆▇█"


# ---------------------------------------------------------------------------
# Per-window prediction record
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    reading: WindowReading
    predicted_class: str  # one of _TYPE_ORDER
    probs: dict[str, float]
    latency_ms: float


# ---------------------------------------------------------------------------
# Sparkline + glyph helpers
# ---------------------------------------------------------------------------

def _sparkline(values: list[float], width: int = 40) -> str:
    if not values:
        return ""
    series = list(values)[-width:]
    lo, hi = min(series), max(series)
    if hi <= lo:
        return _SPARK_CHARS[0] * len(series)
    span = hi - lo
    n = len(_SPARK_CHARS)
    return "".join(_SPARK_CHARS[min(n - 1, int((v - lo) / span * (n - 1)))] for v in series)


def _glyph_for(cls: str) -> Text:
    return Text(_CLASS_GLYPH.get(cls, "?"), style=_CLASS_COLOR.get(cls, ""))


# ---------------------------------------------------------------------------
# View
# ---------------------------------------------------------------------------

class MonitorView:
    """Renders the rolling state into a rich Layout. Stateful so the
    timeline and alert log accumulate across windows."""

    HISTORY = 60

    def __init__(self, interface: str, model_name: str, window_s: float,
                 alert_prob: float):
        self.interface = interface
        self.model_name = model_name
        self.window_s = window_s
        self.alert_prob = alert_prob

        self.history: deque[Prediction] = deque(maxlen=self.HISTORY)
        self.pps_history: deque[float] = deque(maxlen=self.HISTORY)
        self.alerts: deque[tuple[pd.Timestamp, str]] = deque(maxlen=10)
        self.windows_seen = 0
        self.ddos_windows = 0
        self.started_at = time.monotonic()

    def update(self, pred: Prediction) -> None:
        self.history.append(pred)
        self.pps_history.append(pred.reading.pps_mean)
        self.windows_seen += 1
        ddos_prob = pred.probs.get("DDoS attack", 0.0)
        if pred.predicted_class == "DDoS attack":
            self.ddos_windows += 1
        if ddos_prob >= self.alert_prob:
            msg = (
                f"DDoS detected — pps={pred.reading.pps_mean:.0f}, "
                f"src_ips={pred.reading.distinct_src_ips}, "
                f"p={ddos_prob:.2f}"
            )
            self.alerts.append((pred.reading.end_wall, msg))

    def render(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(self._header(), name="header", size=3),
            Layout(self._main(), name="main"),
            Layout(self._alerts(), name="alerts", size=8),
        )
        return layout

    # -- panel builders ----------------------------------------------------

    def _header(self) -> Panel:
        uptime = time.monotonic() - self.started_at
        body = Text.assemble(
            ("interface=", "dim"), (f"{self.interface}  ", "bold"),
            ("model=", "dim"), (f"{self.model_name}  ", "bold"),
            ("window=", "dim"), (f"{self.window_s:.1f}s  ", "bold"),
            ("uptime=", "dim"), (f"{int(uptime)}s  ", "bold"),
            ("windows=", "dim"), (f"{self.windows_seen}  "),
            ("ddos=", "dim"), (f"{self.ddos_windows}",
                               "bold red" if self.ddos_windows else ""),
        )
        return Panel(body, title="DDoS host monitor", border_style="blue")

    def _main(self) -> Panel:
        pps_spark = _sparkline(list(self.pps_history), width=60)
        spark_line = Text.assemble(
            ("pps  ", "dim"), (pps_spark, "cyan"),
        )

        timeline = Text("last ")
        timeline.append(f"{len(self.history)}", style="dim")
        timeline.append(" windows: ")
        for p in self.history:
            timeline.append_text(_glyph_for(p.predicted_class))
            timeline.append(" ")

        if self.history:
            current = self.history[-1]
            cur_class_text = Text(
                current.predicted_class,
                style=_CLASS_COLOR[current.predicted_class],
            )
            top_p = current.probs[current.predicted_class]
            cur_line = Text.assemble(
                (f"{current.reading.end_wall.strftime('%H:%M:%S')}  ", "dim"),
                ("predicted: ", ""), cur_class_text,
                ("  p=", "dim"), (f"{top_p:.2f}", "bold"),
                ("  latency=", "dim"), (f"{current.latency_ms:.1f}ms", ""),
            )
            probs_tbl = Table.grid(padding=(0, 2))
            probs_tbl.add_column(style="dim")
            for cls in _TYPE_ORDER:
                p = current.probs.get(cls, 0.0)
                probs_tbl.add_row(
                    Text(cls, style=_CLASS_COLOR[cls]),
                    Text(f"{p:.3f}", style="bold" if cls == current.predicted_class else ""),
                )
            stats = Text.assemble(
                ("pps_mean=", "dim"), (f"{current.reading.pps_mean:.0f}  ", ""),
                ("pps_max=",  "dim"), (f"{current.reading.pps_max:.0f}  ", ""),
                ("bytes/pkt=", "dim"), (f"{current.reading.bytes_per_pkt_mean:.0f}  ", ""),
                ("src_ips=",  "dim"), (f"{current.reading.distinct_src_ips}  ", ""),
                ("samples=",  "dim"), (f"{current.reading.n_samples}", ""),
            )
            body = Group(spark_line, Text(""), timeline, Text(""), cur_line, Text(""), probs_tbl, Text(""), stats)
        else:
            body = Group(spark_line, Text(""), timeline, Text(""),
                         Text("waiting for first window…", style="dim italic"))

        return Panel(body, title="signal", border_style="cyan")

    def _alerts(self) -> Panel:
        if not self.alerts:
            body: Text | Table = Text("no alerts yet", style="dim italic")
        else:
            tbl = Table.grid(padding=(0, 2))
            tbl.add_column(style="dim")
            tbl.add_column()
            for ts, msg in self.alerts:
                tbl.add_row(ts.strftime("%H:%M:%S"), Text(msg, style="bold red"))
            body = tbl
        return Panel(body, title=f"alerts (p_ddos ≥ {self.alert_prob:.2f})",
                     border_style="red")


# ---------------------------------------------------------------------------
# Glue: sampler thread + main loop
# ---------------------------------------------------------------------------

def _classify(registry: ModelRegistry, reading: WindowReading) -> Prediction:
    X_aug = registry.featurize_window(reading.events_df, reading.components_df)
    t0 = time.perf_counter()
    y_pred = registry.predict(X_aug)
    probs = registry.predict_proba(X_aug)
    latency_ms = (time.perf_counter() - t0) * 1000

    label_int = int(y_pred[0])
    predicted_class = INT_TO_TYPE[label_int]
    prob_row = probs[0]
    probs_by_class = {INT_TO_TYPE[i]: float(prob_row[i]) for i in range(len(prob_row))}
    # Ensure every class has an entry even if predict_proba returned fewer.
    for cls in _TYPE_ORDER:
        probs_by_class.setdefault(cls, 0.0)

    return Prediction(
        reading=reading,
        predicted_class=predicted_class,
        probs=probs_by_class,
        latency_ms=latency_ms,
    )


def _sampler_thread(
    sampler: HostSampler,
    stop_event: threading.Event,
    q: "queue.Queue[WindowReading | None]",
) -> None:
    try:
        for reading in sampler.stream_windows(stop_event):
            if stop_event.is_set():
                break
            q.put(reading)
    except Exception:
        logger.exception("sampler thread crashed")
    finally:
        q.put(None)  # sentinel


def run_monitor(args: argparse.Namespace) -> int:
    interface = args.interface or auto_detect_interface()

    console = Console()
    console.print(f"[dim]loading model registry from[/dim] {args.model_dir}")
    registry = ModelRegistry(model_dir=Path(args.model_dir))
    with console.status("[bold green]loading or training v2+cluster pipeline…[/]"):
        registry.load_or_train()

    if args.model not in registry.model_names:
        console.print(
            f"[red]model {args.model!r} not in registry. Available:[/red] "
            f"{', '.join(registry.model_names)}"
        )
        return 2
    registry.select(args.model)
    console.print(f"[dim]active model:[/dim] [bold]{registry.current_name}[/bold]")
    console.print(f"[dim]interface:[/dim] [bold]{interface}[/bold]")

    sampler = HostSampler(
        interface=interface,
        window_s=args.window,
        sample_s=args.sample,
        enable_src_ip_sniffer=not args.no_sniffer,
        port=args.port,
    )

    stop_event = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())

    q: queue.Queue[WindowReading | None] = queue.Queue(maxsize=8)
    sampler_t = threading.Thread(
        target=_sampler_thread, args=(sampler, stop_event, q), daemon=True,
    )
    sampler_t.start()

    view = MonitorView(
        interface=interface,
        model_name=registry.current_name or "?",
        window_s=args.window,
        alert_prob=args.alert_prob,
    )

    try:
        with Live(view.render(), console=console, refresh_per_second=4,
                  screen=False) as live:
            while not stop_event.is_set():
                try:
                    item = q.get(timeout=0.5)
                except queue.Empty:
                    live.update(view.render())
                    continue
                if item is None:
                    break
                pred = _classify(registry, item)
                view.update(pred)
                live.update(view.render())
    finally:
        stop_event.set()
        sampler_t.join(timeout=3)

    console.print(f"[dim]stopped — {view.windows_seen} windows processed, "
                  f"{view.ddos_windows} flagged DDoS.[/dim]")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.monitor",
        description="Live host-network DDoS monitor (v2+cluster).",
    )
    p.add_argument("--interface", default=None,
                   help="NIC name (default: auto-detect first non-lo)")
    p.add_argument("--window", type=float, default=5.0,
                   help="window length in seconds (default 5.0)")
    p.add_argument("--sample", type=float, default=0.1,
                   help="sub-second sample cadence (default 0.1)")
    p.add_argument("--model", default="Random Forest",
                   help="which classifier to use (default Random Forest)")
    p.add_argument("--model-dir", default="./models",
                   help="path to the cached registry (trained on first use)")
    p.add_argument("--alert-prob", type=float, default=0.5,
                   help="DDoS-class probability that raises an alert (default 0.5)")
    p.add_argument("--port", type=int, default=443,
                   help="port number to attribute the windows to — feeds the "
                        "v2 is_port_* binary features (default 443)")
    p.add_argument("--no-sniffer", action="store_true",
                   help="disable the AF_PACKET source-IP sniffer (no sudo needed)")
    p.add_argument("--log-level", default="WARNING",
                   help="root log level (default WARNING)")
    return p


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(levelname)s %(name)s: %(message)s")
    return run_monitor(args)


if __name__ == "__main__":
    raise SystemExit(main())
