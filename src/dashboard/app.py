"""Streamlit dashboard — real-time DDoS detection monitoring."""

from __future__ import annotations

import json
import time

import numpy as np
import plotly.graph_objects as go
import requests
import redis
import streamlit as st

API_URL = "http://api:8000"
REDIS_URL = "redis://redis:6379"
REDIS_CHANNEL = "ddos:windows"

# ---------------------------------------------------------------------------
# Page config & custom CSS
# ---------------------------------------------------------------------------
st.set_page_config(page_title="DDoS Detection — Live Monitor", layout="wide")

DARK = "#0a0e17"
PANEL = "#111827"
CARD = "#1a2332"
ACCENT = "#00d4ff"
GREEN = "#10b981"
AMBER = "#f59e0b"
RED = "#ef4444"
MUTED = "#64748b"
TEXT = "#e2e8f0"

st.markdown(f"""
<style>
    /* hide Streamlit chrome: toolbar, deploy button, header */
    header[data-testid="stHeader"],
    #MainMenu,
    .stDeployButton,
    footer {{ visibility: hidden; height: 0; }}
    div[data-testid="stToolbar"] {{ display: none; }}
    div[data-testid="stDecoration"] {{ display: none; }}
    .block-container {{ padding-top: 1rem; padding-bottom: 0; }}
    /* header bar */
    .header-bar {{
        background: linear-gradient(135deg, {PANEL} 0%, #0f1729 100%);
        border-bottom: 1px solid #1e293b;
        padding: 0.8rem 1.5rem;
        margin: -1.5rem -1rem 1rem -1rem;
        display: flex; align-items: center; gap: 1rem;
    }}
    .header-bar h1 {{
        margin: 0; font-size: 1.4rem; color: {ACCENT};
        font-family: monospace; letter-spacing: 0.05em;
    }}
    .header-bar .subtitle {{ color: {MUTED}; font-size: 0.85rem; }}
    /* KPI cards */
    .kpi-card {{
        background: {CARD};
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        text-align: center;
    }}
    .kpi-card .label {{ color: {MUTED}; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.2rem; }}
    .kpi-card .value {{ font-size: 1.6rem; font-weight: 700; font-family: monospace; }}
    .kpi-card .delta {{ font-size: 0.75rem; margin-top: 0.15rem; }}
    /* alert pulse */
    @keyframes pulse {{ 0%,100% {{ opacity:1 }} 50% {{ opacity:0.5 }} }}
    .alert-pulse {{ animation: pulse 1s infinite; }}
    /* live feed rows */
    .feed-row {{
        display: flex; align-items: center; gap: 0.5rem;
        padding: 0.3rem 0.5rem; border-radius: 4px;
        font-family: monospace; font-size: 0.78rem;
        margin-bottom: 2px;
    }}
    .feed-row.normal {{ background: rgba(16,185,129,0.08); border-left: 3px solid {GREEN}; }}
    .feed-row.suspicious {{ background: rgba(245,158,11,0.08); border-left: 3px solid {AMBER}; }}
    .feed-row.ddos {{ background: rgba(239,68,68,0.08); border-left: 3px solid {RED}; }}
    .feed-row .tag {{
        display: inline-block; padding: 1px 6px; border-radius: 3px;
        font-size: 0.7rem; font-weight: 600;
    }}
    .tag-normal {{ background: {GREEN}; color: #000; }}
    .tag-suspicious {{ background: {AMBER}; color: #000; }}
    .tag-ddos {{ background: {RED}; color: #fff; }}
    .tag-correct {{ color: {GREEN}; }}
    .tag-wrong {{ color: {RED}; }}
    /* section labels */
    .section-label {{
        color: {MUTED}; font-size: 0.75rem; text-transform: uppercase;
        letter-spacing: 0.1em; margin-bottom: 0.5rem; border-bottom: 1px solid #1e293b;
        padding-bottom: 0.3rem;
    }}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-bar">
    <h1>NETWORK THREAT MONITOR</h1>
    <span class="subtitle">Real-time DDoS Detection &amp; Classification</span>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in [
    ("windows", []),
    ("sim_running", False),
    ("confusion", np.zeros((3, 3), dtype=int)),
    ("alerts", []),
    ("all_samples", []),
    ("cumulative_events", 0),
    ("cumulative_ddos", 0),
    ("cumulative_correct", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def api_get(path: str):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_post(path: str, data: dict | None = None):
    try:
        r = requests.post(f"{API_URL}{path}", json=data or {}, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# API readiness gate
# ---------------------------------------------------------------------------
health_resp = api_get("/api/health")
models_resp = api_get("/api/models")

if models_resp is None:
    loading = health_resp and health_resp.get("status") == "loading"
    if loading:
        st.info("Training ML models on first startup. This takes ~30 seconds. Auto-refreshing ...")
    else:
        st.warning("Waiting for API ...")
    time.sleep(3)
    st.rerun()

# ---------------------------------------------------------------------------
# Control bar (sidebar)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f'<div class="section-label">Model Selection</div>', unsafe_allow_html=True)
    model_names = [m["name"] for m in models_resp["models"]]
    active_model = next((m["name"] for m in models_resp["models"] if m["is_active"]), model_names[0])
    selected_model = st.selectbox("Model", model_names, index=model_names.index(active_model), label_visibility="collapsed")
    if selected_model != active_model:
        api_post("/api/models/select", {"name": selected_model})
        st.rerun()

    st.markdown(f'<div class="section-label">Attack Scenario</div>', unsafe_allow_html=True)
    scenarios_resp = api_get("/api/scenarios")
    scenario_list = scenarios_resp["scenarios"] if scenarios_resp else []
    scenario_map = {s["name"]: s for s in scenario_list}
    selected_scenario_name = st.selectbox("Scenario", list(scenario_map.keys()), label_visibility="collapsed")
    if selected_scenario_name:
        sc = scenario_map[selected_scenario_name]
        st.caption(sc["description"])
        if sc.get("is_live"):
            st.markdown(f'<div style="background:rgba(239,68,68,0.15);border:1px solid {RED};border-radius:4px;padding:0.3rem 0.6rem;text-align:center;font-family:monospace;font-size:0.8rem;color:{RED};margin-top:0.3rem"><b>LIVE TRAFFIC</b> &mdash; hping3</div>', unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    start_clicked = c1.button("Start", type="primary", use_container_width=True)
    stop_clicked = c2.button("Stop", use_container_width=True)
    clear_clicked = st.button("Clear Data", use_container_width=True)

    status_resp = api_get("/api/simulation/status")
    running = status_resp and status_resp.get("running", False)

    if running:
        source_label = ""
        if status_resp and status_resp.get("scenario_id", "").startswith("live_"):
            source_label = f'<div style="text-align:center;margin-top:0.3rem;font-size:0.7rem;color:{RED};font-family:monospace">SRC: hping3</div>'
        else:
            source_label = f'<div style="text-align:center;margin-top:0.3rem;font-size:0.7rem;color:{ACCENT};font-family:monospace">SRC: synthetic</div>'
        st.markdown(f'<div style="text-align:center;margin-top:0.5rem"><span class="alert-pulse" style="color:{GREEN};font-size:1.1rem;font-weight:bold">LIVE</span></div>{source_label}', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="text-align:center;margin-top:0.5rem;color:{MUTED}">IDLE</div>', unsafe_allow_html=True)

# Button handlers
if clear_clicked:
    for k in ["windows", "alerts", "all_samples"]:
        st.session_state[k] = [] if k != "confusion" else None
    st.session_state.confusion = np.zeros((3, 3), dtype=int)
    st.session_state.cumulative_events = 0
    st.session_state.cumulative_ddos = 0
    st.session_state.cumulative_correct = 0
    st.rerun()

if start_clicked and selected_scenario_name:
    for k in ["windows", "alerts", "all_samples"]:
        st.session_state[k] = []
    st.session_state.confusion = np.zeros((3, 3), dtype=int)
    st.session_state.cumulative_events = 0
    st.session_state.cumulative_ddos = 0
    st.session_state.cumulative_correct = 0
    scenario_id = scenario_map[selected_scenario_name]["id"]
    api_post("/api/simulation/start", {"scenario_id": scenario_id})
    st.session_state.sim_running = True
    time.sleep(0.5)
    st.rerun()

if stop_clicked:
    api_post("/api/simulation/stop")
    st.session_state.sim_running = False
    st.rerun()


# ---------------------------------------------------------------------------
# Redis poll
# ---------------------------------------------------------------------------
def poll_redis(max_messages: int = 100) -> list[dict]:
    """Pop all pending window results from the Redis list."""
    messages = []
    try:
        r = redis.from_url(REDIS_URL)
        while len(messages) < max_messages:
            raw = r.lpop(REDIS_CHANNEL)
            if raw is None:
                break
            data = json.loads(raw)
            if data.get("done"):
                st.session_state.sim_running = False
                break
            messages.append(data)
        r.close()
    except Exception:
        pass
    return messages


if st.session_state.sim_running or running:
    new_windows = poll_redis()
    if new_windows:
        st.session_state.windows.extend(new_windows)
        for w in new_windows:
            st.session_state.cumulative_events += w["n_events"]
            st.session_state.cumulative_ddos += w["n_actual_ddos"]
            st.session_state.cumulative_correct += int(w["accuracy"] * w["n_events"])
            # Confusion matrix
            for cls_idx, cls_name in enumerate(["normal", "suspicious", "ddos"]):
                actual = w.get(f"n_actual_{cls_name}", 0)
                pred = w.get(f"n_predicted_{cls_name}", 0)
                st.session_state.confusion[cls_idx][cls_idx] += min(actual, pred)
            # Alerts
            if w.get("alert_raised"):
                alert_type = "TRUE" if w["n_actual_ddos"] > 0 else "FALSE"
                st.session_state.alerts.append({
                    "window": w["window_index"],
                    "type": alert_type,
                    "predicted_ddos": w["n_predicted_ddos"],
                    "actual_ddos": w["n_actual_ddos"],
                    "accuracy": w["accuracy"],
                })
            # Event samples
            for s in w.get("event_samples", []):
                s["window"] = w["window_index"]
                st.session_state.all_samples.append(s)


windows = st.session_state.windows

# Plotly dark template
_PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT, family="monospace", size=11),
    margin=dict(l=45, r=15, t=30, b=35),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
)
_GRID = dict(gridcolor="#1e293b", zerolinecolor="#1e293b")

def dark_layout(**overrides):
    """Build a Plotly layout dict with dark theme, merging axis options."""
    layout = {**_PLOTLY_BASE, **overrides}
    layout["xaxis"] = {**_GRID, **layout.get("xaxis", {})}
    layout["yaxis"] = {**_GRID, **layout.get("yaxis", {})}
    return layout

if not windows:
    st.markdown(f"""
    <div style="text-align:center; padding:4rem 2rem; color:{MUTED}">
        <div style="font-size:3rem; margin-bottom:1rem">&#x1f6e1;</div>
        <div style="font-size:1.1rem; margin-bottom:0.5rem">Network Threat Monitor</div>
        <div style="font-size:0.85rem">Select a model and attack scenario in the sidebar, then click <b>Start</b> to begin real-time detection.</div>
    </div>
    """, unsafe_allow_html=True)
else:
    # -----------------------------------------------------------------------
    # KPI row
    # -----------------------------------------------------------------------
    total_ev = st.session_state.cumulative_events
    total_ddos = st.session_state.cumulative_ddos
    total_correct = st.session_state.cumulative_correct
    n_windows = len(windows)
    overall_acc = total_correct / total_ev if total_ev > 0 else 0
    ddos_pct = total_ddos / total_ev * 100 if total_ev > 0 else 0
    avg_latency = np.mean([w["detection_latency_ms"] for w in windows]) if windows else 0
    n_alerts = len(st.session_state.alerts)
    false_alerts = sum(1 for a in st.session_state.alerts if a["type"] == "FALSE")
    last_w = windows[-1]
    current_rate = last_w.get("avg_pkt_rate", 0)
    threat_level = "CRITICAL" if last_w["n_predicted_ddos"] > 5 else "HIGH" if last_w["n_predicted_ddos"] > 2 else "ELEVATED" if last_w["n_predicted_ddos"] > 0 else "NORMAL"
    threat_color = RED if "CRITICAL" in threat_level else AMBER if "HIGH" in threat_level else AMBER if "ELEVATED" in threat_level else GREEN

    # Determine traffic source from window data
    traffic_source = last_w.get("source", "synthetic")
    source_color = RED if traffic_source == "hping3" else ACCENT
    source_badge = f'<span style="background:{source_color};color:#000;padding:1px 8px;border-radius:3px;font-size:0.7rem;font-weight:700;font-family:monospace;margin-left:0.5rem">{traffic_source.upper()}</span>'

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    def kpi(col, label, value, color=ACCENT, delta=None):
        delta_html = f'<div class="delta" style="color:{MUTED}">{delta}</div>' if delta else ""
        col.markdown(f"""
        <div class="kpi-card">
            <div class="label">{label}</div>
            <div class="value" style="color:{color}">{value}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)

    kpi(k1, "Threat Level", threat_level, threat_color, f"source: {traffic_source}")
    kpi(k2, "Events Processed", f"{total_ev:,}", ACCENT, f"{n_windows} windows")
    kpi(k3, "DDoS Detected", f"{total_ddos:,}", RED if total_ddos > 0 else GREEN, f"{ddos_pct:.1f}% of traffic")
    kpi(k4, "Model Accuracy", f"{overall_acc:.1%}", GREEN if overall_acc > 0.9 else AMBER, f"{selected_model}")
    kpi(k5, "Avg Latency", f"{avg_latency:.1f}ms", GREEN if avg_latency < 5 else AMBER)
    kpi(k6, "Alerts", f"{n_alerts}", RED if false_alerts > 0 else GREEN, f"{false_alerts} false" if false_alerts else "all verified")

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Main charts: traffic flow + predictions
    # -----------------------------------------------------------------------
    col_main, col_feed = st.columns([3, 1])

    with col_main:
        st.markdown(f'<div class="section-label">Live Traffic Flow &mdash; Predictions by Type</div>', unsafe_allow_html=True)
        indices = list(range(len(windows)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=indices, y=[w["n_predicted_normal"] for w in windows],
            mode="lines", name="Normal", fill="tozeroy",
            line=dict(color=GREEN, width=1), fillcolor="rgba(16,185,129,0.15)",
        ))
        fig.add_trace(go.Scatter(
            x=indices, y=[w["n_predicted_suspicious"] for w in windows],
            mode="lines", name="Suspicious", fill="tozeroy",
            line=dict(color=AMBER, width=1), fillcolor="rgba(245,158,11,0.15)",
        ))
        fig.add_trace(go.Scatter(
            x=indices, y=[w["n_predicted_ddos"] for w in windows],
            mode="lines", name="DDoS", fill="tozeroy",
            line=dict(color=RED, width=2), fillcolor="rgba(239,68,68,0.25)",
        ))

        # Alert markers
        alert_idx = [i for i, w in enumerate(windows) if w["alert_raised"]]
        alert_y = [windows[i]["n_events"] for i in alert_idx]
        if alert_idx:
            fig.add_trace(go.Scatter(
                x=alert_idx, y=alert_y,
                mode="markers", name="Alert",
                marker=dict(symbol="triangle-down", size=10, color=RED,
                           line=dict(width=1, color="#fff")),
            ))

        fig.update_layout(**dark_layout(height=300,
                         xaxis_title="Window", yaxis_title="Events"))
        st.plotly_chart(fig, use_container_width=True, key="traffic_flow")

        # --- Actual vs Predicted comparison ---
        st.markdown(f'<div class="section-label">Ground Truth vs Predictions</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=indices, y=[w["n_actual_ddos"] for w in windows],
            mode="lines", name="Actual DDoS",
            line=dict(color="#f87171", width=2, dash="dot"),
        ))
        fig2.add_trace(go.Scatter(
            x=indices, y=[w["n_predicted_ddos"] for w in windows],
            mode="lines", name="Predicted DDoS",
            line=dict(color=RED, width=2),
        ))
        fig2.add_trace(go.Scatter(
            x=indices, y=[w["n_actual_normal"] for w in windows],
            mode="lines", name="Actual Normal",
            line=dict(color="#6ee7b7", width=1, dash="dot"),
        ))
        fig2.add_trace(go.Scatter(
            x=indices, y=[w["n_predicted_normal"] for w in windows],
            mode="lines", name="Predicted Normal",
            line=dict(color=GREEN, width=1),
        ))
        fig2.update_layout(**dark_layout(height=250,
                          xaxis_title="Window", yaxis_title="Events"))
        st.plotly_chart(fig2, use_container_width=True, key="pred_vs_actual")

    # --- Live event feed ---
    with col_feed:
        st.markdown(f'<div class="section-label">Live Event Feed</div>', unsafe_allow_html=True)
        samples = st.session_state.all_samples
        if samples:
            for s in reversed(samples[-30:]):
                pred = s["predicted"]
                cls = "normal" if "Normal" in pred else "ddos" if "DDoS" in pred else "suspicious"
                tag_cls = f"tag-{cls}"
                correct_icon = f'<span class="tag-correct">&#10003;</span>' if s["correct"] else f'<span class="tag-wrong">&#10007;</span>'
                st.markdown(f"""
                <div class="feed-row {cls}">
                    <span class="tag {tag_cls}">{pred.split()[0][:3].upper()}</span>
                    <span style="color:{MUTED}">:{s['port']}</span>
                    <span style="color:{TEXT}">{s['pkt_speed']:,} pps</span>
                    {correct_icon}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="color:{MUTED};font-size:0.8rem;padding:1rem">Waiting for events...</div>', unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Bottom row: Accuracy/Latency + Confusion matrix + Alert log
    # -----------------------------------------------------------------------
    col_metrics, col_cm, col_alerts = st.columns([2, 1.5, 1.5])

    with col_metrics:
        st.markdown(f'<div class="section-label">Model Performance Over Time</div>', unsafe_allow_html=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=indices, y=[w["accuracy"] for w in windows],
            mode="lines", name="Accuracy",
            line=dict(color=ACCENT, width=2),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.08)",
        ))
        # Add rolling average
        if len(windows) > 5:
            accs = [w["accuracy"] for w in windows]
            rolling = np.convolve(accs, np.ones(5)/5, mode="valid")
            fig3.add_trace(go.Scatter(
                x=list(range(4, 4 + len(rolling))), y=rolling.tolist(),
                mode="lines", name="Rolling Avg (5w)",
                line=dict(color="#818cf8", width=2, dash="dash"),
            ))
        fig3.update_layout(**dark_layout(height=220,
                          yaxis=dict(range=[0, 1.05], title="Accuracy"), xaxis_title="Window"))
        st.plotly_chart(fig3, use_container_width=True, key="accuracy")

        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=indices, y=[w["detection_latency_ms"] for w in windows],
            name="Latency",
            marker=dict(
                color=[GREEN if w["detection_latency_ms"] < 5 else AMBER if w["detection_latency_ms"] < 20 else RED for w in windows],
            ),
        ))
        fig4.update_layout(**dark_layout(height=180,
                          yaxis_title="Latency (ms)", xaxis_title="Window"))
        st.plotly_chart(fig4, use_container_width=True, key="latency")

    with col_cm:
        st.markdown(f'<div class="section-label">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = st.session_state.confusion
        labels = ["Normal", "Suspicious", "DDoS"]
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=labels, y=labels,
            text=cm.astype(int).astype(str),
            texttemplate="%{text}",
            textfont=dict(size=14, color="#fff"),
            colorscale=[[0, PANEL], [0.5, "#1e40af"], [1, RED]],
            showscale=False,
        ))
        fig_cm.update_layout(**dark_layout(height=280,
                            xaxis_title="Predicted",
                            yaxis=dict(title="Actual", autorange="reversed")))
        st.plotly_chart(fig_cm, use_container_width=True, key="confusion")

    with col_alerts:
        st.markdown(f'<div class="section-label">Alert Log</div>', unsafe_allow_html=True)
        alerts = st.session_state.alerts
        if alerts:
            for a in reversed(alerts[-15:]):
                is_true = a["type"] == "TRUE"
                bg = "rgba(239,68,68,0.1)" if not is_true else "rgba(16,185,129,0.1)"
                border = RED if not is_true else GREEN
                icon = "&#x2705;" if is_true else "&#x26a0;&#xfe0f;"
                label = "VERIFIED" if is_true else "FALSE POSITIVE"
                st.markdown(f"""
                <div style="background:{bg};border-left:3px solid {border};padding:0.4rem 0.6rem;border-radius:4px;margin-bottom:3px;font-family:monospace;font-size:0.75rem">
                    {icon} <b>{label}</b> W{a['window']}
                    &mdash; pred:{a['predicted_ddos']} actual:{a['actual_ddos']}
                    acc:{a['accuracy']:.0%}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="color:{MUTED};font-size:0.8rem;padding:1rem">No alerts raised yet.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Auto-refresh while running
# ---------------------------------------------------------------------------
if st.session_state.sim_running or running:
    time.sleep(1.0)
    st.rerun()
