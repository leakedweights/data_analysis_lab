"""FastAPI application for the DDoS detection demo."""

from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from src.api.model_registry import ModelRegistry
from src.api.models import (
    ModelInfo,
    ModelListResponse,
    Scenario,
    ScenarioListResponse,
    SelectModelRequest,
    SimulationStatus,
    StartSimulationRequest,
)
from src.api.stream_engine import REDIS_CHANNEL, SCENARIOS, StreamEngine
from src.live_capture import hping3_available

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

REDIS_URL = "redis://redis:6379"
STATIC_DIR = Path(__file__).parent / "static"

registry = ModelRegistry()
engine: StreamEngine | None = None
_models_ready = False


def _train_models_background() -> None:
    global _models_ready, engine
    try:
        registry.load_or_train()
        engine = StreamEngine(registry)
        _models_ready = True
        logger.info("Models ready — API fully operational.")
    except Exception:
        logger.exception("Failed to load/train models")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info("Starting up — training models in background ...")
    thread = threading.Thread(target=_train_models_background, daemon=True)
    thread.start()
    yield
    if engine and engine.running:
        await engine.stop()


app = FastAPI(title="DDoS Detection Demo API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Dashboard (static HTML)
# ---------------------------------------------------------------------------
@app.get("/")
async def dashboard():
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


# ---------------------------------------------------------------------------
# SSE endpoint — streams window results from Redis list
# ---------------------------------------------------------------------------
@app.get("/api/simulation/stream")
async def simulation_stream(request: Request):
    async def generate():
        r = aioredis.from_url(REDIS_URL)
        try:
            while True:
                if await request.is_disconnected():
                    break
                raw = await r.lpop(REDIS_CHANNEL)
                if raw:
                    data = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                    yield f"data: {data}\n\n"
                else:
                    yield ": keepalive\n\n"
                    await asyncio.sleep(0.3)
        except asyncio.CancelledError:
            pass
        finally:
            await r.aclose()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {"status": "ready" if _models_ready else "loading"}


@app.get("/api/models", response_model=ModelListResponse)
async def list_models():
    if not _models_ready:
        raise HTTPException(status_code=503, detail="Models are still loading")
    return ModelListResponse(
        models=[
            ModelInfo(name=n, is_active=(n == registry.current_name))
            for n in registry.model_names
        ]
    )


@app.post("/api/models/select")
async def select_model(req: SelectModelRequest):
    if not _models_ready:
        raise HTTPException(status_code=503, detail="Models are still loading")
    try:
        registry.select(req.name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{req.name}' not found")
    return {"status": "ok", "active_model": registry.current_name}


@app.get("/api/scenarios", response_model=ScenarioListResponse)
async def list_scenarios():
    has_hping3 = hping3_available()
    return ScenarioListResponse(
        scenarios=[
            Scenario(id=s.id, name=s.name, description=s.description, is_live=s.is_live)
            for s in SCENARIOS.values()
            if not s.is_live or has_hping3  # hide live scenarios when hping3 is missing
        ]
    )


@app.post("/api/simulation/start")
async def start_simulation(req: StartSimulationRequest):
    if not _models_ready or engine is None:
        raise HTTPException(status_code=503, detail="Models are still loading")
    if req.scenario_id not in SCENARIOS:
        raise HTTPException(status_code=404, detail=f"Scenario '{req.scenario_id}' not found")
    await engine.start(req.scenario_id)
    return {"status": "started", "scenario_id": req.scenario_id}


@app.post("/api/simulation/stop")
async def stop_simulation():
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    await engine.stop()
    return {"status": "stopped"}


@app.get("/api/simulation/status", response_model=SimulationStatus)
async def simulation_status():
    if engine is None:
        return SimulationStatus(running=False)
    scenario = SCENARIOS.get(engine.scenario_id or "") if engine.scenario_id else None
    return SimulationStatus(
        running=engine.running,
        scenario_id=engine.scenario_id,
        scenario_name=scenario.name if scenario else None,
        active_model=registry.current_name,
        windows_processed=engine.windows_processed,
        total_events=engine.total_events,
        total_ddos_detected=engine.total_ddos_detected,
        total_ddos_actual=engine.total_ddos_actual,
        alert_count=engine.alert_count,
        false_alert_count=engine.false_alert_count,
    )
