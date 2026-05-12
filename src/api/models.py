"""Pydantic request/response schemas for the DDoS detection demo API."""

from __future__ import annotations

from pydantic import BaseModel


class ModelInfo(BaseModel):
    name: str
    is_active: bool


class ModelListResponse(BaseModel):
    models: list[ModelInfo]


class SelectModelRequest(BaseModel):
    name: str


class Scenario(BaseModel):
    id: str
    name: str
    description: str
    is_live: bool = False


class ScenarioListResponse(BaseModel):
    scenarios: list[Scenario]


class StartSimulationRequest(BaseModel):
    scenario_id: str


class SimulationStatus(BaseModel):
    running: bool
    scenario_id: str | None = None
    scenario_name: str | None = None
    active_model: str | None = None
    windows_processed: int = 0
    total_events: int = 0
    total_ddos_detected: int = 0
    total_ddos_actual: int = 0
    alert_count: int = 0
    false_alert_count: int = 0


class WindowResultEvent(BaseModel):
    """Single window result published to Redis."""
    window_index: int
    window_start: str
    window_end: str
    n_events: int
    n_predicted_normal: int
    n_predicted_suspicious: int
    n_predicted_ddos: int
    n_actual_normal: int
    n_actual_suspicious: int
    n_actual_ddos: int
    accuracy: float
    detection_latency_ms: float
    alert_raised: bool
    model_name: str
    scenario_id: str
