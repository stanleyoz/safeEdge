"""Pydantic request/response schemas for the SafeEdge cloud API.

These define the edge↔cloud contract. The edge posts JSON; the backend never
imports edge dataclasses, so the two halves deploy independently.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ── Edge → Cloud ────────────────────────────────────────────────────────────

class RhoValues(BaseModel):
    rho1: Optional[float] = None
    rho2: Optional[float] = None
    rho3: Optional[float] = None
    rho4: Optional[float] = None
    rho5: Optional[float] = None


class Signals(BaseModel):
    d_min: float = 0.0
    v_veh_max: float = 0.0
    d_pred: float = 0.0


class StatePush(BaseModel):
    """Throttled per-frame state for the live dashboard."""
    t: int
    timestamp: float
    level: int
    level_label: str = ""
    rho: RhoValues = Field(default_factory=RhoValues)
    signals: Signals = Field(default_factory=Signals)
    scale_factor: float = 1.0
    frame_jpeg_b64: Optional[str] = None   # optional preview frame


class EventPush(BaseModel):
    """An intervention event from the STL monitor."""
    timestamp: float
    level: int                     # 1=awareness 2=warning 3=emergency
    d_min: float
    v_veh_max: float
    d_pred: float
    rho_min: float
    message: str = ""
    frame_jpeg_b64: Optional[str] = None   # for vision incident report


class PolicyEvalRequest(BaseModel):
    """Edge asks the cloud Policy Manager to evaluate current conditions."""
    rho_summary: dict = Field(default_factory=dict)
    event_counts: dict = Field(default_factory=dict)
    current_params: dict = Field(default_factory=dict)
    context: str = ""


# ── Cloud → Edge / Dashboard ──────────────────────────────────────────────────

class PolicyEvalResponse(BaseModel):
    patch: dict = Field(default_factory=dict)   # STL param patch ({} = no change)


class IncidentRecord(BaseModel):
    id: str
    timestamp: float
    level: int
    report: str
    d_min: float
    v_veh_max: float


class ForecastResponse(BaseModel):
    high_risk_windows: list = Field(default_factory=list)
    recommendations: list = Field(default_factory=list)
    generated_at: Optional[float] = None
