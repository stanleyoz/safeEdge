"""Backend adapters for the three Qwen Cloud skills.

Reuses the existing implementations in cloud/ but exposes dict-friendly entry
points so the backend has no dependency on the edge package. The Qwen calls
themselves live in cloud/qwen_client.py (DashScope / Alibaba Cloud).
"""
from __future__ import annotations

import base64
import logging
from types import SimpleNamespace
from typing import Optional

import numpy as np

from cloud.incident_reporter import IncidentReporter
from cloud.policy_manager import PolicyManager
from cloud.risk_forecaster import RiskForecaster

logger = logging.getLogger(__name__)


class Skills:
    """Lazily-constructed singletons for the three Qwen skills."""

    def __init__(self, location: str = "Car Park"):
        self._location = location
        self._incident: Optional[IncidentReporter] = None
        self._policy: Optional[PolicyManager] = None
        self._risk: Optional[RiskForecaster] = None

    @property
    def incident(self) -> IncidentReporter:
        if self._incident is None:
            self._incident = IncidentReporter(location=self._location)
        return self._incident

    @property
    def policy(self) -> PolicyManager:
        if self._policy is None:
            self._policy = PolicyManager()
        return self._policy

    @property
    def risk(self) -> RiskForecaster:
        if self._risk is None:
            self._risk = RiskForecaster()
        return self._risk

    # ── Incident report from a posted event dict ──────────────────────────────
    def incident_report(self, event: dict) -> Optional[str]:
        """Build the report. event matches EventPush; frame is base64 JPEG."""
        ev = SimpleNamespace(
            timestamp=event["timestamp"],
            new_level=event["level"],
            d_min=event["d_min"],
            v_veh_max=event["v_veh_max"],
            d_pred=event["d_pred"],
            rho_min=event["rho_min"],
            message=event.get("message", ""),
        )
        frame_bgr = _decode_frame(event.get("frame_jpeg_b64"))
        return self.incident.report(ev, frame_bgr)

    def policy_eval(self, req: dict) -> dict:
        patch = self.policy.request_update(
            rho_summary=req.get("rho_summary", {}),
            event_counts=req.get("event_counts", {}),
            current_params=req.get("current_params", {}),
            context=req.get("context", ""),
        )
        return patch or {}

    def forecast(self, events: list[dict]) -> dict:
        result = self.risk.forecast(events)
        return result or {"high_risk_windows": [], "recommendations": []}


def _decode_frame(b64: Optional[str]) -> Optional[np.ndarray]:
    if not b64:
        return None
    try:
        import cv2
        buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception as exc:  # noqa: BLE001
        logger.warning("frame decode failed: %s", exc)
        return None
