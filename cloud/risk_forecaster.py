"""
Qwen Cloud Skill 3: Risk Forecaster.

Runs hourly on the event history and predicts high-risk time windows for the
next 24 hours.  Results feed the dashboard's "upcoming risk" panel.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from cloud.qwen_client import QwenCloudClient

logger = logging.getLogger(__name__)

_SYSTEM = """
You are a predictive safety analyst for a car park safety monitoring system.
Given a list of historical safety events (with timestamps and severity), identify
temporal patterns and predict the next high-risk time windows for the coming day.

Respond ONLY with JSON matching this schema exactly:
{
  "high_risk_windows": [
    {"start": "HH:MM", "end": "HH:MM", "confidence": 0.0-1.0, "reason": "string"}
  ],
  "recommendations": ["string", ...]
}
If there is insufficient data to identify patterns, return empty arrays.
"""


class RiskForecaster:
    def __init__(self, model: str = "qwen-plus"):
        self._client = QwenCloudClient(model=model, max_tokens=512)

    def forecast(self, events: list[dict]) -> Optional[dict]:
        if len(events) < 5:
            return {"high_risk_windows": [], "recommendations": ["Insufficient event history for pattern analysis."]}

        user_msg = json.dumps({"recent_events": events[-50:]}, indent=2)
        raw = self._client.chat(system=_SYSTEM, user=user_msg, json_mode=True)
        if raw is None:
            return None

        try:
            result = json.loads(raw)
            logger.info(
                "Risk forecast: %d high-risk windows identified",
                len(result.get("high_risk_windows", [])),
            )
            return result
        except json.JSONDecodeError as exc:
            logger.warning("Risk forecaster returned invalid JSON: %s", exc)
            return None
