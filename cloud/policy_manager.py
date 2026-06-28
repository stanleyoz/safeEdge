"""
Qwen Cloud Skill 1: Policy Manager.

Analyses recent robustness history and event counts, then returns a JSON
parameter patch that the STLMonitor.apply_cloud_params() method applies live.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from cloud.qwen_client import QwenCloudClient

logger = logging.getLogger(__name__)

_SYSTEM = """
You are a safety policy manager for an edge-deployed car park monitoring system
using Signal Temporal Logic (STL) for formal safety verification.

You receive:
- A summary of robustness values (ρ1–ρ5) over the last monitoring window
- Event counts by severity level
- Current STL parameter values
- Optional operator context (time of day, known activities)

Your task: recommend parameter adjustments that maintain safety without
generating excessive false-positive alerts.

Respond ONLY with a JSON object. Include only specs whose params you want to
change. Example: {"phi2": {"proximity_zone": 6.0}, "phi4": {"stop_window": 75}}
If no changes needed, respond with: {}
"""


class PolicyManager:
    def __init__(self, model: str = "qwen-plus"):
        self._client = QwenCloudClient(model=model, max_tokens=512)

    def request_update(
        self,
        rho_summary: dict,
        event_counts: dict,
        current_params: dict,
        context: str = "",
    ) -> Optional[dict]:
        user_msg = json.dumps({
            "rho_summary": rho_summary,
            "event_counts": event_counts,
            "current_params": current_params,
            "operator_context": context,
        }, indent=2)

        raw = self._client.chat(system=_SYSTEM, user=user_msg, json_mode=True)
        if raw is None:
            return None

        try:
            patch = json.loads(raw)
            if not isinstance(patch, dict):
                return None
            logger.info("Policy Manager patch received: %s", patch)
            return patch
        except json.JSONDecodeError as exc:
            logger.warning("Policy Manager returned invalid JSON: %s", exc)
            return None
