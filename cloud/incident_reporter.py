"""
Qwen Cloud Skill 2: Incident Reporter.

On WARNING/EMERGENCY events, sends the camera frame + structured safety state
to Qwen-VL for a rich natural language incident report. Falls back to text-only
(qwen-turbo) if no frame is available.
"""
from __future__ import annotations

import base64
import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Optional, TYPE_CHECKING

import cv2
import numpy as np

from cloud.qwen_client import QwenCloudClient

if TYPE_CHECKING:
    from edge.safety.intervention import InterventionEvent

logger = logging.getLogger(__name__)

# Report times are narrated in the deployment's local zone (default GMT+8), not
# the FC server's UTC. Override with SAFEEDGE_TZ_OFFSET (hours) if relocating.
_TZ = timezone(timedelta(hours=float(os.environ.get("SAFEEDGE_TZ_OFFSET", "8"))))
_TZ_LABEL = f"GMT+{int(float(os.environ.get('SAFEEDGE_TZ_OFFSET', '8')))}"

_SYSTEM = """
You are a safety incident reporter for a commercial car park monitoring system.
You will receive a camera image and structured safety data from a formal STL monitor.
Write a concise 2-3 sentence incident report for a building safety log.
Describe what you can see in the image (vehicle type, pedestrian position, approximate
clearance). Be factual. Do not speculate about intent.
"""

_SYSTEM_TEXT_ONLY = """
You are a safety incident reporter for a commercial car park monitoring system.
Given structured safety event data, write a concise 2-3 sentence incident report
for a building safety log. Use plain English. Be factual about distances and timing.
"""


class IncidentReporter:
    def __init__(
        self,
        vision_model: Optional[str] = None,
        text_model:   Optional[str] = None,
        location:     str = "Car Park",
    ):
        vision_model = vision_model or os.environ.get("QWEN_VISION_MODEL", "qwen-vl-max")
        text_model   = text_model   or os.environ.get("QWEN_TEXT_MODEL",   "qwen-turbo")
        self._vision_client = QwenCloudClient(model=vision_model, max_tokens=300)
        self._text_client   = QwenCloudClient(model=text_model,   max_tokens=256)
        self._location = location

    def report(
        self,
        event: "InterventionEvent",
        frame_bgr: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        ts = datetime.fromtimestamp(event.timestamp, _TZ).strftime("%Y-%m-%d %H:%M:%S") \
             + f" {_TZ_LABEL}"
        context = {
            "timestamp": ts,
            "location": self._location,
            "severity": {1: "AWARENESS", 2: "WARNING", 3: "EMERGENCY"}.get(
                event.new_level, "UNKNOWN"
            ),
            "pedestrian_vehicle_distance_m": round(event.d_min, 2),
            "vehicle_speed_ms": round(event.v_veh_max, 2),
            "predicted_clearance_m": round(event.d_pred, 2),
            "rho_min": round(event.rho_min, 3),
            "stl_message": event.message,
        }

        if frame_bgr is not None:
            return self._report_with_vision(context, frame_bgr)
        return self._report_text_only(context)

    # ── Private ──────────────────────────────────────────────────────────────

    def _report_with_vision(self, context: dict, frame_bgr: np.ndarray) -> Optional[str]:
        img_b64 = _encode_frame(frame_bgr)
        user_content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            },
            {
                "type": "text",
                "text": (
                    f"Safety alert detected by STL monitor:\n"
                    f"{json.dumps(context, indent=2)}\n\n"
                    f"Write the incident report based on what you can see in the image "
                    f"and the structured data above."
                ),
            },
        ]
        result = self._vision_client.chat_multimodal(
            system=_SYSTEM, user_content=user_content
        )
        if result:
            logger.info("Vision incident report generated (t=%s)", context["timestamp"])
        return result

    def _report_text_only(self, context: dict) -> Optional[str]:
        result = self._text_client.chat(
            system=_SYSTEM_TEXT_ONLY,
            user=json.dumps(context, indent=2),
        )
        if result:
            logger.info("Text incident report generated (t=%s)", context["timestamp"])
        return result


def _encode_frame(frame_bgr: np.ndarray, max_dim: int = 640) -> str:
    """Resize to max_dim on longest side, encode as JPEG base64."""
    h, w = frame_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode()
