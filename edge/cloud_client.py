"""Edge → Cloud HTTP client.

Posts safety state, intervention events, and policy-evaluation requests to the
SafeEdge cloud backend (deployed on Alibaba Cloud Function Compute). Every call
is fire-and-forget on a small thread pool with a hard timeout, so a slow or
unreachable cloud NEVER blocks or crashes the 30Hz safety loop — the edge stays
fully autonomous offline.

Dependency-free on purpose (stdlib urllib only) so it runs unchanged inside the
Jetson inference container without extra installs.
"""
from __future__ import annotations

import base64
import json
import logging
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CloudClient:
    def __init__(
        self,
        base_url: str,
        timeout_s: float = 8.0,
        state_min_interval_s: float = 0.5,
        max_workers: int = 2,
    ):
        self._base = base_url.rstrip("/")
        self._timeout = timeout_s
        self._state_min_interval = state_min_interval_s
        self._last_state_push = 0.0
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="cloud")
        logger.info("CloudClient → %s", self._base)

    # ── public, non-blocking ──────────────────────────────────────────────────

    def push_state(self, state, frame_bgr: Optional[np.ndarray] = None,
                   detections=None, aoi_poly=None) -> None:
        """Throttled live-state push for the dashboard. Drops frames if too soon.

        If `detections`/`aoi_poly` are supplied, thin yellow boxes and the AOI
        outline are drawn on the posted frame (only on frames that actually
        post — negligible cost) so the dashboard visibly signals live detection.
        """
        now = time.monotonic()
        if now - self._last_state_push < self._state_min_interval:
            return
        self._last_state_push = now
        if frame_bgr is not None and (detections or aoi_poly is not None):
            frame_bgr = _draw_boxes(frame_bgr, detections or [], aoi_poly)
        payload = _state_payload(state, frame_bgr)
        self._submit("/api/state", payload)

    def push_event(self, event, frame_bgr: Optional[np.ndarray] = None) -> None:
        """Intervention event → cloud (triggers Qwen incident report if level≥2)."""
        payload = _event_payload(event, frame_bgr)
        self._submit("/api/events", payload)

    def evaluate_policy(
        self,
        rho_summary: dict,
        event_counts: dict,
        current_params: dict,
        context: str,
        on_patch: Callable[[dict], None],
    ) -> None:
        """Ask cloud Policy Manager for an STL patch; apply via on_patch callback."""
        payload = {
            "rho_summary": rho_summary,
            "event_counts": event_counts,
            "current_params": current_params,
            "context": context,
        }

        def _task():
            resp = self._post("/api/policy/evaluate", payload)
            if not resp:
                return
            patch = resp.get("patch") or {}
            if patch:
                try:
                    on_patch(patch)
                    logger.info("STL params hot-swapped from cloud policy: %s", patch)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("apply patch failed: %s", exc)

        self._pool.submit(_task)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ── internals ─────────────────────────────────────────────────────────────

    def _submit(self, path: str, payload: dict) -> None:
        self._pool.submit(self._post, path, payload)

    def _post(self, path: str, payload: dict) -> Optional[dict]:
        url = self._base + path
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as r:
                body = r.read()
                return json.loads(body) if body else {}
        except urllib.error.URLError as exc:
            logger.warning("cloud POST %s failed: %s", path, getattr(exc, "reason", exc))
        except Exception as exc:  # noqa: BLE001
            logger.warning("cloud POST %s error: %s", path, exc)
        return None


# ── payload builders (match backend/models.py) ────────────────────────────────

def _draw_boxes(frame_bgr: np.ndarray, detections, aoi_poly=None) -> np.ndarray:
    """Thin yellow boxes + track labels + AOI outline — a lightweight 'live' cue."""
    img = frame_bgr.copy()
    YELLOW = (0, 255, 255)
    if aoi_poly is not None:
        try:
            cv2.polylines(img, [aoi_poly.astype(np.int32)], True, (0, 200, 200), 1, cv2.LINE_AA)
        except Exception:  # noqa: BLE001
            pass
    for d in detections:
        try:
            x1, y1, x2, y2 = (int(v) for v in d.bbox_xyxy)
        except Exception:  # noqa: BLE001
            continue
        cv2.rectangle(img, (x1, y1), (x2, y2), YELLOW, 1)
        label = getattr(d, "label", "")
        tid = getattr(d, "track_id", "")
        txt = f"{label}" + (f"#{tid}" if tid != "" else "")
        if txt:
            cv2.putText(img, txt, (x1, max(y1 - 3, 10)),
                        cv2.FONT_HERSHEY_PLAIN, 0.8, YELLOW, 1, cv2.LINE_AA)
    return img


def _encode_frame(frame_bgr: Optional[np.ndarray], max_dim: int = 640,
                  quality: int = 70) -> Optional[str]:
    if frame_bgr is None:
        return None
    h, w = frame_bgr.shape[:2]
    if max(h, w) > max_dim:
        s = max_dim / max(h, w)
        frame_bgr = cv2.resize(frame_bgr, (int(w * s), int(h * s)))
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode() if ok else None


def _state_payload(state, frame_bgr) -> dict:
    return {
        "t": state.t,
        "timestamp": state.timestamp,
        "level": state.intervention_level,
        "level_label": {0: "SAFE", 1: "AWARENESS", 2: "WARNING",
                        3: "EMERGENCY"}.get(state.intervention_level, "?"),
        "rho": {
            "rho1": _round(state.rho1),
            "rho2": _round(state.rho2),
            "rho3": _round(state.rho3),
            "rho4": _round(getattr(state, "rho4", None)),
            "rho5": _round(getattr(state, "rho5", None)),
        },
        "signals": {
            "d_min": round(state.signals.d_min, 2),
            "v_veh_max": round(state.signals.v_veh_max, 2),
            "d_pred": round(state.signals.d_pred, 2),
        },
        "scale_factor": round(getattr(state, "scale_factor", 1.0), 3),
        "frame_jpeg_b64": _encode_frame(frame_bgr),
    }


def _event_payload(event, frame_bgr) -> dict:
    return {
        "timestamp": event.timestamp,
        "level": event.new_level,
        "d_min": event.d_min,
        "v_veh_max": event.v_veh_max,
        "d_pred": event.d_pred,
        "rho_min": event.rho_min,
        "message": event.message,
        "frame_jpeg_b64": _encode_frame(frame_bgr),
    }


def _round(x, n: int = 3):
    return round(x, n) if x is not None else None
