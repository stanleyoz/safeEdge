"""
FastAPI dashboard backend.
Serves the static single-page UI and streams live safety state over WebSocket.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from edge.safety.stl_monitor import SafetyState
from edge.safety.intervention import InterventionEvent, LEVEL_LABELS, LEVEL_COLOURS

logger = logging.getLogger(__name__)
app = FastAPI(title="SafeEdge Dashboard")

STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

_connections: list[WebSocket] = []
_latest_state: Optional[dict] = None


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC / "index.html").read_text()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _connections.append(ws)
    if _latest_state:
        await ws.send_text(json.dumps(_latest_state))
    try:
        while True:
            await ws.receive_text()   # keep-alive ping handling
    except WebSocketDisconnect:
        _connections.remove(ws)


async def _broadcast(payload: dict) -> None:
    global _latest_state
    _latest_state = payload
    dead = []
    for ws in _connections:
        try:
            await ws.send_text(json.dumps(payload))
        except Exception:
            dead.append(ws)
    for ws in dead:
        _connections.remove(ws)


def publish(state: SafetyState, frame_bgr: np.ndarray, event: Optional[InterventionEvent]) -> None:
    """Called from the edge loop each frame."""
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
    img_b64 = base64.b64encode(buf).decode()

    payload = {
        "t": state.t,
        "timestamp": state.timestamp,
        "level": state.intervention_level,
        "level_label": LEVEL_LABELS[state.intervention_level],
        "level_colour": LEVEL_COLOURS[state.intervention_level],
        "rho": {
            "rho1": round(state.rho1, 3),
            "rho2": round(state.rho2, 3),
            "rho3": round(state.rho3, 3) if state.rho3 is not None else None,
            "rho4": round(state.rho4, 3) if state.rho4 is not None else None,
            "rho5": round(state.rho5, 3) if state.rho5 is not None else None,
        },
        "signals": {
            "d_min": round(state.signals.d_min, 2),
            "v_veh_max": round(state.signals.v_veh_max, 2),
            "d_pred": round(state.signals.d_pred, 2),
        },
        "scale_factor": round(state.scale_factor, 3),
        "frame": img_b64,
        "event": {
            "message": event.message,
            "level": event.new_level,
        } if event else None,
    }

    asyncio.create_task(_broadcast(payload))


def start(host: str = "0.0.0.0", port: int = 8080) -> None:
    uvicorn.run(app, host=host, port=port, log_level="warning")
