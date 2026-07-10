"""SafeEdge cloud backend — FastAPI service (deployed on Alibaba Cloud FC).

Endpoints
  POST /api/state            edge → throttled live state (broadcast + persist)
  POST /api/events           edge → intervention event (persist; vision report if level≥2)
  POST /api/policy/evaluate  edge → Qwen Policy Manager → STL param patch
  GET  /api/incidents        dashboard → recent incident reports
  GET  /api/forecast         dashboard → latest risk forecast
  GET  /api/events           dashboard → recent events
  GET  /healthz              liveness probe (for FC / load balancer)
  WS   /ws                   dashboard → live state stream
  GET  /                     dashboard UI
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

load_dotenv()

from backend.models import (EventPush, PolicyEvalRequest, PolicyEvalResponse,
                            StatePush)
from backend.skills import Skills
from backend.store import get_store

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("safeedge.backend")

# Generate a Qwen-VL incident report only at/above this level. The event LOG
# still captures EVERY event (a cheap store write); gating the multi-second,
# blocking vision call to EMERGENCY keeps the serverless instance responsive
# under dense bursts while still showcasing Qwen-VL on the highest-impact
# events. Default 2 (all WARNING+) preserves prior behaviour; the deployment
# sets INCIDENT_MIN_LEVEL=3 so the dashboard never stalls during the demo.
INCIDENT_MIN_LEVEL = int(os.environ.get("INCIDENT_MIN_LEVEL", "2"))

app = FastAPI(title="SafeEdge Cloud Backend", version="1.0")

# Allow the dashboard to be served from any origin (laptop / OSS / GitHub Pages)
# and still call this API. The FC *.fcapp.run domain forces Content-Disposition:
# attachment on HTML, so the dashboard is typically hosted off-FC and hits this
# API cross-origin. (Tighten allow_origins for production.)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC = Path(__file__).parent.parent / "dashboard" / "static"
if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

_store = get_store()
_skills = Skills(location=os.environ.get("LOCATION_LABEL", "Car Park"))
_connections: list[WebSocket] = []
_latest_state: Optional[dict] = None

FORECAST_INTERVAL_S = int(os.environ.get("FORECAST_INTERVAL_S", "3600"))


# ── dashboard UI + WebSocket ─────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    idx = STATIC / "index.html"
    if idx.exists():
        return idx.read_text()
    return "<h1>SafeEdge Backend</h1><p>Dashboard UI not bundled.</p>"


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _connections.append(ws)
    if _latest_state:
        await ws.send_text(json.dumps(_latest_state))
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        if ws in _connections:
            _connections.remove(ws)


async def _broadcast(payload: dict) -> None:
    global _latest_state
    _latest_state = payload
    dead = []
    for ws in _connections:
        try:
            await ws.send_text(json.dumps(payload))
        except Exception:  # noqa: BLE001
            dead.append(ws)
    for ws in dead:
        if ws in _connections:
            _connections.remove(ws)


# ── edge → cloud ingest ───────────────────────────────────────────────────────

@app.post("/api/state")
async def post_state(state: StatePush):
    payload = state.model_dump()
    # persist a lightweight ρ sample for forecasting / history
    _store.add_rho_sample({
        "timestamp": state.timestamp,
        "level": state.level,
        "rho": state.rho.model_dump(),
        "signals": state.signals.model_dump(),
    })
    # Persist latest state to the shared store so the dashboard (which polls
    # REST on serverless FC, where WebSocket fan-out doesn't work) can read it.
    _store.set_latest_state(payload)
    await _broadcast(payload)
    return {"ok": True}


@app.get("/api/state/latest")
async def get_latest_state():
    return JSONResponse(_store.get_latest_state() or {})


@app.post("/api/events")
async def post_event(event: EventPush):
    ev = event.model_dump()
    frame_b64 = ev.get("frame_jpeg_b64")
    # Keep the raw frame OUT of the event row itself — /api/events?limit=60 is
    # polled every ~1.5s by the dashboard, and 60 rows of embedded JPEGs would
    # bloat that response badly. Store it separately, keyed by event id, and
    # serve it on demand via /api/events/{id}/frame. (ev itself is left intact
    # with frame_jpeg_b64 present, since _generate_incident below still needs
    # it for the Qwen-VL vision call.)
    eid = _store.add_event({**{k: v for k, v in ev.items() if k != "frame_jpeg_b64"},
                            "has_frame": bool(frame_b64)})
    if frame_b64:
        _store.set_kv(f"frame:{eid}", {"jpeg_b64": frame_b64})

    # Vision/text incident report for WARNING+ events (Qwen-VL on Alibaba).
    # Generated INLINE, not as a background task: Function Compute freezes the
    # instance once the response is sent, so fire-and-forget work never finishes.
    # The edge client is non-blocking (thread pool), so a slower response is fine.
    incident_id = None
    if event.level >= INCIDENT_MIN_LEVEL:
        incident_id = await _generate_incident(ev)
    return {"ok": True, "event_id": eid, "incident_id": incident_id}


@app.get("/api/events/{event_id}/frame")
async def get_event_frame(event_id: str):
    rec = _store.get_kv(f"frame:{event_id}")
    if not rec or not rec.get("jpeg_b64"):
        return JSONResponse({"error": "frame not found"}, status_code=404)
    return Response(content=base64.b64decode(rec["jpeg_b64"]), media_type="image/jpeg")


async def _generate_incident(event: dict) -> Optional[str]:
    loop = asyncio.get_event_loop()
    report = await loop.run_in_executor(None, lambda: _skills.incident_report(event))
    if not report:
        return None
    rec = {
        "timestamp": event["timestamp"],
        "level": event["level"],
        "report": report,
        "d_min": event["d_min"],
        "v_veh_max": event["v_veh_max"],
    }
    iid = _store.add_incident(rec)
    logger.info("Incident report stored (level=%d)", event["level"])
    await _broadcast({"type": "incident", "incident": rec})
    return iid


@app.post("/api/policy/evaluate", response_model=PolicyEvalResponse)
async def policy_evaluate(req: PolicyEvalRequest):
    loop = asyncio.get_event_loop()
    patch = await loop.run_in_executor(None, lambda: _skills.policy_eval(req.model_dump()))
    return PolicyEvalResponse(patch=patch)


# ── dashboard reads ───────────────────────────────────────────────────────────

@app.get("/api/incidents")
async def get_incidents(limit: int = 50):
    return JSONResponse(_store.recent_incidents(limit))


@app.get("/api/events")
async def get_events(limit: int = 100):
    return JSONResponse(_store.recent_events(limit))


@app.get("/api/forecast")
async def get_forecast(refresh: bool = False):
    # On Function Compute the instance freezes between requests, so the periodic
    # background forecaster can't run. Generate on demand (cached in the store).
    fc = None if refresh else _store.get_forecast()
    if fc is None:
        events = _store.recent_events(50)
        loop = asyncio.get_event_loop()
        fc = await loop.run_in_executor(None, lambda: _skills.forecast(events))
        fc["generated_at"] = time.time()
        _store.set_forecast(fc)
    return JSONResponse(fc)


@app.get("/healthz")
async def healthz():
    # Function Compute injects these env vars into every instance. Surfacing
    # them makes /healthz self-prove it is running inside Alibaba FC — a simple
    # curl of the public URL is then reproducible proof of deployment.
    fc = {label: os.environ[var] for label, var in {
        "region": "FC_REGION",
        "account_id": "FC_ACCOUNT_ID",
        "function": "FC_FUNCTION_NAME",
        "instance": "FC_INSTANCE_ID",
    }.items() if os.environ.get(var)}
    return {
        "status": "ok",
        "platform": "alibaba-function-compute" if fc else "local",
        "fc": fc,
        "store": type(_store).__name__,
        "models": {
            "reasoning": os.environ.get("QWEN_REASONING_MODEL", "qwen-max"),
            "vision": os.environ.get("QWEN_VISION_MODEL", "qwen-vl-max"),
        },
        "ts": time.time(),
    }


# ── background risk forecaster ─────────────────────────────────────────────────

async def _forecast_loop() -> None:
    while True:
        await asyncio.sleep(FORECAST_INTERVAL_S)
        try:
            events = _store.recent_events(50)
            loop = asyncio.get_event_loop()
            fc = await loop.run_in_executor(None, lambda: _skills.forecast(events))
            fc["generated_at"] = time.time()
            _store.set_forecast(fc)
            logger.info("Risk forecast refreshed (%d windows)",
                        len(fc.get("high_risk_windows", [])))
        except Exception as exc:  # noqa: BLE001
            logger.warning("forecast loop error: %s", exc)


@app.on_event("startup")
async def _startup():
    asyncio.create_task(_forecast_loop())
    logger.info("SafeEdge backend ready — store=%s", type(_store).__name__)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
