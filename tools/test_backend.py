"""Local smoke test for the SafeEdge cloud backend.

Exercises every endpoint against an in-process TestClient. If DASHSCOPE_API_KEY
is set (via backend/.env or shell), the Qwen skills run for real and you'll see
a generated incident report + policy patch. Without it, endpoints still return
200 with graceful empty results.

Run:  python tools/test_backend.py
"""
from __future__ import annotations

import os
import time

from dotenv import load_dotenv

load_dotenv("backend/.env")
load_dotenv()  # fall back to root .env

from fastapi.testclient import TestClient
from backend.app import app

c = TestClient(app)
have_key = bool(os.environ.get("DASHSCOPE_API_KEY"))
print(f"DASHSCOPE_API_KEY: {'SET — live Qwen calls' if have_key else 'absent — degraded mode'}\n")

print("healthz       :", c.get("/healthz").json())

state = {"t": 1, "timestamp": time.time(), "level": 0,
         "rho": {"rho1": 2.5, "rho2": 1.1},
         "signals": {"d_min": 4.2, "v_veh_max": 1.3, "d_pred": 3.0}}
print("POST /state   :", c.post("/api/state", json=state).json())

# WARNING event → triggers Qwen incident report (text-only, no frame here)
ev = {"timestamp": time.time(), "level": 2, "d_min": 1.2, "v_veh_max": 3.5,
      "d_pred": 0.4, "rho_min": -0.8,
      "message": "WARNING: pedestrian within 1.2m of moving vehicle"}
print("POST /events  :", c.post("/api/events", json=ev).json())

# Policy eval → Qwen returns an STL param patch
pe = {"rho_summary": {"events_last_window": 8},
      "event_counts": {"warning": 3, "emergency": 1},
      "current_params": {"phi2": {"proximity_zone": 5.0}, "phi4": {"stop_window": 60}},
      "context": "Evening shift, high foot traffic near entrance."}
print("POST /policy  :", c.post("/api/policy/evaluate", json=pe).json())

# allow the async incident task to finish its Qwen call
time.sleep(6 if have_key else 0.3)

incidents = c.get("/api/incidents").json()
print(f"\nincidents stored: {len(incidents)}")
for i in incidents:
    print(f"  [L{i['level']}] {i['report']}")

print("\nSMOKE TEST COMPLETE")
