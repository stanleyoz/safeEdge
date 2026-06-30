"""Replay a sample car-park clip to the SafeEdge cloud backend.

Reads real frames from an existing MP4 and drives a scripted pedestrian-vehicle
near-miss: posts /api/state (with the real frame + evolving STL robustness ρ and
signals) and fires /api/events at WARNING/EMERGENCY thresholds. The cloud backend
then generates Qwen incident reports and the operator dashboard populates live.

This exercises the full  clip → backend → Qwen → Tablestore → dashboard  path
without needing the Jetson/GPU. (On the real edge, detections drive these signals;
here they are scripted so the backend + dashboard can be demonstrated anywhere.)

Usage:
  python tools/replay_to_cloud.py \
      --url https://<fc-url> \
      --video data/captures/carpark_daylight_01.mp4
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import time
import urllib.request

import cv2
import numpy as np


def post(url: str, path: str, payload: dict, timeout: float = 40.0):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url.rstrip("/") + path, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def encode(frame, max_dim=640, q=70):
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        s = max_dim / max(h, w)
        frame = cv2.resize(frame, (int(w * s), int(h * s)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf).decode()


def level_for(d_min: float) -> int:
    if d_min < 1.0:  return 3      # EMERGENCY
    if d_min < 2.0:  return 2      # WARNING
    if d_min < 3.5:  return 1      # AWARENESS
    return 0                       # SAFE


LABELS = {0: "SAFE", 1: "AWARENESS", 2: "WARNING", 3: "EMERGENCY"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Cloud backend base URL")
    ap.add_argument("--video", required=True, help="Sample MP4 to source frames from")
    ap.add_argument("--steps", type=int, default=40, help="Number of state posts")
    ap.add_argument("--interval", type=float, default=1.0, help="Seconds between posts")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    print(f"Source: {args.video} ({total} frames) → {args.url}")

    last_level = 0
    for i in range(args.steps):
        # scripted near-miss: d_min dips to ~0.7m at the midpoint, then recovers
        phase = i / max(args.steps - 1, 1)
        d_min = 0.7 + 5.3 * abs(math.sin(math.pi * phase + 0.15)) * (1 - 0.6 * math.exp(-((phase-0.5)**2)/0.02))
        d_min = max(0.6, min(6.0, d_min))
        v_veh = round(1.5 + 2.8 * (1 - phase) , 2)
        d_pred = round(max(0.0, d_min - 0.6 * v_veh), 2)
        lvl = level_for(d_min)

        # simple robustness signals (ρ>0 safe, ρ<0 violated) — illustrative
        rho = {
            "rho1": round(d_min - 1.0, 3),
            "rho2": round((d_min - 1.0) - 0.3 * v_veh, 3),
            "rho3": round(d_pred - 1.5, 3),
            "rho4": round(1.0 - max(0.0, v_veh - 0.1) if d_min < 1.0 else 2.0, 3),
            "rho5": None,
        }

        # pull a real frame from the clip
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(phase * (total - 1)))
        ok, frame = cap.read()
        frame_b64 = encode(frame) if ok else None

        state = {
            "t": i, "timestamp": time.time(), "level": lvl, "level_label": LABELS[lvl],
            "rho": rho,
            "signals": {"d_min": round(d_min, 2), "v_veh_max": v_veh, "d_pred": d_pred},
            "scale_factor": 1.0, "frame_jpeg_b64": frame_b64,
        }
        try:
            post(args.url, "/api/state", state)
        except Exception as exc:  # noqa: BLE001
            print(f"  state post failed: {exc}")

        # fire an event when severity escalates to WARNING/EMERGENCY
        if lvl >= 2 and lvl > last_level:
            msg = {2: "WARNING: pedestrian within proximity zone of moving vehicle",
                   3: "EMERGENCY: predicted collision — pedestrian in vehicle path"}[lvl]
            ev = {"timestamp": time.time(), "level": lvl, "d_min": round(d_min, 2),
                  "v_veh_max": v_veh, "d_pred": d_pred, "rho_min": min(v for v in rho.values() if v is not None),
                  "message": msg, "frame_jpeg_b64": frame_b64}
            try:
                r = post(args.url, "/api/events", ev)
                print(f"  [{LABELS[lvl]}] event fired (d_min={d_min:.2f}) → incident_id={r.get('incident_id')}")
            except Exception as exc:  # noqa: BLE001
                print(f"  event post failed: {exc}")
        last_level = lvl

        print(f"step {i+1}/{args.steps}  level={LABELS[lvl]:<10} d_min={d_min:.2f}m v={v_veh}m/s")
        time.sleep(args.interval)

    cap.release()
    print("\nReplay complete — open the dashboard to view the operator console.")


if __name__ == "__main__":
    main()
