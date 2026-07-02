"""Scripted scenario driven through the REAL SafeEdge pipeline.

Simulates a pedestrian walking diagonally across the road while a car pulls out
of a parking spot and accelerates toward them. Detections are synthetic, but
everything downstream is the real production code:

  synthetic detections (world→image via the calibrated homography)
    → SignalExtractor (homography positions + VelocityEstimator + occupant filter)
    → predict_min_distance (φ3)
    → STLMonitor (motion gate + distance bands)
    → InterventionEngine (hysteresis)
    → CloudClient → Function Compute → Qwen-VL incident report → dashboard

So it validates the danger LOGIC (and animates the dashboard) with no actors.

Usage:
  python tools/simulate_scenario.py                 # local only, prints the arc
  python tools/simulate_scenario.py --cloud         # also post to the cloud/dashboard
  python tools/simulate_scenario.py --cloud --loop  # repeat continuously
NOTE: with --cloud, stop the Jetson live pipeline first (they'd both post).
"""
from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv("backend/.env")

from edge.detection.signal_extractor import SignalExtractor, RawDetection
from edge.safety.stl_monitor import STLMonitor, SignalFrame
from edge.safety.trajectory import predict_min_distance
from edge.safety.intervention import InterventionEngine, LEVEL_LABELS, LEVEL_COLOURS
from edge.cloud_client import CloudClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FC_URL = "https://safeedg-backend-nkmqevdhff.ap-southeast-1.fcapp.run"
FPS = 15
DT = 1.0 / FPS
DUR = 15.0                      # seconds
FRAME_W, FRAME_H = 1280, 720

H = np.load(os.path.join(ROOT, "config", "homography.npy"))
H_INV = np.linalg.inv(H)


def world_to_img(X: float, Z: float) -> tuple[float, float]:
    p = H_INV @ np.array([X, Z, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def smoothstep(s: float) -> float:
    s = max(0.0, min(1.0, s))
    return s * s * (3 - 2 * s)


# ── scripted world trajectories (metres; X across, Z away from camera) ─────────

def ped_world(t: float) -> tuple[float, float]:
    """Diagonal crossing: far-left → near-right over the whole clip."""
    s = t / DUR
    return (-6.0 + 12.0 * s, 27.0 - 8.0 * s)


def car_world(t: float) -> tuple[float, float]:
    """Parked, then pulls out and drives at a STEADY speed toward the pedestrian.

    Constant-velocity approach (linear, not a smoothstep pulse) so v_veh reads a
    sustained ~2.5 m/s for several seconds — clearly 'moving' on the dashboard —
    then the car stops near the pedestrian (v→0) to show 'stopped = calm'.
    """
    t0, t1 = 3.0, 8.0                      # steady approach window (5 s)
    start = np.array([10.0, 25.0])         # parking spot (right)
    end   = np.array([-2.0, 22.0])         # drives across to the left ~2.5 m/s
    if t < t0:
        return tuple(start)
    if t > t1:
        return tuple(end)                  # stopped near/past the pedestrian
    frac = (t - t0) / (t1 - t0)            # linear → constant velocity
    p = start + (end - start) * frac
    return float(p[0]), float(p[1])


def bbox_from_foot(u: float, v: float, real_h_m: float, Z: float,
                   aspect: float) -> np.ndarray:
    """Plausible bbox with its bottom-centre at the projected foot point.
    Apparent pixel height shrinks with distance."""
    h = float(np.clip(1500.0 * real_h_m / max(Z, 1.0), 18, 260))
    w = h * aspect
    return np.array([u - w / 2, v - h, u + w / 2, v], dtype=np.float32)


def render(frame, person_box, car_box, bundle, state, t):
    cv2.rectangle(frame, (0, 0), (FRAME_W, FRAME_H), (35, 35, 40), -1)
    # crude road wedge for context
    cv2.fillPoly(frame, [np.array([[540, 400], [740, 400], [1180, 700], [100, 700]])],
                 (55, 55, 62))
    x1, y1, x2, y2 = car_box.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 30, 30), -1)        # black sedan
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
    cv2.putText(frame, "sedan", (x1, y1 - 4), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 255), 1)
    px1, py1, px2, py2 = person_box.astype(int)
    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 220, 60), 2)     # pedestrian
    cv2.putText(frame, "person", (px1, py1 - 4), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 220, 60), 1)
    # distance line
    pc = ((px1 + px2) // 2, py2); cc = ((x1 + x2) // 2, y2)
    lvl = state.intervention_level
    col = tuple(int(LEVEL_COLOURS[lvl].lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
    cv2.line(frame, pc, cc, col, 2)
    cv2.rectangle(frame, (0, 0), (FRAME_W, 30), col, -1)
    cv2.putText(frame, f"SIM  {LEVEL_LABELS[lvl]}  t={t:4.1f}s  d_min={bundle.d_min:.1f}m  "
                       f"v_veh={bundle.v_veh_max:.1f}m/s", (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", action="store_true", help="post to Function Compute / dashboard")
    ap.add_argument("--loop", action="store_true", help="repeat continuously")
    ap.add_argument("--cycles", type=int, default=1, help="number of scenario passes then stop")
    args = ap.parse_args()

    extractor = SignalExtractor(H, fps=FPS)               # homography mode (no intrinsics)
    stl = STLMonitor(os.path.join(ROOT, "config", "stl_specs.yaml"))
    engine = InterventionEngine(upgrade_hold=3, downgrade_hold=15,
                                on_event=lambda e: print(f"  >>> {LEVEL_LABELS[e.new_level]}: {e.message}"))
    # Post ~every 0.4 s (not every frame) so sustained moving frames actually
    # land on the dashboard rather than being coalesced away.
    cloud = CloudClient(base_url=FC_URL, state_min_interval_s=0.4) if args.cloud else None
    if cloud:
        print(f"posting to {FC_URL}  (dashboard will animate)")

    total = 10**9 if args.loop else max(1, args.cycles)
    for cycle in range(total):
        for i in range(int(DUR * FPS)):
            t = i * DT
            pX, pZ = ped_world(t); cX, cZ = car_world(t)
            pu, pv = world_to_img(pX, pZ); cu, cv = world_to_img(cX, cZ)
            person = RawDetection(1, "person", bbox_from_foot(pu, pv, 1.7, pZ, 0.35), 0.92)
            car    = RawDetection(2, "car",    bbox_from_foot(cu, cv, 1.5, cZ, 2.2), 0.90)

            bundle = extractor.extract([person, car])
            d_pred = predict_min_distance(bundle.pedestrians, bundle.vehicles, horizon_s=4.0)
            sf = SignalFrame(t=i, d_min=bundle.d_min, v_veh_max=bundle.v_veh_max,
                             d_pred=d_pred, alert_active=engine.alert_active)
            state = stl.update(sf)
            event = engine.process(state)

            frame = render(np.zeros((FRAME_H, FRAME_W, 3), np.uint8),
                           person.bbox_xyxy, car.bbox_xyxy, bundle, state, t)
            if cloud:
                cloud.push_state(state, frame)
                if event:
                    cloud.push_event(event, frame)

            print(f"t={t:4.1f}  d_min={bundle.d_min:5.1f}  v_veh={bundle.v_veh_max:4.2f}  "
                  f"level={LEVEL_LABELS[state.intervention_level]}")
            time.sleep(DT)
        print(f"--- cycle {cycle + 1}/{'∞' if args.loop else total} complete ---")

    if cloud:
        cloud.close()
    print("done")


if __name__ == "__main__":
    main()
