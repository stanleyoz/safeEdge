"""
Scenario A — Near Miss.

A pedestrian walks across a parking bay while a vehicle approaches from the
side.  They converge to ~1.2m clearance at t=5s then diverge.

Use this to:
  - Verify all 5 STL specs fire correctly
  - Generate a demo trace without physical camera
  - Validate intervention level hysteresis

Run standalone:  python -m tests.mock_scenarios.scenario_a_near_miss
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from edge.safety.stl_monitor import STLMonitor, SignalFrame
from edge.safety.trajectory import TrackedObject, VelocityEstimator, predict_min_distance
from edge.safety.intervention import InterventionEngine

ROOT     = Path(__file__).parent.parent.parent
STL_PATH = ROOT / "config" / "stl_specs.yaml"
FPS      = 30.0
DURATION = 20.0   # seconds — must be long enough for 15-frame downgrade hold to clear


def generate_trace() -> list[dict]:
    """
    Returns list of per-frame signal dicts.
    Pedestrian: walks left→right at 1.0 m/s starting at x=-6, z=4.
    Vehicle:    enters at t=2s from x=2, z=10 moving toward z=0 at 2.0 m/s.
    """
    frames = []
    n = int(DURATION * FPS)

    for i in range(n):
        t = i / FPS

        ped_x = -6.0 + t * 1.0
        ped_z =  4.0
        ped_xyz = np.array([ped_x, 0.0, ped_z])
        ped_vel = np.array([1.0, 0.0, 0.0])

        if t >= 2.0:
            veh_x =  2.0
            veh_z = 10.0 - (t - 2.0) * 2.0
            veh_xyz = np.array([veh_x, 0.0, max(0.5, veh_z)])
            veh_vel = np.array([0.0, 0.0, -2.0])
        else:
            veh_xyz = np.array([2.0, 0.0, 10.0])
            veh_vel = np.zeros(3)

        ped = TrackedObject(0, "person",  ped_xyz, ped_vel)
        veh = TrackedObject(1, "car",     veh_xyz, veh_vel)

        d_min  = float(np.linalg.norm(ped_xyz[[0,2]] - veh_xyz[[0,2]]))
        v_veh  = float(np.linalg.norm(veh_vel))
        d_pred = predict_min_distance([ped], [veh], horizon_s=4.0)

        frames.append({
            "t": i, "time_s": t,
            "d_min": d_min, "v_veh_max": v_veh, "d_pred": d_pred,
        })

    return frames


def run_scenario():
    monitor  = STLMonitor(STL_PATH)
    engine   = InterventionEngine(upgrade_hold=3, downgrade_hold=15)

    trace = generate_trace()
    events_fired = []

    for row in trace:
        frame = SignalFrame(
            t=row["t"],
            d_min=row["d_min"],
            v_veh_max=row["v_veh_max"],
            d_pred=row["d_pred"],
            alert_active=engine.alert_active,
        )
        state = monitor.update(frame)
        event = engine.process(state)

        if event:
            events_fired.append(event)
            print(
                f"  t={row['time_s']:5.2f}s  L{event.prev_level}→L{event.new_level}"
                f"  ρ1={state.rho1:.2f}  ρ2={state.rho2:.2f}"
                f"  d={row['d_min']:.2f}m  msg={event.message[:60]}"
            )

    print(f"\nScenario A complete — {len(trace)} frames, {len(events_fired)} events fired")

    # Basic assertions
    levels = [e.new_level for e in events_fired]
    assert 2 in levels or 3 in levels, "Expected at least one WARNING/EMERGENCY event"
    assert any(e.new_level == 0 for e in events_fired), "Expected recovery to SAFE"
    print("Assertions passed ✓")


if __name__ == "__main__":
    print("=== Scenario A: Near Miss ===\n")
    run_scenario()
