"""Vehicle-motion options of SignalExtractor (the `signals` section of
config/stl_specs.yaml): robust speed, near-pedestrian scope, vehicle-only
closing, and their effect on EMERGENCY timing through the STL monitor."""
from pathlib import Path

import numpy as np
import pytest

from edge.detection.signal_extractor import RawDetection, SignalExtractor, extractor_options
from edge.safety.stl_monitor import SignalFrame, STLMonitor
from edge.safety.trajectory import predict_min_distance

FPS = 7.7
SPEC = Path(__file__).resolve().parents[1] / "config" / "stl_specs.yaml"
# image px -> ground metres: x = 0.02 u - 12.8, z = 36 - 0.05 v
H = np.array([[0.02, 0, -12.8], [0, -0.05, 36.0], [0, 0, 1]])
NEW = dict(speed_window_s=1.0, vehicle_scope="near", closing_mode="vehicle")


def box_at(x, z, w=200, h=80):
    u, v, s = np.linalg.inv(H) @ [x, z, 1]
    u, v = u / s, v / s
    return np.array([u - w / 2, v - h, u + w / 2, v])


def run(objects_at, seconds, opts, seed=0, jitter_px=2.0):
    """objects_at(t) -> [(id, label, x, z, w, h)]; returns per-frame
    (bundle, raw level)."""
    rng = np.random.default_rng(seed)
    ex, stl, out = SignalExtractor(H, fps=FPS, **opts), STLMonitor(SPEC), []
    for i in range(int(seconds * FPS)):
        dets = [RawDetection(k, lab, box_at(x, z, w, h) + rng.normal(0, jitter_px, 4), 1.0)
                for k, lab, x, z, w, h in objects_at(i / FPS)]
        b = ex.extract(dets)
        s = stl.update(SignalFrame(t=i, d_min=b.d_min, v_veh_max=b.v_veh_max,
                                   d_pred=predict_min_distance(b.pedestrians, b.vehicles, 4.0),
                                   alert_active=0.0, v_closing=b.v_closing))
        out.append((b, s.intervention_level))
    return out


def test_options_from_spec():
    assert extractor_options({}) == {}
    assert extractor_options({"signals": {"speed_window_s": 1.0, "closing_mode": "vehicle", "other": 1}}) == \
        {"speed_window_s": 1.0, "closing_mode": "vehicle"}
    with pytest.raises(ValueError):
        SignalExtractor(H, closing_mode="sideways")


def test_id_switch_is_not_speed():
    """A parked car's track id jumps to another car 8 m away: the original
    estimator resyncs and reports motion; the windowed speed restarts."""
    at = lambda t: [(2, "car", 1.0 if t < 3 else 9.0, 20.0, 200, 80)]
    new = run(at, 6, NEW, jitter_px=0.5)
    speeds = [float(np.linalg.norm(b.vehicles[0].vel)) for b, _ in new]
    assert max(speeds) < 0.3


def test_far_jittery_car_does_not_set_vehicle_speed():
    """Pedestrian standing 1 m from a parked car; a far-row car jitters hard.
    'all' takes the far car's phantom speed; 'near' ignores it."""
    def at(t):
        far_x = -8.0 + 0.6 * np.sin(7 * t)                 # +-0.6 m wobble: phantom speed
        return [(1, "person", 0.0, 20.0, 30, 80), (2, "car", 1.0, 20.0, 200, 80), (3, "car", far_x, 30.0, 120, 50)]
    old = run(at, 6, dict(speed_window_s=1.0, vehicle_scope="all"), jitter_px=0.5)
    new = run(at, 6, dict(speed_window_s=1.0, vehicle_scope="near"), jitter_px=0.5)
    assert max(b.v_veh_max for b, _ in old[10:]) > 0.5
    assert max(b.v_veh_max for b, _ in new[10:]) < 0.3


def test_walking_up_to_a_parked_car_is_not_an_emergency():
    """Pedestrian walks up to a parked car and stops 1.0 m from it: mutual
    closing calls it converging (EMERGENCY); vehicle closing does not."""
    at = lambda t: [(1, "person", min(-4.0 + 1.2 * t, 0.2), 20.0, 30, 80), (2, "car", 1.2, 20.0, 200, 80)]
    old = run(at, 6, {})
    new = run(at, 6, NEW)
    assert max(lv for _, lv in old) == 3
    assert max(lv for _, lv in new) <= 2


@pytest.mark.parametrize("speed", [1.0, 2.0, 4.0])
def test_approaching_car_raises_emergency_on_entering_the_band(speed):
    """Car drives at a standing pedestrian: EMERGENCY on the first frame
    inside proximity_emergency (1.8 m), as with the original settings."""
    def at(t):
        x = max(0.0, 8.0 - speed * t) + 1.0
        return [(1, "person", 0.0, 20.0, 30, 80), (2, "car", x, 20.0, 200, 80),
                (3, "car", -8.0, 30.0, 120, 50)]
    frames = run(at, 12, NEW)
    band = next(i for i, (b, _) in enumerate(frames) if b.d_min < 1.8)
    em = next(i for i, (_, lv) in enumerate(frames) if lv == 3)
    assert em <= band


@pytest.mark.parametrize("direction", ["toward", "along"])
def test_pull_out_beside_pedestrian_is_as_fast_as_before(direction):
    """Parked 1.5 m from a pedestrian, the car starts moving at 1 m/s: no
    EMERGENCY while parked, and EMERGENCY no later than with the original
    settings (about 0.5 s after it starts)."""
    def at(t):
        dt = max(0.0, t - 5.0)
        x, z = (1.5 - min(dt, 1.2), 20.0) if direction == "toward" else (1.5, 20.0 + dt)
        return [(1, "person", 0.0, 20.0, 30, 80), (2, "car", x, z, 200, 80)]

    def first_em(opts):
        frames = run(at, 10, opts, seed=1)
        assert all(lv < 3 for _, lv in frames[:int(5 * FPS)])     # parked: no EMERGENCY
        return next(i for i, (_, lv) in enumerate(frames) if lv == 3)

    assert first_em(NEW) <= first_em({})
    assert (first_em(NEW) - 5 * FPS) / FPS < 0.8
