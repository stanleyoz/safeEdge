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
NEW = dict(speed_window_s=1.0, vehicle_scope="near", closing_mode="vehicle",
           vehicle_motion="rigid", motion_deadband_px=2.0, moving_speed=0.3)


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
                                   alert_active=0.0, v_closing=b.v_closing,
                                   d_moving=b.d_moving, ttc=b.ttc))
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
    EMERGENCY while parked, and EMERGENCY within a frame of the original
    settings (about 0.5 s after it starts; the 2 px motion deadband costs
    one frame when the car moves along the pedestrian)."""
    def at(t):
        dt = max(0.0, t - 5.0)
        x, z = (1.5 - min(dt, 1.2), 20.0) if direction == "toward" else (1.5, 20.0 + dt)
        return [(1, "person", 0.0, 20.0, 30, 80), (2, "car", x, z, 200, 80)]

    def first_em(opts):
        frames = run(at, 10, opts, seed=1)
        assert all(lv < 3 for _, lv in frames[:int(5 * FPS)])     # parked: no EMERGENCY
        return next(i for i, (_, lv) in enumerate(frames) if lv == 3)

    assert first_em(NEW) <= first_em({}) + 1
    assert (first_em(NEW) - 5 * FPS) / FPS < 0.8


def run_boxes(boxes_at, seconds, opts, frame_size=(720, 1280)):
    """boxes_at(t) -> [(id, label, box)]; returns per-frame bundles."""
    ex, out = SignalExtractor(H, fps=FPS, **opts), []
    for i in range(int(seconds * FPS)):
        dets = [RawDetection(k, lab, np.asarray(b, float), 1.0) for k, lab, b in boxes_at(i / FPS)]
        out.append(ex.extract(dets, frame_size=frame_size))
    return out


def test_pedestrian_in_front_of_a_parked_car_is_not_motion():
    """A pedestrian walking past a parked car hides, then reveals, its left
    end: the box's left edge moves 60 px and the centre-based speed reads it
    as motion; the rigid speed (both edges must agree) does not."""
    def at(t):
        left = 640 + 60 * min(1.0, max(0.0, t - 2.0))      # occlusion grows over 1 s
        return [(1, "person", [600 + 40 * t, 420, 630 + 40 * t, 520]), (2, "car", [left, 440, 840, 520])]
    centre = run_boxes(at, 4, dict(speed_window_s=1.0, vehicle_scope="near"))
    rigid = run_boxes(at, 4, dict(speed_window_s=1.0, vehicle_scope="near", vehicle_motion="rigid",
                                  motion_deadband_px=2.0))
    assert max(b.v_veh_max for b in centre) > 0.4
    assert max(b.v_veh_max for b in rigid) < 0.05


def test_car_entering_at_the_frame_edge_is_motion():
    """A car driving in from the left border: its left edge is clipped at 0,
    so only the right edge counts and the car still reads as moving."""
    at = lambda t: [(1, "person", [300, 420, 330, 520]), (2, "car", [0, 440, 50 + 50 * t, 520])]
    rigid = run_boxes(at, 3, dict(speed_window_s=1.0, vehicle_scope="near", vehicle_motion="rigid",
                                  motion_deadband_px=2.0))
    assert np.median([b.v_veh_max for b in rigid[10:]]) == pytest.approx(1.0, abs=0.15)  # 50 px/s x 0.02 m/px


def test_parked_cars_are_awareness_moving_cars_warning():
    """Pedestrian standing 3.5 m from a car's body (WARNING band):
    AWARENESS while the car is parked, WARNING once it drives past at 1 m/s
    (its 4 m box spans x 3.5-7.5 m). Parked, its centre is 3.5 m away."""
    parked = run(lambda t: [(1, "person", 0.0, 20.0, 30, 80), (2, "car", 3.5, 20.0, 200, 80)], 6, NEW)
    moving = run(lambda t: [(1, "person", 0.0, 20.0, 30, 80), (2, "car", 5.5, 18.0 + 1.0 * t, 200, 80)], 4, NEW)
    assert max(lv for _, lv in parked) == 1
    assert max(lv for _, lv in moving[int(FPS):]) == 2


def test_pedestrian_walking_into_a_moving_cars_path_is_an_emergency():
    """July clip, 33 s and 72 s: a car drives along the lane at 1 m/s while
    a pedestrian walks towards its path, and a parked car is the nearest
    vehicle. To the moving car's body the pair closes inside 3 s:
    EMERGENCY more than 2 s before the nearest-pair rule's 1.8 m."""
    def at(t):
        return [(1, "person", 0.0, min(20.0 + 0.8 * t, 20.6), 30, 80),    # walks towards the lane
                (2, "car", 8.0 - 1.0 * t, 22.5, 200, 60),             # 4 m long, 2.5 m off, driving left
                (3, "car", -2.6, 20.0, 200, 80)]                        # parked, nearest
    first_em = lambda frames: next(i for i, (_, lv) in enumerate(frames) if lv == 3)
    frames = run(at, 8, NEW)
    old = run(at, 8, {k: v for k, v in NEW.items() if k != "moving_speed"})
    em = first_em(frames)
    assert frames[em][0].ttc < 3.0 and frames[em][0].d_moving > 1.8   # on time to collision
    assert (first_em(old) - em) / FPS > 2.0           # the nearest-pair rule waits for 1.8 m


def test_moving_car_passing_far_from_pedestrian_is_not_an_emergency():
    """A car drives past 6 m away: no EMERGENCY, whatever its speed."""
    at = lambda t: [(1, "person", 0.0, 20.0, 30, 80), (2, "car", 10.0 - 2.0 * t, 26.0, 200, 60)]
    assert max(lv for _, lv in run(at, 10, NEW)) <= 2
