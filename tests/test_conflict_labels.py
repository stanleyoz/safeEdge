"""Unit tests for the conflict self-labeller (tools/conflict_labels.py), on
synthetic traces whose conflicts are known in closed form."""
import json

import numpy as np
import pytest

from tools.conflict_labels import (
    GRID, Params, Track, Obs, build_targets, cv_forecast, frame_conflicts, group_events,
    hull, load_tracks, present,
)

FPS = 10.0
CROP = (0, 300, 1280, 600)


def box_at(x, z, w=60, h=40):
    """Ground (x, z) -> a full-frame box: x along the image, z up the image."""
    cx, by = 640 + 50 * x, 560 - 40 * z
    return [cx - w / 2, by - h, cx + w / 2, by]


def write_trace(path, frames):
    """frames: list of per-frame object lists [(id, label, x, z)]."""
    with open(path, "w") as f:
        for i, objs in enumerate(frames):
            rec = {"frame": i, "level": 0, "objects": [
                {"id": oid, "label": lab, "in_aoi": True, "box": box_at(x, z), "xz": [x, z],
                 "speed": 0.0} for oid, lab, x, z in objs]}
            f.write(json.dumps(rec) + "\n")
    return path


def conflicts(path, **kw):
    _, tracks = load_tracks(path, all_persons=False)
    p = Params(FPS, **kw)
    n = max(max(t.obs) for t in tracks.values()) + 1
    cf = {}
    for f in range(n):
        c = frame_conflicts(f, present(f, tracks, "ped"), present(f, tracks, "veh"), tracks, p)
        if c:
            cf[f] = c
    return tracks, p, n, cf


@pytest.fixture
def approach(tmp_path):
    """Pedestrian standing at x=0; a car drives in from x=6 m at 1 m/s and
    stops at x=1. Distance crosses 1.8 m at t = 4.2 s."""
    frames = []
    for i in range(80):
        x = max(1.0, 6.0 - i / FPS)
        frames.append([(1, "person", 0.0, 2.0), (2, "car", x, 2.0)])
    return write_trace(tmp_path / "approach.jsonl", frames)


def test_moving_car_conflict_onset(approach):
    _, p, _, cf = conflicts(approach)
    events = group_events(cf, FPS, merge_gap_s=1.0)
    assert len(events) == 1
    e = events[0]
    assert e["onset_s"] == pytest.approx(4.3, abs=0.11)     # first frame with d < 1.8
    assert e["vehicle_moving"] and e["pairs"] == [[1, 2]]
    assert e["min_d"] == pytest.approx(1.0, abs=1e-6)


def test_conflict_ends_when_car_stops(approach):
    # once stopped, speed decays over the 1 s velocity window and closing goes to 0
    _, _, _, cf = conflicts(approach)
    assert max(cf) < 62


def test_parked_car_and_walking_pedestrian(tmp_path):
    """Pedestrian walks towards a parked car: a conflict under the default
    (closing) rule, none when vehicle motion is required."""
    frames = [[(1, "person", -4.0 + i / FPS, 2.0), (2, "car", 0.0, 2.0)] for i in range(40)]
    path = write_trace(tmp_path / "walk.jsonl", frames)
    _, _, _, cf = conflicts(path)
    assert cf
    _, _, _, cf2 = conflicts(path, require_vehicle_motion=True)
    assert not cf2


def test_id_switch_is_not_motion(tmp_path):
    """A parked car's id jumps 8 m (tracker hands the id to another car) next
    to a standing pedestrian: no vehicle speed right after the jump."""
    frames = []
    for i in range(40):
        x = 0.8 if i < 20 else 8.8
        frames.append([(1, "person", 0.0, 2.0), (2, "car", x, 2.0), (3, "car", 0.8, 2.0)])
    path = write_trace(tmp_path / "switch.jsonl", frames)
    tracks, p, _, cf = conflicts(path, require_vehicle_motion=True)
    assert not cf
    assert tracks[2].velocity(21, p.k, FPS, p.max_speed) is None   # segment too short after the break
    v = tracks[2].velocity(35, p.k, FPS, p.max_speed)
    assert v is not None and np.linalg.norm(v[0]) < 1e-9          # stationary after it


def test_outside_aoi_pedestrian_ignored(tmp_path):
    frames = [[(1, "person", 0.0, 2.0), (2, "car", 6.0 - i / FPS, 2.0)] for i in range(60)]
    path = write_trace(tmp_path / "aoi.jsonl", frames)
    lines = [json.loads(line) for line in open(path)]
    for r in lines:
        r["objects"][0]["in_aoi"] = False
    path.write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    _, _, _, cf = conflicts(path)
    assert not cf


def test_targets_and_onset(approach):
    tracks, p, n, cf = conflicts(approach)
    t_end = np.array([2.0, 3.0, 4.0, 5.0])
    horizons = [1.0, 2.0, 3.0]
    _, target, onset, ongoing, cv = build_targets(t_end, horizons, cf, tracks, n, p, CROP)
    onset_s = min(cf) / FPS                                         # 4.3 s
    for j, h in enumerate(horizons):
        for i, t in enumerate(t_end):
            expect = t < onset_s <= t + h
            assert onset[j, i] == expect, (h, t)
            assert target[j, i].any() == (expect or (t >= onset_s and t + h >= onset_s))
    assert list(ongoing) == [False, False, False, True]
    # the target lies where the pair is in the image
    region = target[2, 1]                                           # t = 3 s, h = 3 s
    rows, cols = np.nonzero(region)
    assert region.shape == (GRID, GRID) and rows.size
    x0, y0, x1, y1 = CROP
    assert cols.min() * (x1 - x0) / GRID <= 640 <= (cols.max() + 1) * (x1 - x0) / GRID


def test_constant_velocity_forecast_leads(approach):
    """At t = 3 s the car is 3 m away closing at 1 m/s: constant-velocity
    extrapolation predicts the conflict within 2 s but not within 1 s."""
    tracks, p, _, _ = conflicts(approach)
    f = 30
    assert not cv_forecast(f, 1.0, tracks, p)
    assert cv_forecast(f, 2.0, tracks, p)


def test_hull_and_merge():
    assert hull(np.array([0, 0, 10, 10]), np.array([5, -5, 20, 8])) == [0, -5, 20, 10]
    c = {"ped": 1, "veh": 2, "d": 1.0, "closing": 1.0, "v_speed": 1.0, "region": [0, 0, 1, 1]}
    ev = group_events({10: [c], 15: [c], 40: [c]}, FPS, merge_gap_s=1.0)
    assert [(e["onset_frame"], e["end_frame"]) for e in ev] == [(10, 15), (40, 40)]


def test_track_velocity_uses_net_displacement():
    t = Track("veh")
    for f in range(11):
        jitter = 0.05 * (-1) ** f                                    # +-5 cm per frame
        t.obs[f] = Obs(np.array([0.1 * f + jitter, 0.0]), np.zeros(4))
    v = t.velocity(10, 10, FPS, 6.0)
    assert v[0][0] == pytest.approx(1.0, abs=0.011)
