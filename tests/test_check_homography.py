"""Tests for tools/check_homography.py against a synthetic camera with known
focal length, mount height and tilt."""
import numpy as np
import pytest

from tools.check_homography import (
    apply, camera_from_homography, dlt, fit, leave_one_out, line_residuals,
)

W, H_PX = 1280, 720


def camera(f=900.0, height=4.0, tilt_deg=12.0, yaw_deg=20.0, at=(3.0, -5.0)):
    """World -> image homography of a pinhole camera at `height` metres above
    the ground plane (X, Z, 0), looking along `yaw` and tilted down."""
    K = np.array([[f, 0, W / 2], [0, f, H_PX / 2], [0, 0, 1]])
    t_, y_ = np.radians(tilt_deg), np.radians(yaw_deg)
    fwd = np.array([np.sin(y_) * np.cos(t_), np.cos(y_) * np.cos(t_), -np.sin(t_)])
    right = np.array([np.cos(y_), -np.sin(y_), 0.0])
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd])                 # world -> camera rows
    C = np.array([at[0], at[1], height])
    t = -R @ C
    G = K @ np.c_[R[:, 0], R[:, 1], t]                # ground (X, Z, 1) -> image
    return G / G[2, 2]


@pytest.fixture
def scene():
    G = camera()
    world = np.array([[x, z] for x in (-2, 4, 10) for z in (8, 14, 22)], float)
    img = apply(G, world)
    points = [{"name": f"p{i}", "img": list(u), "world": list(w)} for i, (u, w) in enumerate(zip(img, world))]
    edge = np.array([[-6.0, 10.0], [16.0, 10.0]])
    along = apply(G, np.linspace(edge[0] + [2, 0], edge[1] - [2, 0], 6))
    lines = [{"name": "path edge", "world": edge.tolist(), "img": along.tolist()}]
    return G, points, lines


def test_dlt_recovers_exact_homography(scene):
    G, points, _ = scene
    H = dlt([p["img"] for p in points], [p["world"] for p in points])
    Hi = np.linalg.inv(G); Hi /= Hi[2, 2]
    assert np.allclose(H, Hi, rtol=1e-6, atol=1e-9)


def test_camera_model_recovered(scene):
    G, points, lines = scene
    cam = camera_from_homography(fit(points, lines), (W, H_PX))
    assert cam["focal_px"] == pytest.approx(900, rel=1e-3)
    assert cam["height_m"] == pytest.approx(4.0, abs=0.01)
    assert cam["tilt_down_deg"] == pytest.approx(12.0, abs=0.05)
    assert cam["camera_ground_xz"] == pytest.approx([3.0, -5.0], abs=0.02)
    # horizon of a 12 degree down-tilt sits f * tan(12) above the centre row
    assert cam["horizon_row_at_centre"] == pytest.approx(H_PX / 2 - 900 * np.tan(np.radians(12)), abs=0.5)


def test_lines_constrain_and_noise_is_averaged(scene):
    G, points, lines = scene
    rng = np.random.default_rng(0)
    noisy = [dict(p, img=list(np.asarray(p["img"]) + rng.normal(0, 1.5, 2))) for p in points]
    spread = [noisy[i] for i in (0, 2, 4, 6, 8)]       # corners and centre of the 3 x 3 layout
    H = fit(spread, lines)                             # few points + a long line
    probe = np.array([[0.0, 12.0], [8.0, 18.0]])       # inside the landmarks' extent
    err = np.linalg.norm(apply(H, apply(G, probe)) - probe, axis=1)
    assert err.max() < 0.35
    assert np.abs(line_residuals(H, lines)).max() < 0.2


def test_leave_one_out_flags_a_bad_landmark(scene):
    _, points, lines = scene
    bad = [dict(p) for p in points]
    bad[4] = dict(bad[4], world=[bad[4]["world"][0] + 3.0, bad[4]["world"][1]])   # 3 m map mistake
    loo = leave_one_out(bad, lines)
    assert loo[4] == max(loo) and loo[4] > 2.0


def test_needs_four_points(scene):
    _, points, lines = scene
    with pytest.raises(SystemExit):
        fit(points[:3], lines)


def test_camera_fit_on_two_kerb_lines():
    """Landmarks along only two lines (like kerbs), with 1.5 px click noise:
    over 20 draws the 7-parameter camera fit recovers the camera and maps the
    ground several times more accurately than the free homography, which
    bends to fit the noise."""
    from tools.check_homography import fit_camera
    G = camera(f=850.0, height=2.0, tilt_deg=4.0, yaw_deg=-10.0, at=(0.0, 0.0))
    world = np.array([[x, 8.0] for x in (-6, -2, 2, 6)] + [[3 + 0.3 * z, z] for z in (10, 16, 24)], float)
    near = np.array([[-4.0, 5.0], [4.0, 5.0], [0.0, 12.0]])            # lawn and off-line ground
    ec, eh, heights = [], [], []
    for seed in range(20):
        img = apply(G, world) + np.random.default_rng(seed).normal(0, 1.5, (len(world), 2))
        points = [{"name": f"p{i}", "img": list(u), "world": list(w)} for i, (u, w) in enumerate(zip(img, world))]
        Hc, Hh = fit_camera(points, [], (W, H_PX)), fit(points, [])
        cam = camera_from_homography(Hc, (W, H_PX))
        assert cam["focal_consistency"] < 0.01                     # physical by construction
        heights.append(cam["height_m"])
        ec.append(np.linalg.norm(apply(Hc, apply(G, near)) - near, axis=1))
        eh.append(np.linalg.norm(apply(Hh, apply(G, near)) - near, axis=1))
    assert np.median(heights) == pytest.approx(2.0, abs=0.1)
    assert np.median(ec) < 0.1
    assert np.median(ec) * 3 < np.median(eh)


def test_rotation_log_inverts_rodrigues():
    from tools.check_homography import rodrigues, rotation_log
    for r in ([0.1, -0.2, 0.3], [1.2, 0.4, -0.7], [0.0, 0.0, 1e-8]):
        assert np.allclose(rotation_log(rodrigues(r)), r, atol=1e-7)
