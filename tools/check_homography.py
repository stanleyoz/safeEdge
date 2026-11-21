"""
Fit and check the image -> ground homography from landmarks on a scaled site map.

Input is the JSON that docs/tools/map-calibrator.html exports (or one written by
hand):

  {"frame_size": [1280, 720],
   "points": [{"name": "lamp post base", "img": [821, 509], "world": [12.4, 30.1]}, ...],
   "lines":  [{"name": "path far edge", "world": [[x1, z1], [x2, z2]],
               "img": [[u, v], [u, v], ...]}, ...],
   "camera_height_m": 4.5}                  # optional: the mount height, if known

"world" is metres on the map (any origin and orientation: STL only uses
distances and speeds). A line pairs a straight edge drawn on the map (two map
points) with any number of image points clicked along the same edge; it
constrains the fit without needing matching endpoints, which is what long
kerbs and path edges are good for.

  python tools/check_homography.py calib.json --frame frame.png \\
      --trace research/vjepa-surprise/results/traces/segment_0455_to_end.trace.jsonl \\
      --old config/homography.npy --out calib_check/

The fit starts from the points (direct linear transform, at least 4) and is
refined on points and lines together. The report gives, per landmark, the
ground error in metres and the leave-one-out error (refit without it: a
landmark that only fits when it is in the fit is suspect); the camera model
implied by the fit (focal length, mount height, tilt, horizon row), which must
look physically plausible; and scale checks: side-on parked-car lengths from a
trace, and the lawn depth on the centre column, to compare with the map. It
writes homography.npy as a candidate, never over config/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ── Geometry ────────────────────────────────────────────────────────────────

def apply(H: np.ndarray, pts) -> np.ndarray:
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def dlt(img, world) -> np.ndarray:
    """Normalised direct linear transform, image -> world."""
    img, world = np.asarray(img, float), np.asarray(world, float)

    def norm(p):
        c = p.mean(0)
        s = np.sqrt(2) / max(np.sqrt(((p - c) ** 2).sum(1)).mean(), 1e-9)
        return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])

    Ti, Tw = norm(img), norm(world)
    a, b = apply(Ti, img), apply(Tw, world)
    rows = []
    for (x, y), (u, v) in zip(a, b):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    h = np.linalg.svd(np.asarray(rows))[2][-1].reshape(3, 3)
    H = np.linalg.inv(Tw) @ h @ Ti
    return H / H[2, 2]


def line_residuals(H, lines) -> np.ndarray:
    """Signed ground distance (m) of each projected image point from its map
    line."""
    out = []
    for ln in lines:
        (x1, z1), (x2, z2) = ln["world"]
        d = np.array([x2 - x1, z2 - z1], float)
        n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        out.extend((apply(H, ln["img"]) - [x1, z1]) @ n)
    return np.asarray(out)


def residuals(H, points, lines) -> np.ndarray:
    r = []
    if points:
        r.append((apply(H, [p["img"] for p in points]) - [p["world"] for p in points]).ravel())
    if lines:
        r.append(line_residuals(H, lines))
    return np.concatenate(r) if r else np.zeros(0)


def refine(H0, points, lines, iters: int = 50) -> np.ndarray:
    """Levenberg-Marquardt on the 8 free entries, minimising ground-plane
    error over points and lines together."""
    h = (H0 / H0[2, 2]).ravel()[:8].copy()
    to_H = lambda v: np.append(v, 1.0).reshape(3, 3)
    lam, r = 1e-3, residuals(to_H(h), points, lines)
    for _ in range(iters):
        J = np.empty((len(r), 8))
        for k in range(8):
            dh = np.zeros(8); dh[k] = 1e-6 * max(1.0, abs(h[k]))
            J[:, k] = (residuals(to_H(h + dh), points, lines) - r) / dh[k]
        A, g = J.T @ J, J.T @ r
        while True:
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            r_new = residuals(to_H(h + step), points, lines)
            if r_new @ r_new < r @ r:
                h, r, lam = h + step, r_new, lam / 3
                break
            lam *= 4
            if lam > 1e8:
                return to_H(h)
        if np.abs(step).max() < 1e-10:
            break
    return to_H(h)


def fit(points, lines) -> np.ndarray:
    if len(points) < 4:
        raise SystemExit(f"need at least 4 point landmarks to start the fit (have {len(points)})")
    return refine(dlt([p["img"] for p in points], [p["world"] for p in points]), points, lines)


# ── Physical camera fit ─────────────────────────────────────────────────────
# A free homography has 8 degrees of freedom; a pinhole camera with square
# pixels and the principal point at the image centre has 7 (focal length,
# orientation, position). When landmarks are few or lie along one or two
# lines (kerbs), the free fit bends to absorb bad pairs; the camera fit cannot,
# so bad pairs stand out and the result is physical by construction. Residuals
# are in image pixels, where click noise lives.

def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def rodrigues(r) -> np.ndarray:
    th = float(np.linalg.norm(r))
    if th < 1e-12:
        return np.eye(3) + _skew(r)
    K = _skew(np.asarray(r) / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def rotation_log(R) -> np.ndarray:
    th = float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v / 2 if th < 1e-9 else v * th / (2 * np.sin(th))


def camera_homography(f, rvec, C, frame_size) -> np.ndarray:
    """World (X, Z, 1) on the ground -> image, for a camera at C = (X, Z,
    height) with rotation rodrigues(rvec) (world -> camera)."""
    w, h = frame_size
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    R = rodrigues(rvec)
    G = K @ np.c_[R[:, 0], R[:, 1], -R @ np.asarray(C, float)]
    return G / G[2, 2]


def decompose(H, frame_size):
    """(f, R, C) of the camera closest to an image -> ground homography."""
    w, h = frame_size
    cx, cy = w / 2, h / 2
    G = np.linalg.inv(H)
    cam = camera_from_homography(H, frame_size)
    if "focal_px" not in cam:
        raise SystemExit("the homography fit is not a physical camera; cannot start the camera fit")
    f = cam["focal_px"]
    Kinv = np.array([[1 / f, 0, -cx / f], [0, 1 / f, -cy / f], [0, 0, 1]])
    m1, m2, t = Kinv @ G[:, 0], Kinv @ G[:, 1], Kinv @ G[:, 2]
    lam = 2.0 / (np.linalg.norm(m1) + np.linalg.norm(m2))
    r1, r2, t = m1 * lam, m2 * lam, t * lam
    if t[2] < 0:
        r1, r2, t = -r1, -r2, -t
    U, _, Vt = np.linalg.svd(np.c_[r1, r2, np.cross(r1, r2)])
    R = U @ Vt
    if np.linalg.det(R) < 0:                         # map frame is mirrored: flip its second axis
        raise SystemExit("map axes are mirrored relative to the camera; flip the map's y axis")
    return f, R, -R.T @ t


def image_residuals(G, points, lines) -> np.ndarray:
    """Pixels: landmark reprojection errors, and each clicked edge point's
    distance from its map line projected into the image."""
    r = []
    if points:
        r.append((apply(G, [p["world"] for p in points]) - [p["img"] for p in points]).ravel())
    for ln in lines:
        a, b = apply(G, ln["world"])
        d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        n = np.array([-d[1], d[0]])
        r.append((np.asarray(ln["img"], float) - a) @ n)
    return np.concatenate(r) if r else np.zeros(0)


def lm(fun, x0, iters: int = 100) -> np.ndarray:
    x = np.asarray(x0, float).copy()
    lam, r = 1e-3, fun(x)
    for _ in range(iters):
        J = np.empty((len(r), len(x)))
        for k in range(len(x)):
            dx = np.zeros(len(x)); dx[k] = 1e-6 * max(1.0, abs(x[k]))
            J[:, k] = (fun(x + dx) - r) / dx[k]
        A, g = J.T @ J, J.T @ r
        while True:
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            r_new = fun(x + step)
            if np.isfinite(r_new).all() and r_new @ r_new < r @ r:
                x, r, lam = x + step, r_new, lam / 3
                break
            lam *= 4
            if lam > 1e8:
                return x
        if np.abs(step).max() < 1e-10:
            break
    return x


def fit_camera(points, lines, frame_size, height_prior: tuple[float, float] | None = None) -> np.ndarray:
    """Image -> ground homography of the best pinhole camera. height_prior =
    (metres, sd) adds a soft constraint on the mount height, weighted so one
    sd counts like a 3 px click error."""
    f, R, C = decompose(fit(points, lines), frame_size)

    def fun(x):
        G = camera_homography(x[0], x[1:4], x[4:7], frame_size)
        r = image_residuals(G, points, lines)
        if height_prior:
            r = np.append(r, 3.0 * (x[6] - height_prior[0]) / height_prior[1])
        return r

    x = lm(fun, [f, *rotation_log(R), *C])
    H = np.linalg.inv(camera_homography(x[0], x[1:4], x[4:7], frame_size))
    return H / H[2, 2]


def camera_from_homography(H, frame_size) -> dict:
    """Camera implied by an image -> ground homography, assuming square pixels
    and the principal point at the image centre (true enough for a webcam).
    Returns focal length (px), horizontal field of view, mount height above
    the ground plane (m), downward tilt (deg) and the horizon row."""
    w, h = frame_size
    cx, cy = w / 2, h / 2
    G = np.linalg.inv(H)                            # world (X, Z, 1) -> image
    g1, g2 = G[:, 0], G[:, 1]
    a = lambda g: np.array([g[0] - cx * g[2], g[1] - cy * g[2], g[2]])
    p, q = a(g1), a(g2)
    cands = []
    if abs(p[2] * q[2]) > 1e-12:                    # r1 . r2 = 0
        f2 = -(p[0] * q[0] + p[1] * q[1]) / (p[2] * q[2])
        if f2 > 0:
            cands.append(f2)
    if abs(q[2] ** 2 - p[2] ** 2) > 1e-12:          # |r1| = |r2|
        f2 = (p[0] ** 2 + p[1] ** 2 - q[0] ** 2 - q[1] ** 2) / (q[2] ** 2 - p[2] ** 2)
        if f2 > 0:
            cands.append(f2)
    horizon = H[2]                                  # image points mapping to infinity
    horizon_row = float(-(horizon[0] * cx + horizon[2]) / horizon[1]) if abs(horizon[1]) > 1e-12 else None
    out = {"horizon_row_at_centre": horizon_row}
    if not cands:
        out["note"] = "no consistent focal length: the fit is not a physical camera"
        return out
    f = float(np.sqrt(np.mean(cands)))
    Kinv = np.array([[1 / f, 0, -cx / f], [0, 1 / f, -cy / f], [0, 0, 1]])
    m1, m2, t = Kinv @ g1, Kinv @ g2, Kinv @ G[:, 2]
    lam = 2.0 / (np.linalg.norm(m1) + np.linalg.norm(m2))
    r1, r2, t = m1 * lam, m2 * lam, t * lam
    if t[2] < 0:                                    # ground in front of the camera
        r1, r2, t = -r1, -r2, -t
    r3 = np.cross(r1, r2)
    R = np.c_[r1, r2, r3]
    C = -R.T @ t                                    # camera centre in (X, Z, normal)
    axis = R.T @ np.array([0.0, 0.0, 1.0])          # optical axis in world
    tilt = float(np.degrees(np.arcsin(min(1.0, abs(axis[2])))))
    out.update({"focal_px": round(f, 1), "hfov_deg": round(float(np.degrees(2 * np.arctan(w / 2 / f))), 1),
                "height_m": round(float(abs(C[2])), 2), "tilt_down_deg": round(tilt, 1),
                "camera_ground_xz": [round(float(C[0]), 2), round(float(C[1]), 2)],
                "focal_consistency": round(float(np.ptp(np.sqrt(cands)) / f), 3) if len(cands) == 2 else None})
    return out


# ── Checks ──────────────────────────────────────────────────────────────────

def leave_one_out(points, lines, fitter=None) -> list[float | None]:
    fitter = fitter or fit
    errs = []
    for i in range(len(points)):
        rest = points[:i] + points[i + 1:]
        if len(rest) < 4:
            errs.append(None)
            continue
        H = fitter(rest, lines)
        errs.append(float(np.linalg.norm(apply(H, [points[i]["img"]])[0] - points[i]["world"])))
    return errs


def car_lengths(H, trace: Path, min_bottom_px: float, min_width_px: float) -> dict:
    """Footprint length of large, near-row parked cars (side-on in this view):
    bottom-left to bottom-right of the box on the ground plane. A side-on car
    is about 4.2-4.9 m; boxes include some perspective, so expect a little
    over for angled cars."""
    per_track = {}
    for line in open(trace):
        if not line.strip():
            continue
        for o in json.loads(line).get("objects") or []:
            x0, y0, x1, y1 = o["box"]
            if o["label"] == "car" and y1 >= min_bottom_px and x1 - x0 >= min_width_px:
                a, b = apply(H, [(x0, y1), (x1, y1)])
                per_track.setdefault(o["id"], []).append(float(np.linalg.norm(a - b)))
    med = [float(np.median(v)) for v in per_track.values() if len(v) >= 10]
    return {"n_tracks": len(med), "median_m": round(float(np.median(med)), 2) if med else None,
            "per_track_m": sorted(round(m, 2) for m in med)}


def column_depths(H, u: float, rows) -> list[dict]:
    pts = apply(H, [(u, v) for v in rows])
    return [{"row": int(v), "dist_from_first_m": round(float(np.linalg.norm(p - pts[0])), 2)}
            for v, p in zip(rows, pts)]


# ── Figures ─────────────────────────────────────────────────────────────────

def draw_overlay(path: Path, frame, H, points, lines, cam: dict, step_m: float = 1.0):
    import cv2
    im = frame.copy()
    h, w = im.shape[:2]
    Ginv = np.linalg.inv(H)
    hz = cam.get("horizon_row_at_centre")
    top = int(max(0, (hz or 0) + 60))         # grid lines crowd into a band near the horizon
    # ground grid: project dense samples of each world grid line back into the image
    wb = apply(H, [(u, v) for u in np.linspace(0, w - 1, 25) for v in np.linspace(top, h - 1, 25)])
    lo, hi = np.floor(wb.min(0) / step_m) * step_m, np.ceil(wb.max(0) / step_m) * step_m
    for axis in (0, 1):
        for c in np.arange(lo[axis], hi[axis] + 1e-9, step_m):
            s = np.linspace(lo[1 - axis], hi[1 - axis], 400)
            world = np.c_[np.full_like(s, c), s] if axis == 0 else np.c_[s, np.full_like(s, c)]
            img = apply(Ginv, world)
            ok = (img[:, 1] >= top) & (img[:, 1] < h) & (img[:, 0] >= 0) & (img[:, 0] < w)
            bold = abs(c / (5 * step_m) - round(c / (5 * step_m))) < 1e-6
            for a, b, m in zip(img[:-1], img[1:], ok[:-1] & ok[1:]):
                if m:
                    cv2.line(im, tuple(map(int, a)), tuple(map(int, b)), (0, 220, 255) if bold else (0, 140, 160), 1)
    for ln in lines:
        img = apply(Ginv, np.linspace(ln["world"][0], ln["world"][1], 200))
        for a, b in zip(img[:-1], img[1:]):
            if 0 <= a[1] < h and 0 <= b[1] < h:
                cv2.line(im, tuple(map(int, a)), tuple(map(int, b)), (255, 120, 0), 2)
        for u, v in ln["img"]:
            cv2.circle(im, (int(u), int(v)), 3, (255, 120, 0), -1)
    for p in points:
        u, v = map(int, p["img"])
        pu, pv = apply(Ginv, [p["world"]])[0]
        cv2.circle(im, (u, v), 5, (0, 0, 255), 2)
        cv2.line(im, (u, v), (int(pu), int(pv)), (0, 0, 255), 2)
        cv2.putText(im, p["name"], (u + 7, v - 7), 0, 0.5, (255, 255, 255), 2)
    if hz is not None and 0 <= hz < h:
        cv2.line(im, (0, int(hz)), (w, int(hz)), (200, 0, 200), 1)
        cv2.putText(im, "horizon (fit)", (8, int(hz) - 6), 0, 0.5, (200, 0, 200), 1)
    cv2.putText(im, f"grid {step_m:g} m (bright every {5 * step_m:g} m); red: landmark -> where the fit puts it",
                (8, h - 12), 0, 0.55, (255, 255, 255), 2)
    cv2.imwrite(str(path), im)


def draw_topdown(path: Path, frame, H, points, horizon_row: float | None,
                 px_per_m: float = 20.0, max_range_m: float = 60.0):
    import cv2
    h, w = frame.shape[:2]
    # only rows well below the horizon: above it the plane maps behind the camera
    top = max(0.0, (horizon_row if horizon_row is not None else 0.0) + 25)
    cam_rows = np.linspace(top, h - 1, 30)
    wb = apply(H, [(u, v) for u in np.linspace(0, w - 1, 30) for v in cam_rows])
    wb = wb[np.isfinite(wb).all(1)]
    centre = np.median(wb, 0)
    wb = wb[np.linalg.norm(wb - centre, axis=1) < max_range_m]
    lo, hi = wb.min(0) - 2, wb.max(0) + 2
    S = np.array([[px_per_m, 0, -lo[0] * px_per_m], [0, -px_per_m, hi[1] * px_per_m], [0, 0, 1]])
    size = (int((hi[0] - lo[0]) * px_per_m), int((hi[1] - lo[1]) * px_per_m))
    masked = frame.copy()
    masked[: int(top)] = 0
    top_img = cv2.warpPerspective(masked, S @ H, size)
    for p in points:
        x, y = apply(S, [p["world"]])[0]
        cv2.circle(top_img, (int(x), int(y)), 5, (0, 0, 255), 2)
        cv2.putText(top_img, p["name"], (int(x) + 6, int(y) - 6), 0, 0.45, (255, 255, 255), 1)
    cv2.putText(top_img, f"top-down, {px_per_m:g} px per metre: kerbs straight, cars and bays right-angled and to scale"
                "; upright objects smear away from the camera", (8, 20), 0, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(path), top_img)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("calib", type=Path, help="landmark JSON (map-calibrator export)")
    ap.add_argument("--frame", type=Path, default=None, help="the camera frame the landmarks were clicked on")
    ap.add_argument("--trace", type=Path, default=None, help="replay trace for the car-length check")
    ap.add_argument("--old", type=Path, default=None, help="current homography.npy, to compare scale")
    ap.add_argument("--car-min-bottom", type=float, default=465, help="px; near-row cars only")
    ap.add_argument("--car-min-width", type=float, default=200, help="px; full, unoccluded boxes")
    ap.add_argument("--column", type=float, default=640, help="image column for the depth profile")
    ap.add_argument("--rows", type=float, nargs="+", default=[505, 560, 615, 690],
                    help="rows for the depth profile (default: kerb, mid-lawn, path far edge, path near edge)")
    ap.add_argument("--camera", action="store_true",
                    help="fit a pinhole camera (7 parameters) instead of a free homography (8): "
                         "physical by construction, and robust when landmarks lie along few lines")
    ap.add_argument("--height-prior-sd", type=float, default=None,
                    help="with --camera, softly constrain the mount height to camera_height_m with this sd (m)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cal = json.loads(args.calib.read_text())
    points, lines = cal.get("points", []), cal.get("lines", [])
    frame_size = cal.get("frame_size", [1280, 720])
    if args.camera:
        prior = ((cal["camera_height_m"], args.height_prior_sd)
                 if args.height_prior_sd and cal.get("camera_height_m") else None)
        fitter = lambda pts, lns: fit_camera(pts, lns, frame_size, prior)
    else:
        fitter = fit
    H = fitter(points, lines)
    cam = camera_from_homography(H, frame_size)
    pt_err = (np.linalg.norm(apply(H, [p["img"] for p in points]) - [p["world"] for p in points], axis=1)
              if points else np.zeros(0))
    loo = leave_one_out(points, lines, fitter)
    G = np.linalg.inv(H)
    px_err = (np.linalg.norm(apply(G, [p["world"] for p in points]) - [p["img"] for p in points], axis=1)
              if points else np.zeros(0))
    m_per_px = [float(np.linalg.norm(apply(H, [(p["img"][0], p["img"][1] + 1)])[0] - apply(H, [p["img"]])[0]))
                for p in points]
    ln_err = [float(np.sqrt(np.mean(line_residuals(H, [ln]) ** 2))) for ln in lines]

    report = {
        "model": "camera" if args.camera else "homography",
        "n_points": len(points), "n_lines": len(lines),
        "points": [{"name": p["name"], "err_m": round(float(e), 3), "err_px": round(float(ep), 1),
                    "m_per_px": round(mp, 3), "leave_one_out_m": None if l is None else round(l, 3)}
                   for p, e, ep, mp, l in zip(points, pt_err, px_err, m_per_px, loo)],
        "lines": [{"name": ln["name"], "rms_m": round(e, 3), "n_img_points": len(ln["img"])}
                  for ln, e in zip(lines, ln_err)],
        "rms_m": round(float(np.sqrt(np.mean(residuals(H, points, lines) ** 2))), 3),
        "camera": cam,
        "depth_profile": column_depths(H, args.column, args.rows),
    }
    if cal.get("camera_height_m") and "height_m" in cam:
        report["camera"]["height_vs_known_m"] = round(cam["height_m"] - cal["camera_height_m"], 2)
    if args.trace:
        report["car_lengths"] = car_lengths(H, args.trace, args.car_min_bottom, args.car_min_width)
    if args.old:
        H_old = np.load(args.old)
        report["old"] = {"camera": camera_from_homography(H_old, frame_size),
                         "depth_profile": column_depths(H_old, args.column, args.rows)}
        if args.trace:
            report["old"]["car_lengths"] = car_lengths(H_old, args.trace, args.car_min_bottom, args.car_min_width)
        new_d = report["depth_profile"][-1]["dist_from_first_m"]
        old_d = report["old"]["depth_profile"][-1]["dist_from_first_m"]
        report["old"]["depth_scale_new_over_old"] = round(new_d / old_d, 2) if old_d else None

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "homography.npy", H)
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    if args.frame:
        import cv2
        frame = cv2.imread(str(args.frame))
        draw_overlay(args.out / "overlay.png", frame, H, points, lines, cam)
        draw_topdown(args.out / "topdown.png", frame, H, points, cam.get("horizon_row_at_centre"))

    print(f"{report['model']} fit: {len(points)} points, {len(lines)} lines, RMS {report['rms_m']} m"
          f" (image RMS {np.sqrt(np.mean(image_residuals(G, points, lines) ** 2)):.1f} px)")
    for p in report["points"]:
        # a click is good to about 3 px; judge in pixels, since far points carry
        # metres per pixel and their metre errors say little about the fit
        flag = "  <- check this pair" if p["err_px"] > 6 else ""
        print(f"  {p['name']:28s} err {p['err_px']:5.1f} px = {p['err_m']:6.3f} m ({p['m_per_px']:.2f} m/px)"
              f"   leave-one-out {p['leave_one_out_m']} m{flag}")
    for ln in report["lines"]:
        print(f"  line {ln['name']:23s} rms {ln['rms_m']:6.3f} m over {ln['n_img_points']} points")
    print("camera:", json.dumps(cam))
    print("depth profile (column {:g}):".format(args.column),
          ", ".join(f"row {d['row']}: {d['dist_from_first_m']} m" for d in report["depth_profile"]))
    if "car_lengths" in report:
        print("side-on car lengths:", report["car_lengths"]["median_m"], "m median over",
              report["car_lengths"]["n_tracks"], "tracks (expect about 4.2-4.9)")
    if "old" in report:
        o = report["old"]
        print("old homography: camera", json.dumps(o["camera"]),
              "| car lengths", o.get("car_lengths", {}).get("median_m"),
              "| depth scale new/old", o["depth_scale_new_over_old"])
    print(f"wrote {args.out}/homography.npy (candidate; config/ untouched)")


if __name__ == "__main__":
    main()
