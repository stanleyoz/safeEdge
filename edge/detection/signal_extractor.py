"""
Converts tracked 2-D detections into the scalar signals the STL monitor needs.

Uses a homography matrix (image pixels → ground plane metres) computed once
at calibration time.  No depth sensor required.

Output each frame:
  d_min     — minimum pedestrian-vehicle distance across all pairs (m)
  v_veh_max — maximum speed of any vehicle in the zone (m/s)
  + TrackedObject list for the trajectory predictor
"""
from __future__ import annotations

import logging
import os
from collections import deque
import numpy as np
from dataclasses import dataclass

from edge.safety.trajectory import TrackedObject, VelocityEstimator

logger = logging.getLogger(__name__)


VEHICLE_LABELS = {"car", "truck", "bus", "motorcycle"}
PERSON_LABELS  = {"person"}

_SENTINEL_DISTANCE = 100.0    # returned when no ped-vehicle pair is present


@dataclass
class RawDetection:
    track_id: int
    label: str
    bbox_xyxy: np.ndarray   # [x1, y1, x2, y2] pixels
    confidence: float


@dataclass
class SignalBundle:
    d_min: float
    v_veh_max: float
    v_closing: float                  # mutual closing speed of nearest pair (m/s, +ve = converging), median-smoothed
    pedestrians: list[TrackedObject]
    vehicles: list[TrackedObject]


def extractor_options(stl_cfg: dict) -> dict:
    """SignalExtractor keyword arguments from the `signals` section of
    stl_specs.yaml; an absent section keeps the original behaviour."""
    sig = (stl_cfg or {}).get("signals") or {}
    keys = ("speed_window_s", "max_speed", "vehicle_scope", "near_radius_m", "closing_mode")
    return {k: sig[k] for k in keys if k in sig}


class SignalExtractor:
    """
    Converts 2-D detections → metric (X, Z) ground positions.

    Two positioning modes selected automatically:
      depth mode   — RealSenseSource provides depth_m; uses back-projection
                     with camera intrinsics (fx, fy, cx, cy). No homography.
      homography   — webcam or mock; projects foot point through H matrix.

    Pass intrinsics=(fx, fy, cx, cy) when constructing for depth mode.
    """

    def __init__(
        self,
        homography: np.ndarray,
        history_frames: int = 6,
        fps: float = 15.0,
        intrinsics: tuple[float, float, float, float] | None = None,
        vclose_smooth: int | None = None,
        speed_window_s: float | None = None,
        max_speed: float = 6.0,
        vehicle_scope: str = "all",
        near_radius_m: float = 5.0,
        closing_mode: str = "mutual",
    ):
        """speed_window_s, vehicle_scope and closing_mode select how vehicle
        motion is measured (the defaults reproduce the original behaviour):

          speed_window_s  None: vehicle velocity from the smoothed per-frame
                          estimator. A number: net displacement over that
                          window within the unbroken part of the track (a
                          step faster than max_speed m/s is a track-ID switch
                          and ends it); unknown (zero) until half a window of
                          clean track exists. Parked-car box jitter averages
                          out over ~2 s, a moving vehicle's displacement adds
                          up.
          vehicle_scope   "all": v_veh_max over every vehicle in view.
                          "near": only vehicles within near_radius_m of a
                          pedestrian, so a jittering car in the far row cannot
                          speak for the car next to the pedestrian.
          closing_mode    "mutual": v_closing is the pair's mutual closing
                          speed, which includes the pedestrian's own walking.
                          "vehicle": the vehicle's velocity towards the
                          pedestrian only, so walking up to a parked car does
                          not count as converging.
        """
        self._H = homography
        self._vel_est = VelocityEstimator(history_frames, fps)
        self._fps = fps
        self._speed_k = None if speed_window_s is None else max(2, int(round(speed_window_s * fps)))
        self._max_speed = max_speed
        self._pos_hist: dict[int, dict[int, np.ndarray]] = {}
        self._frame = 0
        if vehicle_scope not in ("all", "near") or closing_mode not in ("mutual", "vehicle"):
            raise ValueError(f"vehicle_scope {vehicle_scope!r} / closing_mode {closing_mode!r}")
        self._vehicle_scope = vehicle_scope
        self._near_radius = near_radius_m
        self._closing_mode = closing_mode
        self._dbg_n = 0
        # Rolling-median filter on v_closing. Homography position jitter can spike
        # the nearest pair's radial velocity for a single frame on an essentially
        # static car — enough to trip the EMERGENCY convergence gate falsely. A
        # genuine approach keeps a majority of frames converging, so the median
        # stays above the gate while isolated spikes are rejected. Window in
        # frames; env-overridable for offline tuning (SAFEEDGE_VCLOSE_SMOOTH).
        n = vclose_smooth if vclose_smooth is not None else \
            int(os.environ.get("SAFEEDGE_VCLOSE_SMOOTH", "5"))
        self._vclose_smooth = max(1, n)
        self._vclose_hist: deque[float] = deque(maxlen=self._vclose_smooth)
        # Unpack (fx, fy, cx, cy) for depth back-projection
        if intrinsics is not None:
            self._fx, self._fy, self._cx, self._cy = intrinsics
        else:
            self._fx = None

    def extract(
        self,
        detections: list[RawDetection],
        depth_m: np.ndarray | None = None,
    ) -> SignalBundle:
        pedestrians: list[TrackedObject] = []
        vehicles: list[TrackedObject] = []
        active_ids: set[int] = set()

        use_depth = depth_m is not None and self._fx is not None
        dbg: list[str] = []

        # NOTE: no occupant/driver filtering. At operational (car-park) resolution
        # a driver behind glass is not reliably detected, so every `person` is a
        # real pedestrian. The motion gate (parked vs moving vehicle) is the true
        # discriminator; an occupant filter only risks suppressing a genuine
        # pedestrian at the closest-approach danger moment.

        for det in detections:
            z_dbg = None
            if use_depth:
                ground_xz, z_dbg = self._foot_to_ground_depth(det.bbox_xyxy, depth_m)
            else:
                ground_xz = self._foot_to_ground_homography(det.bbox_xyxy)
            # diagnostic: record every detection's resolved depth + kept/dropped
            zt = "None" if z_dbg is None else f"{z_dbg:.1f}m"
            dbg.append(f"{det.label}#{det.track_id} z={zt} {'OK' if ground_xz is not None else 'DROP'}")
            if ground_xz is None:
                continue

            xyz = np.array([ground_xz[0], 0.0, ground_xz[1]], dtype=np.float32)

            active_ids.add(det.track_id)
            self._vel_est.update(det.track_id, xyz)
            vel = self._vel_est.get_velocity(det.track_id)
            if self._speed_k is not None and det.label in VEHICLE_LABELS:
                vel = self._net_velocity(det.track_id, xyz)

            obj = TrackedObject(track_id=det.track_id, label=det.label, xyz=xyz, vel=vel)
            if det.label in PERSON_LABELS:
                pedestrians.append(obj)
            elif det.label in VEHICLE_LABELS:
                vehicles.append(obj)

        self._vel_est.prune(active_ids)
        for k in [k for k in self._pos_hist if k not in active_ids]:
            del self._pos_hist[k]
        self._frame += 1

        d_min, v_pair_raw, veh_closing_raw = _nearest_pair_closing(pedestrians, vehicles)
        self._vclose_hist.append(v_pair_raw if self._closing_mode == "mutual" else veh_closing_raw)
        # The EMERGENCY gate keys on v_closing, median-smoothed to trim jitter
        # flaps. closing_mode "mutual" (original) is the pair's mutual closing,
        # which also fires when a pedestrian walks up to a parked car; it was
        # chosen because the July homography under-measured the car's own
        # radial approach (veh_closing peaked ~0.2 on a real approach). With a
        # calibrated homography, "vehicle" keys on the car's own approach, so a
        # pedestrian walking up to a parked car stays a WARNING.
        v_closing = float(np.median(self._vclose_hist))
        scoped = vehicles
        if self._vehicle_scope == "near" and pedestrians:
            ped_pts = np.array([p.xyz[[0, 2]] for p in pedestrians])
            scoped = [v for v in vehicles
                      if np.linalg.norm(ped_pts - v.xyz[[0, 2]], axis=1).min() < self._near_radius]
        v_veh_max = max((float(np.linalg.norm(v.vel)) for v in scoped), default=0.0)

        # Throttled depth-probe diagnostic (~1/s at 15fps) — shows which objects
        # get a valid metric position vs are dropped (e.g. beyond D455 range).
        self._dbg_n += 1
        if dbg and self._dbg_n % 15 == 0:
            logger.info("depth-probe [%s]: %s  →  d_min=%.1f v_veh_max=%.2f "
                        "closing=%.2f(pair_raw %.2f veh_raw %.2f)",
                        "depth" if use_depth else "homog", " | ".join(dbg),
                        d_min, v_veh_max, v_closing, v_pair_raw, veh_closing_raw)

        return SignalBundle(
            d_min=d_min,
            v_veh_max=v_veh_max,
            v_closing=v_closing,
            pedestrians=pedestrians,
            vehicles=vehicles,
        )

    # ── Private ──────────────────────────────────────────────────────────────

    def _net_velocity(self, track_id: int, xyz: np.ndarray) -> np.ndarray:
        """Velocity from net displacement over the last speed window, within
        the unbroken part of the track; zero until half a window exists."""
        hist = self._pos_hist.setdefault(track_id, {})
        f = self._frame
        hist[f] = xyz
        for old in [g for g in hist if g < f - self._speed_k]:
            del hist[old]
        g = f
        for b in range(f - 1, f - self._speed_k - 1, -1):
            if b not in hist:
                continue
            if np.linalg.norm((hist[g] - hist[b])[[0, 2]]) > self._max_speed * (g - b) / self._fps:
                for old in [h for h in hist if h <= b]:
                    del hist[old]                     # ID switch: the track restarts here
                break
            g = b
        if f - g < max(1, self._speed_k // 2):
            return np.zeros(3, dtype=np.float32)
        return ((xyz - hist[g]) / ((f - g) / self._fps)).astype(np.float32)

    def _foot_to_ground_homography(self, bbox: np.ndarray) -> np.ndarray | None:
        """Project bbox bottom-centre through homography → [X, Z] metres."""
        x1, y1, x2, y2 = bbox.astype(float)
        foot_u = (x1 + x2) / 2.0
        foot_v = float(y2)

        pt = self._H @ np.array([foot_u, foot_v, 1.0])
        if abs(pt[2]) < 1e-6:
            return None
        return (pt[:2] / pt[2]).astype(np.float32)

    def _foot_to_ground_depth(
        self, bbox: np.ndarray, depth_m: np.ndarray
    ) -> tuple[np.ndarray | None, float | None]:
        """Back-project foot point using metric depth + intrinsics → ([X, Z], z_raw).

        Returns (None, z) when z is out of the reliable range (z still returned
        for diagnostics), or (None, None) when no valid depth pixels are found.
        """
        x1, y1, x2, y2 = bbox.astype(int)
        foot_u = int((x1 + x2) / 2)
        foot_v = min(int(y2), depth_m.shape[0] - 1)
        foot_u = max(0, min(foot_u, depth_m.shape[1] - 1))

        # Median over a small patch — robust to edge noise
        v1 = max(0, foot_v - 3);  v2 = min(depth_m.shape[0], foot_v + 4)
        u1 = max(0, foot_u - 3);  u2 = min(depth_m.shape[1], foot_u + 4)
        patch = depth_m[v1:v2, u1:u2]
        valid = patch[patch > 0.1]
        if len(valid) == 0:
            return None, None
        z = float(np.median(valid))
        if not (0.3 < z < 15.0):   # D455 reliable range
            return None, z

        x = (foot_u - self._cx) * z / self._fx
        return np.array([x, z], dtype=np.float32), z


def _nearest_pair_closing(
    pedestrians: list[TrackedObject],
    vehicles: list[TrackedObject],
) -> tuple[float, float, float]:
    """Return (d_min, v_closing, veh_closing) for the closest ped-vehicle pair.

    Let n̂ be the unit vector from the vehicle toward the pedestrian.
      v_closing  = -(Δpos · Δvel)/|Δpos|  — MUTUAL radial closing of the pair;
                   positive when going towards each other. NOTE this includes the
                   pedestrian's OWN motion, so a pedestrian walking toward a
                   PARKED car reads as strongly "closing" though nothing dangerous
                   is happening.
      veh_closing = veh.vel · n̂           — the VEHICLE's velocity component
                   toward the pedestrian alone. Positive only when the CAR is
                   driving at the pedestrian; ~0 for a parked car regardless of
                   what the pedestrian does; negative when the car pulls away.

    veh_closing is the EMERGENCY discriminator (a car bearing down on a person),
    matching operator ground-truth that "pedestrian approaching a parked car" is
    a WARNING, not an emergency. v_closing is retained for diagnostics.
    """
    if not pedestrians or not vehicles:
        return _SENTINEL_DISTANCE, 0.0, 0.0

    ped_pts = np.array([p.xyz[[0, 2]] for p in pedestrians])
    veh_pts = np.array([v.xyz[[0, 2]] for v in vehicles])

    # Brute-force pairwise — N is tiny (< 20 objects per frame)
    diffs = ped_pts[:, None, :] - veh_pts[None, :, :]   # (P, V, 2)
    dists = np.linalg.norm(diffs, axis=-1)               # (P, V)
    pi, vi = np.unravel_index(int(dists.argmin()), dists.shape)
    d_min = float(dists[pi, vi])

    ped, veh = pedestrians[pi], vehicles[vi]
    rel_pos = (ped.xyz - veh.xyz)[[0, 2]]     # vehicle → pedestrian
    rel_vel = (ped.vel - veh.vel)[[0, 2]]
    veh_vel = veh.vel[[0, 2]]
    n = float(np.linalg.norm(rel_pos))
    if n < 1e-6:
        return d_min, 0.0, 0.0
    nhat = rel_pos / n
    v_closing   = float(-(rel_pos @ rel_vel) / n)   # mutual (diagnostic)
    veh_closing = float(veh_vel @ nhat)             # car toward ped (gate)
    return d_min, v_closing, veh_closing


def mock_homography(world_w: float = 12.0, world_h: float = 10.0,
                    img_w: int = 640, img_h: int = 480) -> np.ndarray:
    """
    Identity-style homography for the MockCamera.
    Maps the full image to a world_w × world_h metre ground plane.
    """
    src = np.float32([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]])
    dst = np.float32([[-world_w/2, 0], [world_w/2, 0],
                      [world_w/2, world_h], [-world_w/2, world_h]])
    H, _ = cv2.findHomography(src, dst)
    return H


# Lazy import for mock_homography helper
import cv2  # noqa: E402
