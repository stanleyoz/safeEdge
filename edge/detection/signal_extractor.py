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
    pedestrians: list[TrackedObject]
    vehicles: list[TrackedObject]


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
    ):
        self._H = homography
        self._vel_est = VelocityEstimator(history_frames, fps)
        self._dbg_n = 0
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

            obj = TrackedObject(track_id=det.track_id, label=det.label, xyz=xyz, vel=vel)
            if det.label in PERSON_LABELS:
                pedestrians.append(obj)
            elif det.label in VEHICLE_LABELS:
                vehicles.append(obj)

        self._vel_est.prune(active_ids)

        d_min     = _min_ped_veh_distance(pedestrians, vehicles)
        v_veh_max = max((float(np.linalg.norm(v.vel)) for v in vehicles), default=0.0)

        # Throttled depth-probe diagnostic (~1/s at 15fps) — shows which objects
        # get a valid metric position vs are dropped (e.g. beyond D455 range).
        self._dbg_n += 1
        if dbg and self._dbg_n % 15 == 0:
            logger.info("depth-probe [%s]: %s  →  d_min=%.1f v_veh_max=%.2f",
                        "depth" if use_depth else "homog", " | ".join(dbg), d_min, v_veh_max)

        return SignalBundle(
            d_min=d_min,
            v_veh_max=v_veh_max,
            pedestrians=pedestrians,
            vehicles=vehicles,
        )

    # ── Private ──────────────────────────────────────────────────────────────

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


def _min_ped_veh_distance(
    pedestrians: list[TrackedObject],
    vehicles: list[TrackedObject],
) -> float:
    if not pedestrians or not vehicles:
        return _SENTINEL_DISTANCE

    ped_pts = np.array([p.xyz[[0, 2]] for p in pedestrians])
    veh_pts = np.array([v.xyz[[0, 2]] for v in vehicles])

    # Brute-force pairwise — N is tiny (< 20 objects per frame)
    diffs = ped_pts[:, None, :] - veh_pts[None, :, :]   # (P, V, 2)
    dists = np.linalg.norm(diffs, axis=-1)               # (P, V)
    return float(dists.min())


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
