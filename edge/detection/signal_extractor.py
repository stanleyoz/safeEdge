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

import numpy as np
from dataclasses import dataclass

from edge.safety.trajectory import TrackedObject, VelocityEstimator


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
    homography: 3×3 np.float64 mapping image coords → ground plane metres.
    Load from config/homography.npy, or use IdentityHomography for mock mode.
    """

    def __init__(
        self,
        homography: np.ndarray,
        history_frames: int = 6,
        fps: float = 15.0,
    ):
        self._H = homography
        self._vel_est = VelocityEstimator(history_frames, fps)

    def extract(self, detections: list[RawDetection]) -> SignalBundle:
        pedestrians: list[TrackedObject] = []
        vehicles: list[TrackedObject] = []
        active_ids: set[int] = set()

        for det in detections:
            ground_xz = self._foot_to_ground(det.bbox_xyxy)
            if ground_xz is None:
                continue

            # Store as [X, 0, Z] so trajectory module (uses xyz[[0,2]]) works unchanged
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

        d_min    = _min_ped_veh_distance(pedestrians, vehicles)
        v_veh_max = max((float(np.linalg.norm(v.vel)) for v in vehicles), default=0.0)

        return SignalBundle(
            d_min=d_min,
            v_veh_max=v_veh_max,
            pedestrians=pedestrians,
            vehicles=vehicles,
        )

    # ── Private ──────────────────────────────────────────────────────────────

    def _foot_to_ground(self, bbox: np.ndarray) -> np.ndarray | None:
        """Project bbox bottom-centre (foot of object) through homography → [X, Z] metres."""
        x1, y1, x2, y2 = bbox.astype(float)
        foot_u = (x1 + x2) / 2.0
        foot_v = float(y2)           # bottom edge = ground contact point

        pt = self._H @ np.array([foot_u, foot_v, 1.0])
        if abs(pt[2]) < 1e-6:
            return None
        return (pt[:2] / pt[2]).astype(np.float32)


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
