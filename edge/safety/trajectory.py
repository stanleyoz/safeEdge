"""
Linear trajectory extrapolation for predictive STL.

Takes the current positions and velocity vectors of all tracked objects,
projects them forward T_horizon seconds, and returns the minimum predicted
pedestrian-vehicle distance — fed into φ3 as the d_pred signal.

Linear extrapolation is intentionally simple: low compute overhead, runs in
the actuator path on every frame, still gives the STL monitor 3–5 seconds of
advance warning for the scenarios we care about (parking lot speeds).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Sequence


@dataclass
class TrackedObject:
    track_id: int
    label: str          # "person" | "car" | "truck" | "bus" | "motorcycle"
    xyz: np.ndarray     # 3-D position in camera frame (metres), shape (3,)
    vel: np.ndarray     # estimated velocity vector (m/s), shape (3,)


def predict_min_distance(
    pedestrians: Sequence[TrackedObject],
    vehicles: Sequence[TrackedObject],
    horizon_s: float = 4.0,
    n_steps: int = 20,
) -> float:
    """
    Return the minimum predicted pedestrian-vehicle distance over [0, horizon_s].

    If no pedestrians or no vehicles are present returns a large sentinel value
    (100.0 m) so φ3 robustness stays positive and no predictive alert fires.
    """
    if not pedestrians or not vehicles:
        return 100.0

    dt = horizon_s / n_steps
    min_dist = np.inf

    for ped in pedestrians:
        for veh in vehicles:
            d = _min_dist_pair(ped, veh, dt, n_steps)
            if d < min_dist:
                min_dist = d

    return float(min_dist)


def _min_dist_pair(
    ped: TrackedObject,
    veh: TrackedObject,
    dt: float,
    n_steps: int,
) -> float:
    """Minimum Euclidean distance over n_steps linear extrapolation steps."""
    ped_pos = ped.xyz.copy()
    veh_pos = veh.xyz.copy()

    # Use XZ plane distance (ignore vertical) for car-park geometry
    min_d = np.linalg.norm(ped_pos[[0, 2]] - veh_pos[[0, 2]])

    for _ in range(n_steps):
        ped_pos = ped_pos + ped.vel * dt
        veh_pos = veh_pos + veh.vel * dt
        d = np.linalg.norm(ped_pos[[0, 2]] - veh_pos[[0, 2]])
        if d < min_d:
            min_d = d

    return min_d


# ── Velocity estimator ───────────────────────────────────────────────────────

class VelocityEstimator:
    """
    Maintains a rolling position history per track ID and returns a smoothed
    velocity estimate.  Call update() each frame, get_velocity() any time.
    """

    def __init__(self, history_frames: int = 6, fps: float = 30.0):
        self._history: dict[int, list[np.ndarray]] = {}
        self._n = history_frames
        self._dt = 1.0 / fps

    def update(self, track_id: int, xyz: np.ndarray) -> None:
        buf = self._history.setdefault(track_id, [])
        buf.append(xyz.copy())
        if len(buf) > self._n:
            buf.pop(0)

    def get_velocity(self, track_id: int) -> np.ndarray:
        buf = self._history.get(track_id, [])
        if len(buf) < 2:
            return np.zeros(3)
        # Central difference over available window for smoothing
        delta = buf[-1] - buf[0]
        elapsed = (len(buf) - 1) * self._dt
        return delta / elapsed if elapsed > 0 else np.zeros(3)

    def prune(self, active_ids: set[int]) -> None:
        stale = [k for k in self._history if k not in active_ids]
        for k in stale:
            del self._history[k]
