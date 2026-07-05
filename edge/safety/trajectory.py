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

import os
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

    def __init__(self, history_frames: int = 10, fps: float = 30.0,
                 max_speed: float = 5.0, pos_smooth: float | None = None,
                 resync_after: int = 3):
        self._history: dict[int, list[np.ndarray]] = {}
        self._smoothed: dict[int, np.ndarray] = {}
        self._misses: dict[int, int] = {}
        self._n = history_frames
        self._dt = 1.0 / fps
        self._max_speed = max_speed                 # physical cap (m/s)
        self._max_jump = max_speed * self._dt       # max plausible move / frame
        # EMA position low-pass (per track). A parked car's bounding box is
        # redrawn slightly every frame; the homography amplifies a few px of
        # foot-point wobble into a phantom velocity approaching ~1 m/s. Smoothing
        # the POSITION before differentiating settles a static object toward zero
        # while real motion still passes (with a small lag). alpha in (0,1]:
        # 1.0 = no smoothing, lower = calmer. Env-tunable for offline sweeps.
        a = pos_smooth if pos_smooth is not None else \
            float(os.environ.get("SAFEEDGE_POS_SMOOTH", "0.35"))
        self._alpha = min(1.0, max(0.05, a))
        # Consecutive outlier-rejections tolerated before resyncing to the raw
        # sample. Without this, rejecting a jump vs the SMOOTHED reference is a
        # one-way lock: the reference never advances, so a real fast move (or a
        # detection gap from brief occlusion) makes every subsequent frame fail
        # the same check even harder — velocity freezes at its last value
        # forever. Found live: v_veh_max stuck at one value for the rest of a run.
        self._resync_after = max(1, resync_after)

    def update(self, track_id: int, xyz: np.ndarray) -> None:
        prev_sm = self._smoothed.get(track_id)
        if prev_sm is not None:
            jump = float(np.linalg.norm((xyz - prev_sm)[[0, 2]]))
            if jump > self._max_jump:
                misses = self._misses.get(track_id, 0) + 1
                self._misses[track_id] = misses
                if misses < self._resync_after:
                    return  # transient glitch — hold position, wait for confirmation
                # Sustained disagreement: this is real motion (or a detection
                # gap) we lost sync with, not one-frame jitter. Resync hard —
                # accept the raw sample as the new anchor and drop stale history
                # (it no longer reflects where the object actually is).
                self._smoothed[track_id] = xyz.copy()
                self._history[track_id] = [xyz.copy()]
                self._misses[track_id] = 0
                return
            sm = self._alpha * xyz + (1.0 - self._alpha) * prev_sm
        else:
            sm = xyz.copy()
        self._misses[track_id] = 0
        self._smoothed[track_id] = sm

        buf = self._history.setdefault(track_id, [])
        buf.append(sm.copy())
        if len(buf) > self._n:
            buf.pop(0)

    def get_velocity(self, track_id: int) -> np.ndarray:
        buf = self._history.get(track_id, [])
        if len(buf) < 3:
            return np.zeros(3)
        # Component-wise MEDIAN of per-step velocities — robust to single-frame
        # jitter (vs endpoint difference which a single outlier corrupts).
        steps = np.array([(buf[i] - buf[i - 1]) / self._dt
                          for i in range(1, len(buf))])
        vel = np.median(steps, axis=0)
        # Final safety clamp to the physical cap
        speed = float(np.linalg.norm(vel))
        if speed > self._max_speed:
            vel = vel * (self._max_speed / speed)
        return vel

    def prune(self, active_ids: set[int]) -> None:
        stale = [k for k in self._history if k not in active_ids]
        for k in stale:
            del self._history[k]
            self._smoothed.pop(k, None)
            self._misses.pop(k, None)
