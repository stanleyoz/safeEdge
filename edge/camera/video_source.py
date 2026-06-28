"""
Generic RGB video source — USB camera, RTSP stream, or video file.
Replaces the RealSense node. No depth required.

MockCamera generates a synthetic near-miss scenario for offline dev/demo.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraFrame:
    color_bgr: np.ndarray    # H×W×3 uint8
    timestamp: float


class VideoSource:
    """
    Wraps cv2.VideoCapture.  source can be:
      - int          : USB device index (0, 1, …)
      - "rtsp://…"   : RTSP stream from WiFi camera
      - "/path/…mp4" : pre-recorded video file (demo / testing)
    """

    def __init__(self, source: int | str, target_fps: float = 15.0):
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source!r}")
        self._target_fps = target_fps
        # request resolution from camera if supported (ignored by files)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self) -> CameraFrame | None:
        ok, frame = self._cap.read()
        if not ok:
            return None
        return CameraFrame(color_bgr=frame, timestamp=time.time())

    def stop(self) -> None:
        self._cap.release()


class MockCamera:
    """
    Generates synthetic pedestrian + vehicle frames for offline demo.
    No depth — positions are painted directly on a top-down schematic.
    Scenario: pedestrian crosses left→right, vehicle approaches from top,
    converge to ~1.2m clearance at t≈5s, then diverge.
    """

    # Ground plane extents rendered in the frame (metres)
    WORLD_W = 12.0
    WORLD_H = 10.0

    def __init__(self, fps: float = 15.0):
        self._fps = fps
        self._dt  = 1.0 / fps
        self._t   = 0.0
        self._img_w = 640
        self._img_h = 480

    def read(self) -> CameraFrame:
        frame = self._render()
        self._t += self._dt
        return CameraFrame(color_bgr=frame, timestamp=time.time())

    def stop(self) -> None:
        pass

    # ── Private ──────────────────────────────────────────────────────────────

    def _world_to_px(self, x: float, z: float) -> tuple[int, int]:
        """Map world (X,Z) metres → image pixel (u,v)."""
        u = int((x / self.WORLD_W) * self._img_w)
        v = int((z / self.WORLD_H) * self._img_h)
        return u, v

    def _render(self) -> np.ndarray:
        img = np.full((self._img_h, self._img_w, 3), 30, dtype=np.uint8)

        # Parking bay grid (visual only)
        for xi in range(0, self._img_w, self._img_w // 4):
            cv2.line(img, (xi, 0), (xi, self._img_h), (50, 50, 50), 1)
        for zi in range(0, self._img_h, self._img_h // 5):
            cv2.line(img, (0, zi), (self._img_w, zi), (50, 50, 50), 1)

        t = self._t

        # Pedestrian (green box)
        ped_x = -1.0 + t * 1.0       # walks right
        ped_z = 5.0
        pu, pv = self._world_to_px(ped_x + self.WORLD_W / 2, ped_z)
        cv2.rectangle(img, (pu - 12, pv - 30), (pu + 12, pv + 10), (0, 200, 60), -1)
        cv2.putText(img, "PED", (pu - 14, pv - 34), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 80), 1)

        # Vehicle (blue box, enters at t=2s)
        if t >= 2.0:
            veh_x = 1.0
            veh_z = max(3.0, 9.0 - (t - 2.0) * 2.0)
            vu, vv = self._world_to_px(veh_x + self.WORLD_W / 2, veh_z)
            cv2.rectangle(img, (vu - 30, vv - 20), (vu + 30, vv + 20), (200, 60, 0), -1)
            cv2.putText(img, "VEH", (vu - 14, vv - 24), cv2.FONT_HERSHEY_PLAIN, 0.8, (80, 120, 255), 1)

            # Distance line
            d = np.hypot(ped_x - veh_x, ped_z - veh_z)
            colour = (0, 255, 0) if d > 3.0 else ((0, 165, 255) if d > 1.5 else (0, 0, 255))
            cv2.line(img, (pu, pv), (vu, vv), colour, 1)
            cv2.putText(img, f"{d:.1f}m", ((pu + vu) // 2, (pv + vv) // 2 - 6),
                        cv2.FONT_HERSHEY_PLAIN, 0.9, colour, 1)

        cv2.putText(img, f"t={t:.1f}s  MOCK", (8, 18),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (180, 180, 180), 1)
        return img
