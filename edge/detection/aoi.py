"""Area-of-Interest (AOI) helper — two roles, geometry-safe by design.

  B) crop-zoom for detection: crop the frame to the AOI bounding box so the
     detector sees the region larger (YOLO upscales the crop to its input size)
     → better recall on small/distant pedestrians. Detections are remapped back
     to FULL-FRAME pixel coords, so depth back-projection + camera intrinsics are
     never touched and metric distances / ρ stay correct.

  A) polygon filter: keep only tracks whose foot-point lies inside the AOI
     polygon → fewer irrelevant person↔vehicle pairs, sharper STL signal.

Polygon is configured in normalised [0,1] coords, so it is resolution-independent.
"""
from __future__ import annotations

import cv2
import numpy as np


class AOI:
    def __init__(self, polygon_norm, frame_w: int, frame_h: int,
                 pad: float = 0.08, crop_zoom: bool = True):
        self.crop_zoom = crop_zoom
        self.poly = np.array([[x * frame_w, y * frame_h] for x, y in polygon_norm],
                             dtype=np.float32)
        xs, ys = self.poly[:, 0], self.poly[:, 1]
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        pw, ph = (x1 - x0) * pad, (y1 - y0) * pad
        self.x0 = max(0, int(x0 - pw))
        self.y0 = max(0, int(y0 - ph))
        self.x1 = min(frame_w, int(x1 + pw))
        self.y1 = min(frame_h, int(y1 + ph))

    # ── B: crop-zoom detection ────────────────────────────────────────────────
    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y0:self.y1, self.x0:self.x1]

    def remap_dets(self, dets):
        """Shift crop-space bboxes back to full-frame pixel coords (in place)."""
        off = np.array([self.x0, self.y0, self.x0, self.y0], dtype=np.float32)
        for d in dets:
            d.bbox_xyxy = (d.bbox_xyxy.astype(np.float32) + off)
        return dets

    # ── A: polygon filter ──────────────────────────────────────────────────────
    def contains_foot(self, bbox_xyxy) -> bool:
        fx = float((bbox_xyxy[0] + bbox_xyxy[2]) / 2.0)
        fy = float(bbox_xyxy[3])          # bottom-centre = ground contact point
        return cv2.pointPolygonTest(self.poly, (fx, fy), False) >= 0

    # ── overlay ─────────────────────────────────────────────────────────────────
    def draw(self, img: np.ndarray, color=(0, 200, 200)) -> None:
        cv2.polylines(img, [self.poly.astype(np.int32)], True, color, 1, cv2.LINE_AA)
