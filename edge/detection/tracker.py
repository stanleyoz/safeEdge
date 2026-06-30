"""
ByteTrack wrapper via the supervision library.
Assigns persistent track IDs across frames to each detection.
"""
from __future__ import annotations

import numpy as np
import supervision as sv

from edge.detection.signal_extractor import RawDetection


class ObjectTracker:
    def __init__(self, lost_track_buffer: int = 30):
        # supervision ≥0.22 renamed ByteTracker → ByteTrack
        TrackerCls = getattr(sv, "ByteTrack", None) or sv.ByteTracker
        self._tracker = TrackerCls(
            track_activation_threshold=0.4,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=0.8,
            frame_rate=30,
        )

    def update(
        self, detections: list[RawDetection], frame_hw: tuple[int, int]
    ) -> list[RawDetection]:
        if not detections:
            return []

        xyxy  = np.array([d.bbox_xyxy  for d in detections])
        confs = np.array([d.confidence for d in detections])
        clsids = np.zeros(len(detections), dtype=int)  # class grouping handled upstream

        sv_dets = sv.Detections(
            xyxy=xyxy,
            confidence=confs,
            class_id=clsids,
        )
        tracked = self._tracker.update_with_detections(sv_dets)

        result: list[RawDetection] = []
        for i, track_id in enumerate(tracked.tracker_id):
            if i < len(detections):
                d = detections[i]
                result.append(RawDetection(
                    track_id=int(track_id),
                    label=d.label,
                    bbox_xyxy=d.bbox_xyxy,
                    confidence=d.confidence,
                ))
        return result
