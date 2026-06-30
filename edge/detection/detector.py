"""
YOLOv8 wrapper.  On Jetson (JetPack 7.x) export the model to TensorRT first:
  yolo export model=yolov8n.pt format=engine device=0 half=True imgsz=640

On dev machine runs in standard PyTorch mode.  Both return the same
RawDetection list — the rest of the pipeline is hardware-agnostic.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

from edge.detection.signal_extractor import RawDetection

TRACKED_CLASSES = {0: "person", 2: "car", 5: "bus", 7: "truck", 3: "motorcycle"}


class ObjectDetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.4,
        device: str = "auto",
    ):
        self._model = None
        self._conf = confidence
        self._device = device
        try:
            import torch
            from ultralytics import YOLO
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = device
            self._model = YOLO(model_path)
            logger.info("ObjectDetector: YOLO loaded, device=%s", device)
        except Exception as e:
            logger.warning("ObjectDetector: YOLO unavailable (%s) — returning empty detections", e)

    def detect(self, frame_bgr: np.ndarray) -> list[RawDetection]:
        if self._model is None:
            return []
        results = self._model.predict(
            frame_bgr,
            conf=self._conf,
            device=self._device,
            classes=list(TRACKED_CLASSES.keys()),
            verbose=False,
        )
        detections: list[RawDetection] = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = TRACKED_CLASSES.get(cls_id)
                if label is None:
                    continue
                detections.append(RawDetection(
                    track_id=-1,
                    label=label,
                    bbox_xyxy=box.xyxy[0].cpu().numpy(),
                    confidence=float(box.conf[0]),
                ))
        return detections
