"""
Offline detector benchmark — runs the SafeEdge pipeline on a captured video
and compares YOLO model sizes and confidence thresholds.

Usage:
  python tools/benchmark_detector.py --video data/captures/carpark_daylight_01.mp4
  python tools/benchmark_detector.py --video data/captures/carpark_daylight_01.mp4 --max-frames 300
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
CONF_THRESHOLDS = [0.25, 0.35, 0.45]
TRACKED_CLASSES = {0: "person", 2: "car", 5: "bus", 7: "truck", 3: "motorcycle"}


@dataclass
class FrameResult:
    persons: int = 0
    vehicles: int = 0
    inference_ms: float = 0.0


@dataclass
class RunStats:
    model: str
    conf: float
    frames: int = 0
    frames_with_person: int = 0
    frames_with_vehicle: int = 0
    frames_with_both: int = 0
    total_persons: int = 0
    total_vehicles: int = 0
    inference_ms: list[float] = field(default_factory=list)

    @property
    def person_detection_rate(self) -> float:
        return self.frames_with_person / self.frames if self.frames else 0.0

    @property
    def vehicle_detection_rate(self) -> float:
        return self.frames_with_vehicle / self.frames if self.frames else 0.0

    @property
    def median_ms(self) -> float:
        return float(np.median(self.inference_ms)) if self.inference_ms else 0.0

    @property
    def p95_ms(self) -> float:
        return float(np.percentile(self.inference_ms, 95)) if self.inference_ms else 0.0

    @property
    def fps(self) -> float:
        return 1000.0 / self.median_ms if self.median_ms > 0 else 0.0


def run_model(video_path: Path, model_name: str, conf: float,
              max_frames: int, device: str) -> RunStats:
    from ultralytics import YOLO
    model = YOLO(model_name)
    stats = RunStats(model=model_name, conf=conf)

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0

    while cap.isOpened() and frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.perf_counter()
        results = model.predict(
            frame,
            conf=conf,
            device=device,
            classes=list(TRACKED_CLASSES.keys()),
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        persons = vehicles = 0
        for r in results:
            for box in r.boxes:
                label = TRACKED_CLASSES.get(int(box.cls[0]))
                if label == "person":
                    persons += 1
                elif label in ("car", "truck", "bus", "motorcycle"):
                    vehicles += 1

        stats.frames += 1
        stats.total_persons  += persons
        stats.total_vehicles += vehicles
        stats.inference_ms.append(elapsed_ms)
        if persons  > 0: stats.frames_with_person  += 1
        if vehicles > 0: stats.frames_with_vehicle += 1
        if persons  > 0 and vehicles > 0: stats.frames_with_both += 1

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  [{model_name} conf={conf}] frame {frame_idx}/{max_frames} "
                  f"| {elapsed_ms:.0f}ms | ped:{persons} veh:{vehicles}")

    cap.release()
    return stats


def print_table(all_stats: list[RunStats]) -> None:
    print("\n" + "═" * 100)
    print(f"{'MODEL':<16} {'CONF':>5} {'FRAMES':>7} {'PED%':>6} {'VEH%':>6} "
          f"{'BOTH%':>6} {'PED/f':>6} {'VEH/f':>6} {'med ms':>8} {'p95 ms':>8} {'FPS':>6}")
    print("─" * 100)
    for s in all_stats:
        both_pct = s.frames_with_both / s.frames * 100 if s.frames else 0
        print(
            f"{s.model:<16} {s.conf:>5.2f} {s.frames:>7} "
            f"{s.person_detection_rate*100:>5.1f}% {s.vehicle_detection_rate*100:>5.1f}% "
            f"{both_pct:>5.1f}% "
            f"{s.total_persons/s.frames:>6.2f} {s.total_vehicles/s.frames:>6.2f} "
            f"{s.median_ms:>8.1f} {s.p95_ms:>8.1f} {s.fps:>6.1f}"
        )
    print("═" * 100)

    # Recommendation
    best = max(all_stats, key=lambda s: s.person_detection_rate * 0.6 + s.vehicle_detection_rate * 0.4)
    print(f"\nBest for person detection: {best.model} conf={best.conf} "
          f"({best.person_detection_rate*100:.1f}% frames with person, {best.fps:.1f} fps)")


def render_comparison(video_path: Path, stats: list[RunStats],
                      out_path: Path, device: str, max_frames: int = 300) -> None:
    """Write a side-by-side comparison video for the top-2 configs."""
    from ultralytics import YOLO

    ranked = sorted(stats, key=lambda s: s.person_detection_rate, reverse=True)
    configs = ranked[:2]
    models  = [YOLO(c.model) for c in configs]

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    if not ret:
        return
    h, w = frame.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             15, (w * 2, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    COLOURS = [(0, 220, 60), (220, 80, 0)]   # green=person, blue=vehicle
    frame_idx = 0
    print(f"\nRendering comparison video → {out_path}")
    while cap.isOpened() and frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        panels = []
        for i, (cfg, model) in enumerate(zip(configs, models)):
            panel = frame.copy()
            results = model.predict(panel, conf=cfg.conf, device=device,
                                    classes=list(TRACKED_CLASSES.keys()), verbose=False)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    label = TRACKED_CLASSES.get(int(box.cls[0]), "?")
                    col = COLOURS[0] if label == "person" else COLOURS[1]
                    cv2.rectangle(panel, (x1, y1), (x2, y2), col, 2)
                    cv2.putText(panel, f"{label} {float(box.conf[0]):.2f}",
                                (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_PLAIN, 1.0, col, 1)
            cv2.rectangle(panel, (0, 0), (w, 28), (20, 20, 20), -1)
            cv2.putText(panel, f"{cfg.model}  conf={cfg.conf}  "
                               f"ped:{cfg.person_detection_rate*100:.0f}%",
                        (6, 20), cv2.FONT_HERSHEY_PLAIN, 1.1, (200, 200, 200), 1)
            panels.append(panel)
        writer.write(np.hstack(panels))
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Comparison video saved → {out_path} ({frame_idx} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",      required=True,  help="Path to captured MP4")
    parser.add_argument("--max-frames", type=int, default=450, help="Frames per run (default 450 = 15s@30fps)")
    parser.add_argument("--models",     nargs="+", default=MODELS)
    parser.add_argument("--confs",      nargs="+", type=float, default=CONF_THRESHOLDS)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--no-video",   action="store_true", help="Skip comparison video output")
    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        raise FileNotFoundError(video)

    print(f"Benchmarking on: {video}  ({args.max_frames} frames per run)")
    print(f"Models: {args.models}  Confs: {args.confs}  Device: {args.device}\n")

    all_stats: list[RunStats] = []
    for model_name in args.models:
        for conf in args.confs:
            print(f"→ {model_name}  conf={conf}")
            s = run_model(video, model_name, conf, args.max_frames, args.device)
            all_stats.append(s)

    print_table(all_stats)

    if not args.no_video:
        render_comparison(
            video, all_stats,
            out_path=video.parent / (video.stem + "_comparison.mp4"),
            device=args.device,
            max_frames=min(args.max_frames, 300),
        )
