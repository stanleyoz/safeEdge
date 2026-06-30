"""
Capture video to MP4 for offline SafeEdge testing.

Sources:
  --source realsense   D455 colour via pyrealsense2 (default, best quality)
  --source webcam      USB webcam via OpenCV (fallback)

Usage:
  python tools/capture_video.py --source realsense --duration 180 --out data/captures/carpark_01.mp4
  python tools/capture_video.py --source webcam --device 0 --duration 180

Press Q in the preview window or Ctrl+C to stop early.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ── RealSense capture ────────────────────────────────────────────────────────

def capture_realsense(out_path: Path, width: int, height: int,
                      fps: int, duration: float) -> None:
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(cfg)
    print(f"RealSense D455 opened: {width}×{height} @ {fps}fps")

    _record(pipeline_read=lambda: _rs_read(pipeline),
            width=width, height=height, fps=fps,
            out_path=out_path, duration=duration,
            stop_fn=pipeline.stop)


def _rs_read(pipeline):
    frames = pipeline.wait_for_frames(timeout_ms=5000)
    cf = frames.get_color_frame()
    if not cf:
        return None
    return np.asanyarray(cf.get_data())


# ── Webcam capture ───────────────────────────────────────────────────────────

def capture_webcam(device: int, out_path: Path, width: int, height: int,
                   fps: float, duration: float) -> None:
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open /dev/video{device}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    print(f"Webcam /dev/video{device}: {actual_w}×{actual_h} @ {actual_fps:.0f}fps")

    def read_fn():
        ok, frame = cap.read()
        return frame if ok else None

    _record(pipeline_read=read_fn,
            width=actual_w, height=actual_h, fps=actual_fps,
            out_path=out_path, duration=duration,
            stop_fn=cap.release)


# ── Shared recording loop ────────────────────────────────────────────────────

def _record(pipeline_read, width, height, fps, out_path, duration, stop_fn):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height),
    )
    print(f"Recording → {out_path}  ({duration:.0f}s max, Q to stop early)")

    t_start = time.monotonic()
    frames = 0
    try:
        while True:
            elapsed = time.monotonic() - t_start
            if elapsed >= duration:
                break

            frame = pipeline_read()
            if frame is None:
                print("Frame read failed — stopping.")
                break

            writer.write(frame)
            frames += 1

            preview = frame.copy()
            remaining = max(0, duration - elapsed)
            cv2.putText(preview,
                        f"REC {elapsed:.0f}s/{duration:.0f}s  {remaining:.0f}s left  "
                        f"f:{frames}  Q=stop",
                        (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(preview,
                        f"REC {elapsed:.0f}s/{duration:.0f}s  {remaining:.0f}s left  "
                        f"f:{frames}  Q=stop",
                        (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 220), 1)
            cv2.circle(preview, (width - 24, 24), 10, (0, 0, 220), -1)
            cv2.imshow("SafeEdge Capture", preview)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Stopped by user.")
                break
    except KeyboardInterrupt:
        print("\nCtrl+C — stopping.")
    finally:
        stop_fn()
        writer.release()
        cv2.destroyAllWindows()

    size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0
    print(f"\nSaved {frames} frames ({frames/fps:.1f}s) → {out_path}  ({size_mb:.1f} MB)")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture video for offline SafeEdge testing")
    parser.add_argument("--source",   choices=["realsense", "webcam"], default="realsense")
    parser.add_argument("--device",   type=int,   default=0,     help="Webcam device index")
    parser.add_argument("--out",      type=str,   default="data/captures/capture.mp4")
    parser.add_argument("--duration", type=float, default=180.0, help="Max seconds to record")
    parser.add_argument("--width",    type=int,   default=848)
    parser.add_argument("--height",   type=int,   default=480)
    parser.add_argument("--fps",      type=int,   default=30)
    args = parser.parse_args()

    out = Path(args.out)
    if args.source == "realsense":
        capture_realsense(out, args.width, args.height, args.fps, args.duration)
    else:
        capture_webcam(args.device, out, args.width, args.height, args.fps, args.duration)

