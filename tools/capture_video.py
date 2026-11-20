"""
Capture video to MP4 for offline SafeEdge testing.

Sources:
  --source realsense   D455 colour via pyrealsense2 (default, best quality)
  --source webcam      USB webcam via OpenCV (fallback)

Usage:
  python tools/capture_video.py --source realsense --duration 180 --out data/captures/carpark_01.mp4
  python tools/capture_video.py --source webcam --device 0 --duration 180

Press Q in the preview window or Ctrl+C to stop early. Use --no-preview when
recording headless (e.g. over SSH on the Jetson).

Every capture also writes <out>.frames.csv with one row per frame: frame
index, wall-clock time (Unix seconds, from the system clock, so NTP-synced)
and monotonic seconds since the first frame. The container's fps header is
only the requested rate; the true rate comes from these timestamps.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ── RealSense capture ────────────────────────────────────────────────────────

def capture_realsense(out_path: Path, width: int, height: int,
                      fps: int, duration: float, preview: bool = True) -> None:
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(cfg)
    print(f"RealSense D455 opened: {width}×{height} @ {fps}fps")

    _record(pipeline_read=lambda: _rs_read(pipeline),
            width=width, height=height, fps=fps,
            out_path=out_path, duration=duration,
            stop_fn=pipeline.stop, preview=preview)


def _rs_read(pipeline):
    frames = pipeline.wait_for_frames(timeout_ms=5000)
    cf = frames.get_color_frame()
    if not cf:
        return None
    return np.asanyarray(cf.get_data())


# ── Webcam capture ───────────────────────────────────────────────────────────

def capture_webcam(device: int, out_path: Path, width: int, height: int,
                   fps: float, duration: float, preview: bool = True) -> None:
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
            stop_fn=cap.release, preview=preview)


# ── Shared recording loop ────────────────────────────────────────────────────

def timestamps_path(out_path: Path) -> Path:
    return out_path.with_suffix(".frames.csv")


def _record(pipeline_read, width, height, fps, out_path, duration, stop_fn,
            preview=True):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height),
    )
    ts_file = open(timestamps_path(out_path), "w", newline="")
    ts = csv.writer(ts_file)
    ts.writerow(["frame", "wall_clock_s", "mono_s"])
    stop_hint = "Q to stop early" if preview else "Ctrl+C to stop early"
    print(f"Recording → {out_path}  ({duration:.0f}s max, {stop_hint})")

    t_start = time.monotonic()
    t_first = t_last = None
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
            wall, mono = time.time(), time.monotonic()

            writer.write(frame)
            if t_first is None:
                t_first = mono
            t_last = mono
            ts.writerow([frames, f"{wall:.6f}", f"{mono - t_first:.6f}"])
            frames += 1
            if frames % 100 == 0:
                ts_file.flush()

            if not preview:
                continue

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
        ts_file.close()
        if preview:
            cv2.destroyAllWindows()

    size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0
    span = (t_last - t_first) if frames > 1 else 0.0
    true_fps = (frames - 1) / span if span > 0 else 0.0
    print(f"\nSaved {frames} frames ({span:.1f}s, measured {true_fps:.2f} fps; "
          f"header says {fps:g}) → {out_path}  ({size_mb:.1f} MB)")
    print(f"Per-frame timestamps → {timestamps_path(out_path)}")


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
    parser.add_argument("--no-preview", action="store_true",
                        help="Do not open a preview window (headless capture)")
    args = parser.parse_args()

    out = Path(args.out)
    preview = not args.no_preview
    if args.source == "realsense":
        capture_realsense(out, args.width, args.height, args.fps, args.duration, preview)
    else:
        capture_webcam(args.device, out, args.width, args.height, args.fps, args.duration,
                       preview)

