"""
SafeEdge live recorder — D455 + GPU YOLOv8 + distance annotations → MP4.

Filename: safeedge_YYYYMMDD_HHMMSS__YYYYMMDD_HHMMSS.mp4
Press Q to stop and finalise.

Usage (inside Docker):
  python3 tools/live_record.py [--out-dir /safeedge/data/captures]
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ── config ────────────────────────────────────────────────────────────────────
W, H      = 848, 480
FPS_CAM   = 30
MODEL     = "/safeedge/yolov8s.pt"
CONF      = 0.25
DEVICE    = 0           # GPU
CLASSES   = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COL_PED   = (0,   220,  50)    # green
COL_VEH   = (0,   140, 255)    # orange
COL_DIST  = (0,   220, 255)    # yellow  – danger line
COL_HUD   = (220, 220, 220)
FONT      = cv2.FONT_HERSHEY_SIMPLEX
DEPTH_MAX = 15.0                # metres — clamp depth colourmap

# ── helpers ───────────────────────────────────────────────────────────────────

def foot_depth(depth_m: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    """Median depth in a small patch at the bottom-centre of a bounding box."""
    fx = (x1 + x2) // 2
    fy = min(y2, H - 1)
    patch = depth_m[max(fy-8, 0):fy+8, max(fx-8, 0):fx+8]
    valid = patch[patch > 0.1]
    return float(np.median(valid)) if valid.size else 0.0


def draw_box(frame, x1, y1, x2, y2, label, conf_v, dist_m, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    dist_str = f" {dist_m:.1f}m" if dist_m > 0 else ""
    text = f"{label} {conf_v:.2f}{dist_str}"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.52, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 4), FONT, 0.52, (0, 0, 0), 1, cv2.LINE_AA)


def draw_danger_line(frame, p_box, v_box, p_dist, v_dist):
    """Draw a line between closest person–vehicle pair with combined distance."""
    px = (p_box[0] + p_box[2]) // 2
    py = (p_box[1] + p_box[3]) // 2
    vx = (v_box[0] + v_box[2]) // 2
    vy = (v_box[1] + v_box[3]) // 2
    cv2.line(frame, (px, py), (vx, vy), COL_DIST, 2, cv2.LINE_AA)
    mx, my = (px + vx) // 2, (py + vy) // 2
    if p_dist > 0 and v_dist > 0:
        sep = abs(p_dist - v_dist)
        cv2.putText(frame, f"{sep:.1f}m apart", (mx - 30, my - 6),
                    FONT, 0.5, COL_DIST, 1, cv2.LINE_AA)


def hud(frame, fn, total, fps_inf, fps_cap, n_ped, n_veh, elapsed):
    cv2.rectangle(frame, (0, 0), (W, 28), (20, 20, 20), -1)
    ts = int(elapsed)
    info = (f"Frame {fn}  {ts//60:02d}:{ts%60:02d}  "
            f"ped:{n_ped}  veh:{n_veh}  "
            f"infer:{fps_inf:.1f}fps  cam:{fps_cap:.1f}fps  "
            f"[Q=stop]")
    cv2.putText(frame, info, (6, 19), FONT, 0.48, COL_HUD, 1, cv2.LINE_AA)
    # REC indicator
    cv2.circle(frame, (W - 18, 14), 6, (0, 0, 220), -1)
    cv2.putText(frame, "REC", (W - 48, 19), FONT, 0.45, (0, 0, 220), 1)


# ── main ──────────────────────────────────────────────────────────────────────

def main(out_dir: Path):
    import pyrealsense2 as rs
    from ultralytics import YOLO

    model = YOLO(MODEL)
    print(f"YOLO loaded on device={DEVICE}")

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS_CAM)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS_CAM)
    profile = pipeline.start(cfg)
    align   = rs.align(rs.stream.color)

    intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    print(f"D455: {W}x{H}@{FPS_CAM}  fx={intr.fx:.1f} fy={intr.fy:.1f}")

    start_dt  = datetime.now()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path  = out_dir / f"_tmp_{start_dt.strftime('%Y%m%d_%H%M%S')}.mp4"
    writer    = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS_CAM, (W, H))

    has_display = bool(__import__("os").environ.get("DISPLAY") or
                       __import__("os").environ.get("WAYLAND_DISPLAY"))
    print(f"Preview window: {'YES' if has_display else 'NO (headless)'}")
    print(f"Recording to:   {tmp_path}")
    print("Press Q in preview window or Ctrl-C to stop.")

    frame_n   = 0
    t_start   = time.perf_counter()
    fps_inf   = 0.0
    fps_cap   = 0.0
    t_last    = t_start

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)
            color_f = aligned.get_color_frame()
            depth_f = aligned.get_depth_frame()
            if not color_f or not depth_f:
                continue

            color = np.asanyarray(color_f.get_data())
            depth_m = (np.asanyarray(depth_f.get_data()).astype(np.float32)
                       * depth_f.get_units())

            t_inf0 = time.perf_counter()
            results = model.predict(color, conf=CONF, device=DEVICE,
                                    classes=list(CLASSES.keys()), verbose=False)
            fps_inf = 0.8 * fps_inf + 0.2 * (1.0 / (time.perf_counter() - t_inf0 + 1e-9))

            now = time.perf_counter()
            fps_cap = 0.8 * fps_cap + 0.2 * (1.0 / max(now - t_last, 1e-9))
            t_last  = now

            out = color.copy()
            peds, vehs = [], []  # (box, dist_m)

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label  = CLASSES.get(cls_id, "?")
                    conf_v = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    dist   = foot_depth(depth_m, x1, y1, x2, y2)
                    color_ = COL_PED if label == "person" else COL_VEH
                    draw_box(out, x1, y1, x2, y2, label, conf_v, dist, color_)
                    if label == "person":
                        peds.append(((x1, y1, x2, y2), dist))
                    else:
                        vehs.append(((x1, y1, x2, y2), dist))

            # Danger line: closest ped to any vehicle
            if peds and vehs:
                p_box, p_dist = min(peds, key=lambda x: x[1] if x[1] > 0 else 999)
                v_box, v_dist = min(vehs, key=lambda x: x[1] if x[1] > 0 else 999)
                if p_dist > 0 and v_dist > 0:
                    draw_danger_line(out, p_box, v_box, p_dist, v_dist)

            # Depth thumbnail (bottom-right, 25% size)
            thumb_w, thumb_h = W // 4, H // 4
            depth_vis = np.clip(depth_m / DEPTH_MAX, 0, 1)
            depth_col = cv2.applyColorMap((depth_vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            thumb = cv2.resize(depth_col, (thumb_w, thumb_h))
            out[H - thumb_h:H, W - thumb_w:W] = thumb
            cv2.putText(out, "depth", (W - thumb_w + 4, H - thumb_h + 14),
                        FONT, 0.4, (200, 200, 200), 1)

            hud(out, frame_n + 1, 0, fps_inf, fps_cap,
                len(peds), len(vehs), now - t_start)

            writer.write(out)
            frame_n += 1

            if has_display:
                cv2.imshow("SafeEdge Live", out)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        writer.release()
        if has_display:
            cv2.destroyAllWindows()

        end_dt  = datetime.now()
        final   = out_dir / (
            f"safeedge_{start_dt.strftime('%Y%m%d_%H%M%S')}"
            f"__{end_dt.strftime('%Y%m%d_%H%M%S')}.mp4"
        )
        tmp_path.rename(final)
        elapsed = (end_dt - start_dt).total_seconds()
        print(f"\nSaved: {final}")
        print(f"Frames: {frame_n}  Duration: {elapsed:.1f}s  Avg: {frame_n/elapsed:.1f}fps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/safeedge/data/captures",
                        help="Output directory for MP4")
    args = parser.parse_args()
    main(Path(args.out_dir))
