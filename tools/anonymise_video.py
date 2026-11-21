"""
Blur every person in a clip so it can be published, and encode it as H.264
for browsers.

Detection runs on the full frame (the edge loop only detects inside its zone
crop, so people on the near lawn and path never reach the trace) with a low
confidence threshold, and every box is also applied to the frames on either
side, so a person the detector misses for a frame or two stays blurred.

  python tools/anonymise_video.py data/segment_0455_to_end.mp4 \\
      docs/monitor/media/segment_0455_to_end.blurred.mp4 --fps 7.7 \\
      --review review.jpg

--fps re-times the output to the true capture rate (the July captures claim
15 fps in their header but were recorded at 7.7). --review writes a sheet of
the frames with the largest people, blurred, to check by eye before
publishing. Number plates are not blurred: at 1280x720 the nearest plate in
the car-park view is about 12 px wide and unreadable; check this for any new
camera.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np


def detect_people(path: Path, weights: str, conf: float, imgsz: int) -> list[np.ndarray]:
    from ultralytics import YOLO
    model = YOLO(weights)
    cap = cv2.VideoCapture(str(path))
    boxes = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        r = model.predict(im, classes=[0], conf=conf, imgsz=imgsz, verbose=False)[0]
        boxes.append(r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4)))
    cap.release()
    return boxes


def blur_regions(im: np.ndarray, boxes: np.ndarray, pad: float, kernel_frac: float) -> np.ndarray:
    h, w = im.shape[:2]
    out = im.copy()
    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        x1, x2 = int(max(0, x1 - pad * bw)), int(min(w, x2 + pad * bw))
        y1, y2 = int(max(0, y1 - pad * bh)), int(min(h, y2 + pad * bh))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        roi = out[y1:y2, x1:x2]
        k = max(15, int(max(x2 - x1, y2 - y1) * kernel_frac)) | 1
        # pixelate then smooth: no facial detail survives, the silhouette does
        small = cv2.resize(roi, (max(1, (x2 - x1) // 8), max(1, (y2 - y1) // 8)), interpolation=cv2.INTER_AREA)
        out[y1:y2, x1:x2] = cv2.GaussianBlur(cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR),
                                             (k, k), 0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--fps", type=float, required=True, help="true capture fps (output frame rate)")
    ap.add_argument("--weights", default="yolov8s.pt")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--hold", type=int, default=3, help="also blur each box this many frames before and after")
    ap.add_argument("--pad", type=float, default=0.2, help="grow boxes by this x their size per side")
    ap.add_argument("--kernel-frac", type=float, default=0.5)
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--review", type=Path, default=None)
    args = ap.parse_args()

    dets = detect_people(args.src, args.weights, args.conf, args.imgsz)
    n = len(dets)
    print(f"{n} frames, {sum(len(d) for d in dets)} person detections, "
          f"{sum(1 for d in dets if len(d))} frames with people")

    import imageio_ffmpeg
    cap = cv2.VideoCapture(str(args.src))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{args.fps}",
                            "-i", "-", "-c:v", "libx264", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", "-an", str(args.dst)], stdin=subprocess.PIPE)
    sizes = []
    for i in range(n):
        ok, im = cap.read()
        if not ok:
            break
        near = [dets[j] for j in range(max(0, i - args.hold), min(n, i + args.hold + 1)) if len(dets[j])]
        boxes = np.concatenate(near) if near else np.zeros((0, 4))
        out = blur_regions(im, boxes, args.pad, args.kernel_frac)
        enc.stdin.write(out.tobytes())
        sizes.append((float(((dets[i][:, 3] - dets[i][:, 1]).max()) if len(dets[i]) else 0.0), i))
    enc.stdin.close()
    enc.wait()
    cap.release()
    print(f"wrote {args.dst} ({args.dst.stat().st_size / 1e6:.1f} MB)")

    if args.review:
        # the frames with the tallest (nearest) people, after blurring, 4 x 3
        picks, chosen = [], []
        for size, i in sorted(sizes, reverse=True):
            if size and all(abs(i - j) > 3 * args.fps for j in chosen):
                chosen.append(i)
            if len(chosen) == 12:
                break
        rd = cv2.VideoCapture(str(args.dst))
        for i in sorted(chosen):
            rd.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, im = rd.read()
            if ok:
                im = cv2.resize(im, (640, 360))
                cv2.putText(im, f"f{i}", (8, 24), 0, 0.7, (255, 255, 255), 2)
                picks.append(im)
        while len(picks) % 4:
            picks.append(np.zeros_like(picks[0]))
        cv2.imwrite(str(args.review), np.vstack([np.hstack(picks[r:r + 4]) for r in range(0, len(picks), 4)]))
        print(f"review sheet {args.review}")


if __name__ == "__main__":
    main()
