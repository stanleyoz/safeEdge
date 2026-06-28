"""
One-time homography calibration for SafeEdge.

Place 4 markers on the car park ground at known positions (measure with a tape).
Run this script, click each marker in the live camera feed, enter the real-world
coordinates when prompted.  The 3×3 homography matrix is saved to
config/homography.npy and used by the signal extractor.

Usage:
    python tools/calibrate_homography.py --source 0
    python tools/calibrate_homography.py --source "rtsp://192.168.1.50:554/stream"

Marker layout suggestion (measure from a fixed reference corner):
    P1 ──────── P2
    │            │
    │            │
    P3 ──────── P4

Controls during click phase:
    Left-click  — add a marker point
    'r'         — reset all points
    'q'         — quit without saving
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT    = Path(__file__).parent.parent
OUT_FILE = ROOT / "config" / "homography.npy"

_INSTRUCTIONS = [
    "Click marker P1 (top-left in world)",
    "Click marker P2 (top-right in world)",
    "Click marker P3 (bottom-left in world)",
    "Click marker P4 (bottom-right in world)",
]

_clicked: list[tuple[int, int]] = []


def _on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(_clicked) < 4:
        _clicked.append((x, y))
        print(f"  → Point {len(_clicked)}: pixel ({x}, {y})")


def capture_frame(source: int | str) -> np.ndarray:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Cannot open source: {source!r}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit("Failed to read frame from source")
    return frame


def get_world_points() -> np.ndarray:
    pts = []
    names = ["P1 (top-left)", "P2 (top-right)", "P3 (bottom-left)", "P4 (bottom-right)"]
    print("\nEnter real-world ground plane coordinates for each marker.")
    print("Coordinate system: X = horizontal (right+), Z = into scene (away+), both in metres.")
    print("Tip: set P1 as your origin (0, 0) and measure the others relative to it.\n")
    for name in names:
        while True:
            raw = input(f"  {name}  X Z (metres, space-separated): ").strip()
            try:
                x, z = map(float, raw.split())
                pts.append([x, z])
                break
            except ValueError:
                print("  Enter two numbers, e.g.:  0.0 0.0")
    return np.float32(pts)


def main():
    parser = argparse.ArgumentParser(description="Calibrate homography for SafeEdge")
    parser.add_argument("--source", default=0,
                        help="Camera source: device index, RTSP URL, or video file")
    args = parser.parse_args()

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    print("Capturing reference frame…")
    frame = capture_frame(source)
    display = frame.copy()

    cv2.namedWindow("SafeEdge Calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("SafeEdge Calibration", _on_click)

    print("\n=== SafeEdge Homography Calibration ===")
    print("Click the 4 ground markers in order: P1 → P2 → P3 → P4")
    print("Press 'r' to reset, 'q' to quit.\n")
    print(_INSTRUCTIONS[0])

    while True:
        vis = display.copy()

        for i, (u, v) in enumerate(_clicked):
            cv2.circle(vis, (u, v), 8, (0, 255, 0), -1)
            cv2.putText(vis, f"P{i+1}", (u + 10, v - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if len(_clicked) < 4:
            cv2.putText(vis, _INSTRUCTIONS[len(_clicked)], (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        else:
            cv2.putText(vis, "4 points selected — press ENTER to continue",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("SafeEdge Calibration", vis)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'):
            _clicked.clear()
            print("\nReset — click P1 again.")
            print(_INSTRUCTIONS[0])
        elif key == ord('q'):
            cv2.destroyAllWindows()
            sys.exit("Calibration cancelled.")
        elif key == 13 and len(_clicked) == 4:   # ENTER
            break

    cv2.destroyAllWindows()

    image_pts = np.float32(_clicked)
    world_pts = get_world_points()

    H, mask = cv2.findHomography(image_pts, world_pts)
    if H is None:
        sys.exit("Homography computation failed — check your points.")

    np.save(OUT_FILE, H)
    print(f"\nHomography matrix:\n{H}\n")
    print(f"Saved to {OUT_FILE}")

    # Quick sanity check — reproject the clicked points
    print("\nSanity check (image → world reprojection):")
    for i, (u, v) in enumerate(image_pts):
        pt = H @ np.array([u, v, 1.0])
        xw, zw = pt[0] / pt[2], pt[1] / pt[2]
        print(f"  P{i+1}: pixel ({int(u)},{int(v)}) → world ({xw:.2f}, {zw:.2f})  "
              f"target ({world_pts[i,0]:.2f}, {world_pts[i,1]:.2f})")


if __name__ == "__main__":
    main()
