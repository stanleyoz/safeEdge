"""Synthetic car-park clip + matching hand-made trace, for smoke-testing
tools/vjepa_surprise.py without real footage. It checks plumbing, not validity.

0-40 s: parked cars, a pedestrian walking back and forth. At 40 s a parked car
pulls out toward the pedestrian; the trace marks WARNING within 200 px and
EMERGENCY within 90 px.

  python research/vjepa-surprise/scripts/make_synthetic_clip.py /tmp/synth
  → /tmp/synth.mp4 + /tmp/synth.trace.jsonl
"""
import json, sys, cv2, numpy as np
out = sys.argv[1]; fps, W, H, dur = 15, 1280, 720, 60
rng = np.random.default_rng(0)
bg = np.full((H, W, 3), 95, np.uint8); bg += rng.integers(0, 12, (H, W, 1), dtype=np.uint8)
for x in range(80, W, 220): cv2.line(bg, (x, 380), (x, 600), (200, 200, 200), 3)
vw = cv2.VideoWriter(out + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
tr = open(out + ".trace.jsonl", "w")
for f in range(fps * dur):
    t = f / fps; img = bg.copy()
    for i, x in enumerate(range(100, W - 150, 220)):
        cx = x
        if i == 2 and t > 40: cx = x + min(600, 140 * (t - 40))     # pull-out
        cv2.rectangle(img, (int(cx), 430), (int(cx) + 170, 540), (40, 60, 160 + 15 * i), -1)
    px = 640 + 380 * np.sin(2 * np.pi * t / 20); py = 600
    cv2.ellipse(img, (int(px), int(py)), (14, 34), 0, 0, 360, (30, 200, 230), -1)
    car_x = 540 + 85 + (min(600, 140 * (t - 40)) if t > 40 else 0)
    d = abs(car_x - px) if t > 40 else 999
    rho1 = (d / 30.0) - 3.0
    level = 3 if d < 90 else 2 if d < 200 else 0
    vw.write(img); tr.write(json.dumps({"frame": f, "t_s": t, "rho1": rho1, "level": level}) + "\n")
vw.release(); tr.close()
