# Replay dashboard: STL vs V-JEPA 2.1

`dashboard/static/compare.html` replays a recorded clip and shows, frame by
frame, the alert the STL monitor raised next to the alert V-JEPA 2.1 surprise
would have raised. It runs on the workstation (ae4090), not on the Jetson: the
V-JEPA scores were computed there on the RTX 4090, and the clip stays local.

For the live dashboard, see [Live dashboard](#live-dashboard) at the end.

## Run it

From the repo root on ae4090:

```bash
cd ~/projects/safeEdge
python3 -m http.server 8000
```

Open <http://ae4090:8000/dashboard/static/compare.html> (or
`http://localhost:8000/...` on the machine itself). Stop the server with
Ctrl+C.

The page loads the July clip by default:

| What | Path |
| --- | --- |
| Data (trace + V-JEPA runs) | `research/vjepa-surprise/results/compare/segment_0455_to_end.json` (in git) |
| Video | `data/segment_0455_to_end.h264.mp4` (git-ignored; never leaves the machine) |

Without the video the page still runs: boxes and surprise maps are drawn on a
blank frame. Any other data file or clip can be opened with the **Video**
button, the file pickers, or by dragging files onto the page.

## Controls

| Action | How |
| --- | --- |
| Play / pause | **Play** button or Space (1× is real time) |
| Step one frame | **−1 f** / **+1 f** or Left / Right arrow |
| Jump to a time | Click or drag (scrub) along the timeline, click an event-log entry, or add `?t=<seconds>` to the URL |
| Change scorer | V-JEPA panel: ViT-g predictor, ViT-B + ridge, or the pixel frame-difference baseline |
| Change alert hold | V-JEPA panel: windows ŝ must stay above 1 before V-JEPA raises WARNING (default 2, about 1 s) |
| Overlays | **boxes** and **surprise map** checkboxes |

The same clip replayed with the 21 Nov camera calibration is at
`?data=../../research/vjepa-surprise/results_calib-2026-11-21/compare/segment_0455_to_end.json`.
With the calibration and the STL signal fix:
`?data=../../research/vjepa-surprise/results_calib-2026-11-21_signals/compare/segment_0455_to_end.json`.
With parked cars as a minor concern (WARNING needs a moving car near the
pedestrian):
`?data=../../research/vjepa-surprise/results_calib-2026-11-21_moving/compare/segment_0455_to_end.json`.
Adding EMERGENCY on time to collision with a moving car (current settings):
`?data=../../research/vjepa-surprise/results_calib-2026-11-21_ttc/compare/segment_0455_to_end.json`.

URL parameters: `?data=<url>` and `?video=<url>` load other files; `?t=` sets
the start time. For example
`compare.html?t=125` opens on the SUV pulling out beside a pedestrian.

## Reading the page

The three header badges:

- **STL**: the level the live dashboard showed. The raw per-frame STL level
  goes through the edge loop's own hysteresis (raise after 3 frames, lower
  after 23), so this reproduces the July demo's 24 events.
  While a change is pending the badge says so: **"WARNING · hold 2.3 s"**
  counts down the 3.0 s an alert is held after the raw level drops (so a
  momentary detection dropout cannot clear it), and **"· confirming"** marks
  the 0.4 s before a rise. During a hold the ρ bars can all be green: they
  show the current frame, the badge shows the held level. The page re-runs
  the hysteresis and checks it matches the engine's output before showing
  countdowns.
- **V-JEPA**: WARNING when ŝ = score/τ stays above 1 for the chosen number of
  windows; WARMING UP for the first 3.77 s, before the first window. V-JEPA
  has no notion of distance, so it never raises EMERGENCY.
- **Fused**: max(STL, V-JEPA). V-JEPA can only tighten: it raises the level
  to WARNING but never lowers STL.

The timeline lanes, top to bottom: STL as shown (lower half dimmed during a hold), raw STL, ŝ with τ dashed,
V-JEPA, fused. The event log merges both sources and states for each V-JEPA
alert whether an STL WARNING+ followed within 8 s.

Moments worth showing on the July clip (the notes describe the July
settings; with current settings 33 s, 73 s and 110 s are EMERGENCY, E6 is
WARNING and the rest stay at AWARENESS):

| `?t=` | What happens |
| --- | --- |
| 24 | E2: pedestrian walks up to a parked car; STL EMERGENCY with no vehicle moving |
| 68 | E3: a car drives along the lane with a pedestrian in it (genuine conflict) |
| 92 | E4: walker crossing the lawn; EMERGENCY from the compressed homography |
| 32 | A car drives along the lane while a pedestrian walks into its path (EMERGENCY) |
| 72 | The same again, slower approach (EMERGENCY) |
| 110 | E5: a car passes a pedestrian at 2 m/s (EMERGENCY) |
| 125 | E6: SUV pulls out beside a pedestrian (genuine conflict) |
| 170–210 | Quiet stretch: surprise stays below τ, no V-JEPA alarms |

## Rebuilding the data

Only needed after new V-JEPA runs or a new trace. Two steps:

**1. Browser-playable video.** Browsers cannot play OpenCV's `mp4v` captures,
so make an H.264 copy. `-r` re-times it to the true capture fps, so the
header no longer claims 15 fps for a 7.7 fps clip:

```bash
ffmpeg -r 7.7 -i data/segment_0455_to_end.mp4 -c:v libx264 -crf 23 \
    -pix_fmt yuv420p -movflags +faststart -an data/segment_0455_to_end.h264.mp4
```

No system ffmpeg? `pip install imageio-ffmpeg` bundles one
(`python -c "import imageio_ffmpeg as i; print(i.get_ffmpeg_exe())"`).

**2. Pack the trace and runs:**

```bash
python tools/build_compare_data.py \
    --trace research/vjepa-surprise/results/traces/segment_0455_to_end.trace.jsonl \
    --fps 7.7 --video data/segment_0455_to_end.h264.mp4 --downgrade-hold 23 \
    --out research/vjepa-surprise/results/compare/segment_0455_to_end.json \
    research/vjepa-surprise/results/{vitg_predictor,vitb_ridge_crossseason,framediff}
```

The trace comes from `python -m edge.main --source file --trace ...` and each
run directory from `tools/vjepa_surprise.py`; see
[`research/vjepa-surprise/README.md`](../research/vjepa-surprise/README.md)
for both.

For a new clip, `--fps` is the true capture fps (frames divided by wall-clock
duration, or from the `.frames.csv` that `tools/capture_video.py` now writes),
and `--downgrade-hold` should match the `SAFEEDGE_DOWNGRADE_HOLD` the live
run used.

## Live dashboard

The live dashboard runs on the Jetson, not here.

| Where | URL |
| --- | --- |
| Jetson, over Tailscale | `http://eliza-jetson:8080` |
| Public, via Tailscale Funnel (443 → 8080) | <https://eliza-jetson.tail52ce98.ts.net> |
| GitHub Pages viewer (same feed over WebSocket) | `docs/index.html` |

It shows STL only; V-JEPA is not live, since the ViT-g scorer needs a desktop
GPU (268 ms per window on the RTX 4090) rather than the Orin NX.
