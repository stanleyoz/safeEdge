# SafeEdge — 3-Minute Demo Video Narration Script

**Target runtime: 3:00.** Structure: `docs/submission/qwencloud_presentation.html` Slide 1 → Slide 2 →
demo clip → Slide 4 (arrow back to the closing recap). Read at a natural ~150 wpm pace; timing has
±10s tolerance.

Demo clip: `docs/submission/pedestrian_vehicle_carpark_sub_section_121525_121544.mp4` (19.12s real
footage, source timestamps 12:15:25–12:15:44).

## Timing map

| Segment | Start | End | Length | Visual |
|---|---|---|---|---|
| Slide 1 | 0:00 | 0:48 | 48s | Edge → Alibaba Cloud |
| Slide 2 | 0:48 | 1:25 | 37s | Qwen Cloud — DashScope intl |
| Demo clip | 1:25 | 2:05 | 40s | 19.12s real footage + holds (see cue table) |
| Slide 4 | 2:05 | 3:00 | 55s | Closing recap — "How Qwen + Alibaba Cloud Power SafeEdge" |

## Narration

### Slide 1 (0:00–0:48)

> Car parks are one of the most dangerous, least-monitored places on the road network — tight
> sightlines, reversing vehicles, pedestrians moving between parked cars. SafeEdge doesn't just
> detect a person near a car. It computes, mathematically, how much safety margin is left, right
> now, and whether it's shrinking. A Jetson Orin NX runs a Signal Temporal Logic safety monitor at
> thirty hertz — camera, YOLO detection, tracking, five formal safety properties evaluated every
> frame, fully offline-capable. The instant that monitor detects a violation, it hands off,
> non-blocking, to a backend running on Alibaba Cloud Function Compute, backed by Tablestore. No
> LLM ever sits in the safety-critical loop.

### Slide 2 (0:48–1:25)

> That Alibaba-hosted backend calls Qwen Cloud through DashScope International, three ways.
> Qwen-VL-Max looks at the actual danger frame and writes a real incident report a human can act
> on. Qwen-Max reasons over event history to retune the safety thresholds live. And a second
> Qwen-Max call turns a stream of incidents into a forecast of when a zone gets risky. A formal
> monitor is precise but mute — Qwen is what makes it explain itself.

### Demo clip (1:25–2:05)

Real telemetry, cued to the actual footage timestamps (clip starts at source time 12:15:25):

| Clip time | Source time | Badge | Narration |
|---|---|---|---|
| +0:00–0:03 | 12:15:25–28 | SAFE | "Watch the robustness value here — completely safe, no pedestrian-vehicle pair in the danger zone." |
| +0:03–0:11 *(hold ~4s)* | 12:15:28–32 | SAFE→WARNING | "Now they enter it together — the predictive spec sees them closing, and it flips to WARNING." |
| +0:11–0:15 | 12:15:32–36 | WARNING | "Vehicle speed jumps to zero-point-seven-five metres per second — a real approach, not sensor jitter." |
| +0:15–0:30 *(hold ~11s on the report)* | 12:15:36 | EMERGENCY | "Clearance falls under threshold — EMERGENCY. That just triggered Qwen-VL-Max, writing this report from the actual frame, live." (let the report text sit on screen, readable) |
| +0:30–0:40 | 12:15:36–44 | →WARNING | "As they separate, severity eases back to WARNING — a continuum, not a binary alarm." |

### Slide 4 (2:05–3:00)

> This is the real differentiator: Alibaba Cloud is the vehicle — serverless compute and storage
> shaped for a workload that's silent for hours, then bursts. Qwen Cloud is the brain riding on it
> — explanation, adaptive policy, forecasting. Strip out Alibaba and there's nowhere safe to run
> Qwen without compromising the edge's real-time guarantee; strip out Qwen and Alibaba is just
> hosting a mute number-store. And we didn't fake a clean demo to get here — our first tagging rule
> false-fired on parked cars from bounding-box jitter; ground-truth review then caught it missing a
> real emergency, so we rebuilt it recall-first: never miss real danger, even if it means more
> false alarms. SafeEdge — Track five, EdgeAgent. Formally verified at the edge, explained by Qwen
> Cloud, running on Alibaba Cloud.
