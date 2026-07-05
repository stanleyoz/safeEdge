# SafeEdge — Project Story

## Inspiration

Car parks are a strange blind spot in road safety: tight sightlines, reversing vehicles, pedestrians weaving between parked cars, and — unlike a street — almost no formal traffic control at all. Watching a near-miss in a car park was the spark. The commercial "AI safety camera" products we looked at all answered the same shallow question — *"is there a person near a car, yes or no?"* — a binary alarm that either nags constantly until operators mute it, or fires too late to matter.

We wanted to ask a better question: **how much safety margin is left, right now, and is it shrinking?** That reframing — from detection to a continuous, signed *robustness value* — is the idea the whole project is built around, and it demanded an architecture rigorous enough to trust and fast enough to run at the edge, paired with something smart enough to explain and adapt.

## What we learned

The biggest lesson was how different "detecting" and "understanding" really are. Our edge detector (YOLOv8s + ByteTrack) is excellent at finding pixels — a bounding box, a track ID — but has no concept of a scene or a threshold that should change over time. That gap is exactly what Qwen Cloud closes: `qwen-vl-max` looks at the actual danger frame and writes a real sentence a human can act on; `qwen-max` reasons over event history to retune safety thresholds live; a second `qwen-max` call turns a stream of incidents into a forecast of *when* a zone gets risky — reasoning over structured safety data, not captioning an image.

We also learned, the hard way, that formal correctness on paper doesn't survive contact with real camera noise, and that serverless economics (Alibaba Function Compute + Tablestore) fit a system that's silent for hours and then bursts far better than an always-on server.

## How we built it

SafeEdge is a three-tier pipeline. On a **Jetson Orin NX**, a camera feed runs through YOLOv8s and ByteTrack, then a **Signal Temporal Logic (STL)** monitor (`rtamt`) evaluates five formal safety properties every frame — 30 Hz, offline-capable — and an intervention engine turns the robustness values into SAFE / WARNING / EMERGENCY. Critically, **no LLM sits in the safety-critical path**: the deterministic monitor decides the level on its own, and only *afterward* does a non-blocking, hard-timeout call hand the event to a backend on **Alibaba Cloud Function Compute 3.0**, backed by **Tablestore**. That backend calls the three Qwen Cloud skills and streams results to a live operator dashboard. If the network or cloud is unreachable, the Jetson keeps protecting people on its own — the cloud adds explanation and adaptation, never protection itself.

## Challenges we faced

The hardest problem wasn't the LLM integration — it was making the edge signal trustworthy. We actually started with an **Intel RealSense D455 depth camera**, precisely because real depth sidesteps the pixel-jitter problem entirely — no homography, no phantom velocity from bounding-box noise. But its usable depth range (~15m) fell far short of our real car-park distances (20–57m); past that, the depth channel is unreliable. Range mattered more than precision-at-range, so we pivoted to a plain USB webcam with ground-plane homography — cheaper, unlimited range, but reintroducing exactly the jitter problem the depth camera would have solved. That trade-off is the direct cause of the tagging bug below, not an oversight we patched around later.

Our first EMERGENCY rule gated on absolute vehicle speed and false-fired constantly on **parked cars**, because monocular bounding-box jitter, amplified through the homography, reads as real velocity on a stationary vehicle. Gating on the *pair's* closing speed instead fixed that, but ground-truth review then caught it **missing a genuine emergency**, where the derived closing signal briefly read near zero. We rebuilt again, recall-first: EMERGENCY fires on proximity *and* (closing motion *or* the vehicle's own speed), deliberately biased toward false alarms — a missed emergency is strictly worse than a false one. We proved the fix with exact per-frame signal logs on the transition that had failed, rather than re-running and eyeballing it.

We also hit a geometry limit no tuning fully removes: a monocular camera badly under-measures velocity for a vehicle moving straight along its own axis. Three alternative detectors, benchmarked against the same ground truth, didn't solve it either — it isn't a detector problem. We're shipping with this documented honestly, with the real fix (camera reframing, multi-camera stereo) on the roadmap, rather than overfitting to one clip to make the demo look cleaner than the system actually is.
