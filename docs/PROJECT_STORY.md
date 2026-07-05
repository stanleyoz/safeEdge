# SafeEdge — Project Story

## Inspiration

Car parks are a strange blind spot in road safety: tight sightlines, reversing vehicles, pedestrians weaving between parked cars, and — unlike a street — almost no formal traffic control at all. Watching a near-miss in a car park was the spark: a driver reversing out of a bay, a pedestrian cutting between cars, both invisible to each other until the last second. The commercial "AI safety camera" products we looked at all answered the same shallow question — *"is there a person near a car, yes or no?"* — a binary alarm that either nags constantly until operators mute it, or fires too late to matter.

We wanted to ask a better question: **how much safety margin is left, right now, and is it shrinking?** That reframing — from detection to a continuous, signed *robustness value* — is the idea the whole project is built around, and it demanded an architecture rigorous enough to trust and fast enough to run at the edge, paired with something smart enough to explain and adapt.

## What we learned

The biggest lesson was how different "detecting" and "understanding" really are. Our edge detector (YOLOv8s + ByteTrack) is excellent at finding pixels — a bounding box, a track ID — but has no concept of a scene, a narrative, or a threshold that should change over time. That gap is exactly what Qwen Cloud closes: `qwen-vl-max` looks at the actual danger frame and writes a real sentence a human can act on; `qwen-max` reasons over event history to retune safety thresholds live; a second `qwen-max` call turns a stream of incidents into a forecast of *when* a zone gets risky. None of that is a caption — it's reasoning over structured safety data, a fundamentally different job from object detection.

We also learned, the hard way, that formal correctness on paper doesn't survive contact with real camera noise, and that serverless economics (Alibaba Cloud Function Compute + Tablestore) are a genuinely better fit than an always-on server for a system that's silent for hours and then bursts.

## How we built it

SafeEdge is a three-tier pipeline. On a **Jetson Orin NX**, a camera feed runs through YOLOv8s and ByteTrack, then a **Signal Temporal Logic (STL)** monitor (`rtamt`) evaluates five formal safety properties every frame — 30 Hz, fully offline-capable — and an intervention engine turns the robustness values into SAFE / WARNING / EMERGENCY. Critically, **no LLM sits in the safety-critical path**: the deterministic STL monitor decides the safety level on its own, and only *afterward* does a non-blocking, hard-timeout call hand the event off to a backend on **Alibaba Cloud Function Compute 3.0**, backed by **Tablestore**. That backend calls the three Qwen Cloud skills and streams results to a live operator dashboard. If the network or cloud is unreachable, the Jetson keeps protecting people on its own — the cloud adds explanation and adaptation, never protection itself.

## Challenges we faced

The hardest problem wasn't the LLM integration — it was making the edge signal trustworthy. Our first EMERGENCY rule gated on absolute vehicle speed and false-fired constantly on **parked cars**, because monocular bounding-box jitter, amplified through the ground-plane homography, reads as real velocity even on a stationary vehicle. We rebuilt the rule to gate on the *pair's* closing speed instead — that fixed the false positives, but ground-truth review then caught it **missing a genuine emergency**, where the derived closing signal briefly read near zero. We rebuilt again, recall-first this time: EMERGENCY fires on proximity *and* (closing motion *or* the vehicle's own speed), deliberately biased toward false alarms — a missed emergency is strictly worse than a false one. We proved the fix with exact per-frame signal logs on the exact transition that had failed, rather than re-running and eyeballing it.

We also hit a geometry limit no amount of tuning can fully remove: a single monocular camera badly under-measures velocity for a vehicle moving straight along the camera's own axis. We benchmarked three alternative detectors against the same ground truth — none solved it, because it isn't a detector problem. We're shipping with this limitation documented honestly, with the real fix (camera reframing, multi-camera stereo) on the roadmap, rather than overfitting to one test clip to make a demo look cleaner than the system actually is.
