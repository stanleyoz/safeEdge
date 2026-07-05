# SafeEdge — Submission Writeup

**Track 5: EdgeAgent** | Global AI Hackathon with Qwen Cloud
**Repo:** github.com/stanleyoz/safeEdge · **Live dashboard:** https://stanleyoz.github.io/safeEdge/ · **Cloud proof:** [`/healthz`](https://safeedg-backend-nkmqevdhff.ap-southeast-1.fcapp.run/healthz)
**Alibaba Cloud deployment proof (code):** [`backend/store.py`](https://github.com/stanleyoz/safeEdge/blob/main/backend/store.py#L83) — `TablestoreStore`, the production datastore adapter used by the deployed backend; see also [`deploy/s.yaml`](https://github.com/stanleyoz/safeEdge/blob/main/deploy/s.yaml) (Function Compute 3.0 deployment spec).

---

## Built and significantly updated during the Submission Period

This project existed prior to the Hackathon Submission Period as an edge-only pedestrian safety concept. During the Submission Period we: (1) designed and deployed the entire Alibaba Cloud backend (Function Compute 3.0 + Tablestore) and the three Qwen Cloud skills from scratch; (2) rebuilt the danger-detection logic multiple times against operator ground-truth review, replacing an absolute-speed gate with a validated closing-speed + recall-first rule (see "Honest engineering" below); (3) added the live operator dashboard, frame-synchronized clock, and GPIO-failsafe architecture hook. All of this is visible in the commit history on the `main` branch.

---

## The problem

Car parks are one of the most common places for pedestrian–vehicle collisions: tight sightlines, reversing vehicles, distracted pedestrians moving between parked cars. Existing "AI camera" products answer *"is there a person near a car?"* — a binary, unexplainable alarm that either fires constantly (false alarms operators learn to ignore) or misses the moment that matters.

SafeEdge asks a better question: ***how much safety margin is left, right now, and is it shrinking?*** — and answers it with a real formal-methods safety monitor, not a confidence score.

## What we built

A three-tier system: a **Jetson Orin NX** watches a live camera feed and runs a **Signal Temporal Logic (STL)** safety monitor at the edge, in real time, fully offline-capable. When the formal monitor detects a violation, it hands off — over a non-blocking HTTP call — to a **backend running on Alibaba Cloud Function Compute**, which uses **Qwen Cloud** to turn the raw event into a human-readable incident report, adapt the safety thresholds, and forecast risk windows. The result streams to a live operator dashboard.

```
Jetson Orin NX (edge, 30 Hz)          Alibaba Cloud (FC 3.0 + Tablestore)         Qwen Cloud (DashScope intl)
  camera → YOLOv8s → ByteTrack   →      FastAPI backend                    →       qwen-vl-max  (incident reports)
  → STL monitor (ρ₁–ρ₅)                  · event log (Tablestore)                  qwen-max     (policy + forecast)
  → intervention engine                  · Qwen skills                             qwen-turbo   (text fallback)
                                          · live dashboard (Pages)
```

**No LLM sits in the safety-critical path.** The deterministic STL monitor decides SAFE / AWARENESS / WARNING / EMERGENCY on every frame; Qwen explains, reports, and tunes — opportunistically, never blocking the loop. If the network or Alibaba Cloud is unreachable, the Jetson keeps protecting people on its own.

**This is the actual reason the intelligence has to live on the edge device, not just in the cloud.** `cloud_client.py` is fire-and-forget with a hard timeout — a lost connection to Qwen Cloud or Alibaba Cloud degrades the system to "no incident narration, no policy tuning, no dashboard visibility," but the level computation itself never stops, because it never depended on the network in the first place. That local intervention event is also the natural trigger point for a **physical failsafe** — a GPIO-driven buzzer or floodlight wired directly to the Jetson, firing off the same `on_event` callback that pushes to the cloud, so a real danger still gets a physical, human-perceptible response even if every network link is down. We haven't wired the hardware yet (roadmap item, not built for this submission), but the architecture already has the exact hook for it — the callback that would drive a GPIO pin is the same one already driving the cloud report. That's the case for an edge deployment instead of a pure cloud-camera product: safety cannot be allowed to depend on connectivity.

## Why Qwen Cloud: closing the gap between "correct" and "useful"

A formal STL monitor is precise but **mute**. It outputs `ρ1 = -1.6`, `φ2 violated` — mathematically rigorous, operationally useless to a security guard, facility manager, or insurance auditor who needs to know *what happened and what to do about it*. That translation gap is exactly what most "AI safety camera" products fake with a canned string ("Alert: person detected"). We didn't want to fake it, so we needed a model that can (a) *see* the actual frame and describe the real scene, and (b) *reason* over a history of numeric safety data to adjust policy and forecast risk — not just caption an image.

That's a genuinely different capability from the detector running on the Jetson (YOLOv8s finds pixels; it doesn't understand a scene or write a coherent safety narrative). Qwen Cloud is used three ways, each load-bearing, all verified live against real camera frames — not scripted demo strings:

- **`qwen-vl-max` — Incident Reporter (multimodal).** Takes the actual frame at the moment of violation plus the structured ρ data and writes a real incident report — *"a white sedan close to a pedestrian near the edge of the paved area."* This is the artifact a human actually reads after an incident; without a vision-language model this step is either skipped or replaced by a templated string with no scene understanding.
- **`qwen-max` — Policy Manager (reasoning over structured history).** Given recent event counts and current STL parameters, returns a live parameter patch (e.g. tightening `proximity_zone`) — closing the loop from "the monitor is firing too often/too rarely" to "here's a better threshold," without a human manually re-tuning YAML.
- **`qwen-max` — Risk Forecaster (temporal reasoning).** Aggregates event history into predicted high-risk time windows — turning a stream of individual incidents into an operational insight ("this zone is riskiest 5–6pm") that no single-frame model could produce.

**Counterfactual:** without Qwen Cloud, SafeEdge is a correct-but-silent alarm system — accurate ρ values with no explanation, no adaptation, and no forecasting. Qwen is what turns a formally verified detector into a system a human can actually act on.

## Why Alibaba Cloud: the right economics and locality for an event-driven safety system

A car-park safety monitor doesn't generate steady load — it's **silent for hours, then bursts** when something dangerous happens. That access pattern is a poor fit for always-on infrastructure (a permanently running VM/server billed whether or not anything happens) and a good fit for serverless: pay only when an event actually fires.

- **Function Compute 3.0** hosts the entire cloud backend (FastAPI, the three Qwen skills, dashboard) as a scale-to-zero container — idle cost is effectively $0, and it scales automatically the moment a real incident bursts events. `/healthz` self-reports the live FC region/account/instance identity, giving reproducible, verifiable proof of deployment (not just a screenshot).
- **Tablestore** (serverless NoSQL) persists events, incidents, ρ-history, and latest state with the same pay-per-use economics — no provisioned capacity to size or pay for up front.
- **Locality with DashScope.** The backend calling Qwen Cloud (DashScope intl) from the same cloud region keeps the multimodal/reasoning calls off the safety-critical Jetson entirely — the edge device never needs network egress to an LLM provider to keep running; it only optionally reports outward, non-blockingly, to Alibaba.

**Counterfactual:** without Alibaba Cloud, we'd either run a permanently-billed server for a system that's idle 95%+ of the time, or push Qwen calls onto the Jetson itself — coupling a safety-critical 30 Hz loop to internet latency and a third-party API, which is precisely the architecture risk we designed the system to avoid.

## How the two compound

Alibaba Cloud is the *vehicle* — serverless compute and storage shaped to match a bursty, safety-critical workload. Qwen Cloud is the *brain* riding on that vehicle — multimodal explanation, adaptive policy, and forecasting. Neither replaces the other, and neither is optional: strip out Alibaba Cloud and there's nowhere appropriate to run Qwen without compromising the edge's safety guarantees; strip out Qwen Cloud and Alibaba Cloud is just hosting a mute number-store. Together they turn a formally-correct local monitor into a system that explains itself, tunes itself, and predicts risk — the actual differentiator versus a bare detection-and-alarm product.

## The formal safety layer

Five STL properties (`config/stl_specs.yaml`), evaluated every frame via `rtamt`, each returning a signed robustness value ρ (positive = safe with margin, negative = violated with severity — not a black-box score):

| Spec | Type | What it checks |
|---|---|---|
| φ1 | Arithmetic | Minimum pedestrian–vehicle separation, gated on genuine closing motion (not just proximity) |
| φ2 | Arithmetic | Speed must fall as a vehicle nears the pedestrian zone |
| φ3 | Arithmetic | **Predictive** — trajectory extrapolation flags a closing pair *before* the gap closes |
| φ4 | RTAMT past-time | Emergency-stop compliance: within 1 m, the vehicle must actually stop |
| φ5 | RTAMT past-time | Post-alert clearance before the zone is declared safe again |

Thresholds are hot-swappable at runtime — the Policy Manager patches them live without a redeploy.

## Honest engineering: the tuning story

Rather than curate a clip that always works, we deliberately stress-tested the tagging logic against ground-truth review of our own recorded footage — and it's worth reporting what we found, because it's the more interesting engineering result:

1. **First iteration** gated EMERGENCY on absolute vehicle speed. It false-fired on **parked cars**, because monocular position jitter (bounding-box redraw noise, amplified by the ground-plane homography) reads as ~0.5–1 m/s of phantom velocity even on a stationary vehicle.
2. **Second iteration** gated on the *pair's* radial closing speed instead of absolute speed — this correctly demoted static-car proximity to WARNING, but during operator ground-truth review it also **missed a genuine emergency**: a pedestrian 1.4 m from a car doing 0.68 m/s, where the derived closing signal happened to read near zero for a couple of frames.
3. **Final rule (recall-first):** EMERGENCY fires on proximity **and** (closing motion **or** the vehicle's own speed alone) — deliberately biased toward false alarms over missed danger, per the explicit safety principle that *a missed emergency is worse than a false one*. We proved this fix correct with exact per-frame signal logging on the transition that had failed, not just re-running and eyeballing the result.
4. **Root cause, not just a threshold:** the underlying noise is bounding-box redraw jitter feeding a homography that amplifies pixel wobble into phantom velocity at range. We added position smoothing at the signal-extraction layer (helps, doesn't fully solve it) and benchmarked three additional detector options — YOLO11m, YOLO11x, RT-DETR-L — against the same ground truth. None cleanly solved it either; the real fix is geometric (below).

## Known limitation & roadmap

**Single monocular camera, cars moving mostly along the camera's own axis, badly under-measures radial (toward/away) velocity** — this is a scene-geometry limit, not a software bug, and no amount of filter tuning removes it cleanly. The honest fix:

- **Camera reframing** — position the danger lane *across* the field of view, where monocular pixel motion measures speed reliably.
- **Multiple cameras / stereo** — resolves depth and velocity directly, removing the radial blind spot entirely.
- **NVIDIA DeepStream + TrafficCamNet/PeopleNet + NvDCF** (TensorRT-native, purpose-built for traffic/parking scenes, Kalman-smoothed output boxes) — a stronger detector+tracker stack than our current YOLOv8s+ByteTrack MVP, deployable on the same Orin NX 16GB.

We're shipping the MVP with this limitation documented rather than over-fitting the current single-camera setup to one test clip.

## Demo

The submission video is real edge output — the Jetson pipeline replaying our recorded car-park footage through the live formal-safety monitor, posting real events to the Alibaba-hosted backend, generating real `qwen-vl-max` incident reports, visible on the live dashboard with a frame-synchronized clock so every tag can be cross-checked against the footage in real time.

## Cross-track relevance

- **Track 4 (Autopilot):** the Policy Manager / Risk Forecaster / Incident Reporter loop is a self-adjusting system that monitors its own ρ drift and hot-swaps STL parameters — a self-healing control loop.
- **Track 3 (Agent Society):** the edge (fast, reactive, formally verified) and the cloud (slow, deliberative, LLM-reasoning) form a two-agent system with a clean, well-defined handoff contract.

## License & third-party components

This project is MIT-licensed (see [`LICENSE`](../LICENSE)). It uses [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0) for edge object detection — used here under AGPL terms, satisfied by this repository itself being fully open-source and publicly available. All other dependencies (FastAPI, rtamt, supervision/ByteTrack, Alibaba Cloud SDKs, DashScope/OpenAI SDK) are permissively licensed.
