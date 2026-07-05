# SafeEdge — Formally-Verified Edge Safety Agent

<!--Real-time pedestrian–vehicle safety monitoring built for the **Global AI Hackathon with Qwen Cloud** (Track 5: **EdgeAgent**).-->

A Jetson Orin NX watches a scene with a USB camera (Razer Kiyo X) and continuously computes *how safe* it is — not a vague confidence score, but a mathematically rigorous **robustness value (ρ)** from Signal Temporal Logic (STL), evaluated at 30 Hz on-device. A Qwen-powered **cloud brain deployed on Alibaba Cloud** adds multimodal incident reporting, adaptive safety policy, and risk forecasting — without ever sitting in the safety-critical path.

> **For engineering teammates:** this README is the single reference for concept → architecture → deployment → current status. See the rendered system diagram in [`docs/submission/architecture.html`](docs/submission/architecture.html) (self-contained, opens offline).

---

## Table of contents

1. [Concept](#1-concept)
2. [Architecture](#2-architecture)
3. [STL specifications](#3-stl-specifications)
4. [The edge↔cloud contract](#4-the-edgecloud-contract)
5. [Hardware & models](#5-hardware--models)
6. [Repository layout](#6-repository-layout)
7. [Local development](#7-local-development)
8. [Edge (Jetson) — GPU inference](#8-edge-jetson--gpu-inference)
9. [Cloud backend — Alibaba Cloud deployment](#9-cloud-backend--alibaba-cloud-deployment)
10. [Known limitations & framework strategy](#10-known-limitations--framework-strategy)
11. [Testing](#11-testing)
12. [Background & license](#12-background--license)

---

## 1. Concept

A static camera observes a zone where people and vehicles mix (a car park, in the reference build). The system answers a harder question than "is there a person?": **"is the current scene drifting toward an unsafe state, and by how much?"**

- **Formal, quantified safety.** Instead of a binary alarm, we compute STL robustness ρ — a signed margin. ρ > 0 = spec satisfied (with margin); ρ < 0 = violated (with severity). This is continuous, explainable, and tunable.
- **Predictive, not reactive.** A trajectory extrapolator feeds a *predicted* future clearance `d_pred` into the STL spec, so the monitor warns *before* a violation, not after.
- **Edge-autonomous.** The full safety loop runs on the Jetson. If the network or cloud is down, it keeps protecting people. The cloud adds intelligence opportunistically.
- **Qwen as the reasoning layer.** Three custom cloud skills turn raw ρ signals into human-readable incident reports (multimodal), adaptive STL policy patches, and predictive risk windows.

---

## 2. Architecture

Three tiers. The edge is real-time and offline-capable; the Alibaba-hosted backend is the deployed "brain"; Qwen Cloud provides the models.

```
┌──────────────── EDGE — Jetson Orin NX (30 Hz, offline-capable) ─────────────────┐
│  USB cam (Razer Kiyo X) → YOLOv8s (GPU) → ByteTrack → SignalExtractor (metric   │
│  d_min via homography, v_veh, d_pred) → STLMonitor (ρ₁–ρ₅, rtamt) → Intervention│
│       │                                                  │                      │
│       └─► LocalQwen (Ollama, optional)      edge/cloud_client.py (non-blocking) │
└───────────────────────────────────────────────────────────────│─────────────────┘
                                                                │ HTTPS
┌──────────────── ALIBABA CLOUD — Function Compute 3.0 ─────────▼─────────────────┐
│  FastAPI (backend/app.py)                                                       │
│    POST /api/state · /api/events · /api/policy/evaluate · GET /api/incidents …  │
│    ├─ Policy Manager   (qwen-max)     → STL param patch ──► back to edge        │
│    ├─ Incident Reporter(qwen-vl-max)  → NL safety report                        │
│    ├─ Risk Forecaster  (qwen-max)     → high-risk windows                       │
│    ├─ Tablestore (events · incidents · ρ-history · forecasts)                   │
│    └─ Live dashboard (WebSocket)                                                │
└──────────────────────────────────────────│──────────────────────────────────────┘
                                           ▼  cloud/qwen_client.py (DashScope intl)
                              Qwen Cloud: qwen-max · qwen-vl-max · qwen-turbo
```

Full interactive/accessible diagram: **[`docs/submission/architecture.html`](docs/submission/architecture.html)**.

**Key design principle:** no LLM in the safety-critical path. The deterministic STL monitor decides safety; Qwen explains, reports, and tunes. `cloud_client.py` is fire-and-forget with hard timeouts on a worker thread, so cloud latency or outages can never stall the 30 Hz loop.

---

## 3. STL specifications

Five formal safety properties evaluated every frame (`config/stl_specs.yaml`, hot-swappable at runtime via the Policy Manager skill):

| Spec | Type | Description |
|---|---|---|
| φ1 | Arithmetic | Minimum separation margin: `d_min − clearance_critical` |
| φ2 | Arithmetic | Speed-proximity coupling: vehicle must slow within the proximity zone |
| φ3 | Arithmetic | **Predictive** near-miss: `d_pred − warning_horizon` |
| φ4 | RTAMT (past-time) | Emergency-stop compliance: `(d_min < 1 m) → once[0,2s](v < 0.1 m/s)` |
| φ5 | RTAMT (past-time) | Post-alert clearance: zone must clear within the stop window after an alert |

ρ > 0 satisfied, ρ < 0 violated; magnitude = margin/severity. All thresholds (`proximity_zone`, `stop_window`, …) are patched live by Qwen.

---

## 4. The edge↔cloud contract

The edge posts JSON; the backend never imports the edge package (clean deploy split). Schemas in `backend/models.py`.

| Endpoint | Method | From → To | Purpose |
|---|---|---|---|
| `/api/state` | POST | edge → cloud | Throttled (0.5 s) live state for the dashboard |
| `/api/events` | POST | edge → cloud | Intervention event + JPEG frame; level ≥ 2 triggers a Qwen-VL incident report |
| `/api/policy/evaluate` | POST | edge → cloud | ρ-summary + event counts + current params → returns STL patch |
| `/api/incidents` | GET | dashboard | Recent incident reports |
| `/api/forecast` | GET | dashboard | Latest risk forecast |
| `/api/events` | GET | dashboard | Recent events |
| `/healthz` | GET | infra / proof | Liveness probe **+ deployment self-report** — returns the live FC region/account/instance and active datastore (see §9.1) |
| `/ws` | WS | dashboard | Live state stream |
| `/` | GET | dashboard | UI |

Returned STL patches are applied on the edge via `STLMonitor.apply_cloud_params()`.

---

## 5. Hardware & models

| Component | Details |
|---|---|
| Edge device | NVIDIA **Jetson Orin NX 16GB**, JetPack 6.2 (R36.4.7), CUDA 12.6, TensorRT 10.3 |
| Camera | **USB webcam (Razer Kiyo X)** via OpenCV V4L2, metric distance from a homography ground-plane projection; RTSP / file / RealSense D455 also supported |
| Edge detector | **YOLOv8s** on GPU (~19 fps via container; vs 3–5 fps CPU) |
| Local AI | Qwen2.5-VL via Ollama (on-device scene note; *install pending*) |
| Cloud compute | **Alibaba Cloud Function Compute 3.0** (serverless container) + **Tablestore** + ACR |
| Cloud AI | Qwen Cloud / DashScope **intl** endpoint — `qwen-max` (reasoning), `qwen-vl-max` (vision), `qwen-turbo` (text fallback) |
| Connectivity | Tailscale VPN for remote Jetson access |

Model IDs were empirically verified on the intl endpoint (2026-06-30) and are env-overridable: `QWEN_REASONING_MODEL`, `QWEN_VISION_MODEL`, `QWEN_TEXT_MODEL`. Free tier: 1M in+out tokens / 90 days (ample for our call volume).

---

## 6. Repository layout

```
qwen_cloud/
├── config/
│   ├── stl_specs.yaml          # STL specs + intervention params (hot-swappable)
│   ├── camera_config.yaml      # Video source + homography path
│   └── qwen_config.yaml        # Qwen endpoint + model IDs (intl)
│
├── edge/                       # ── runs on the Jetson ──
│   ├── main.py                 # Orchestration loop (sync, 30 Hz)
│   ├── cloud_client.py         # Non-blocking HTTP → cloud backend (stdlib only)
│   ├── camera/video_source.py  # VideoSource (USB/RTSP/file) · RealSenseSource · MockCamera
│   ├── detection/
│   │   ├── detector.py         # YOLOv8 wrapper (device=auto → CUDA)
│   │   ├── tracker.py          # ByteTrack via supervision
│   │   └── signal_extractor.py # Depth back-projection / homography → d_min, v_veh
│   ├── safety/
│   │   ├── stl_monitor.py      # Dual-track STL monitor (arithmetic + rtamt)
│   │   ├── trajectory.py       # Trajectory extrapolation → d_pred
│   │   └── intervention.py     # Severity engine + hysteresis
│   └── local_ai/qwen_local.py  # Local Qwen via Ollama (≤3 s, offline fallback)
│
├── cloud/                      # ── Qwen skills (imported by the backend) ──
│   ├── qwen_client.py          # DashScope OpenAI-compat client (text + vision)
│   ├── policy_manager.py       # Skill 1: adaptive STL patches      (qwen-max)
│   ├── incident_reporter.py    # Skill 2: multimodal incident report (qwen-vl-max)
│   └── risk_forecaster.py      # Skill 3: risk-window forecast       (qwen-max)
│
├── backend/                    # ── deployed on Alibaba Function Compute ──
│   ├── app.py                  # FastAPI: REST + WebSocket + dashboard + forecast loop
│   ├── models.py               # Pydantic edge↔cloud schemas
│   ├── store.py                # Tablestore (prod) + in-memory (dev) adapters
│   ├── skills.py               # Dict-friendly wrappers over cloud/ skills
│   ├── Dockerfile              # Slim py3.10 container image
│   ├── requirements.txt        # Backend deps (note httpx pin — see gotchas)
│   └── .env.example
│
├── dashboard/static/index.html # Live UI (served by the backend)
│
├── deploy/                     # ── Alibaba Cloud deployment ──
│   ├── s.yaml                  # Serverless Devs (Function Compute 3.0) manifest
│   ├── build_and_push.sh       # Docker buildx (amd64) → ACR
│   └── README.md               # Full deploy runbook + proof-of-deployment checklist
│
├── docs/submission/architecture.html # Accessible system diagram (self-contained)
│
├── tools/
│   ├── live_record.py          # USB cam/D455 + GPU YOLO + distance annotations → MP4
│   ├── benchmark_detector.py   # Model/threshold sweep on a captured clip
│   ├── capture_video.py        # USB cam/D455 recorder
│   ├── calibrate_homography.py # 4-point ground calibration (webcam mode)
│   ├── test_backend.py         # Local backend smoke test (live Qwen if key set)
│   ├── docker_*.sh             # Jetson GPU container runners (see §8)
│
└── tests/
    ├── test_stl_monitor.py
    └── mock_scenarios/scenario_a_near_miss.py
```

---

## 7. Local development

The backend runs with **zero Alibaba dependency** — it falls back to an in-memory store when `TABLESTORE_ENDPOINT` is unset.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt           # edge deps
pip install -r backend/requirements.txt   # backend deps

# Qwen key (gitignored)
echo "DASHSCOPE_API_KEY=sk-..." > backend/.env

# Run the backend locally (in-memory store, real Qwen calls)
PYTHONPATH=$(pwd) python -m uvicorn backend.app:app --host 127.0.0.1 --port 8099

# In another shell — exercise every endpoint + live Qwen skills:
PYTHONPATH=$(pwd) python tools/test_backend.py
```

Point an edge instance at it with `SAFEEDGE_CLOUD_URL=http://127.0.0.1:8099`.

---

## 8. Edge (Jetson) — GPU inference

**GPU on Jetson works only via NVIDIA's container** (host pip/conda torch is CPU-only or driver-mismatched). We use `dustynv/l4t-pytorch:r36.4.0`.

```bash
# one-time: docker needs nvidia runtime (restart after JetPack install)
sudo systemctl restart docker
docker pull dustynv/l4t-pytorch:r36.4.0

# Live USB cam + GPU YOLO + distance overlays → timestamped MP4 (preview on VNC :1)
bash tools/docker_live_record.sh

# Offline model/threshold benchmark on a captured clip
bash tools/docker_benchmark.sh 450
```

Inside the container we `pip install 'numpy<2' ultralytics` (see gotchas; add `pyrealsense2` only if running `--source realsense`). Benchmarked result: **yolov8s @ conf 0.25 ≈ 19 fps** on the Orin GPU, chosen as the live default.

To run the full pipeline against the **live** cloud backend, set on the Jetson:

```bash
export SAFEEDGE_CLOUD_URL=https://safeedg-backend-nkmqevdhff.ap-southeast-1.fcapp.run
python -m edge.main --source webcam --model yolov8s.pt --conf 0.25
# or: bash tools/docker_live.sh --source webcam
```

---

## 9. Cloud backend — Alibaba Cloud deployment

Full runbook (with the **proof-of-deployment checklist** the hackathon requires) is in **[`deploy/README.md`](deploy/README.md)**. Summary:

```bash
# 1. Build amd64 image and push to a public Docker Hub repo
#    (ACR Enterprise costs a monthly fee; Docker Hub is free and the image
#     carries no secrets. FC pulls the public image.)
DOCKERHUB_USER=youruser ./deploy/build_and_push.sh
#    → set vars.image in deploy/s.yaml to docker.io/youruser/safeedge-backend:latest

# 2. Create a Tablestore instance (CU mode, reserved CU = 0; tables auto-create)

# 3. Deploy to Function Compute (Serverless Devs)
export DASHSCOPE_API_KEY=sk-... TABLESTORE_ENDPOINT=... TABLESTORE_INSTANCE=safeedge
export ALIBABA_CLOUD_ACCESS_KEY_ID=... ALIBABA_CLOUD_ACCESS_KEY_SECRET=...
s deploy -t deploy/s.yaml -a default

# 4. Verify (see §9.1)
curl -s https://<fc-url>/healthz
```

Function Compute scales to zero (pay-per-request); Tablestore reserved throughput is 0 (pay-per-use).

---

## 10. Known limitations & framework strategy

We are deliberately treating car-park as a **reference implementation of a reusable pipeline**, not necessarily the final product domain. Two honest limitations of the car-park use case:

1. **Hazard ambiguity.** Geometry alone can't tell whether a car is about to move or merely parked. Mitigations: we gate on tracked velocity + predicted clearance (a parked car gives v≈0). The remaining *intent* gap is exactly where a VLM helps — Qwen-VL can read reverse/brake lights and driver presence. (Idea, not yet built.)
2. **No actuation channel.** Car-park can detect but not warn the driver — it's monitoring/reporting, not active prevention, absent infrastructure (buzzer/bollard/V2X).

**Strategy:** the deliverable is the **domain-swappable framework** (camera → edge → Qwen → dashboard). The swap points are config-driven: detector classes (subjects/objects), signal definitions, STL specs (objectives), skill prompts, dashboard labels — plus a planned explicit **actuation interface**. Candidate higher-fit domains that resolve both limitations (active hazard + real actuation): **warehouse forklift↔pedestrian** (closest swap), construction exclusion zones, rail platform-edge, factory human-robot cells.

> Sequencing decision: complete the end-to-end pipeline with car-park first, then refactor for swappability.

---

## 11. Testing

```bash
pytest tests/test_stl_monitor.py -v                  # STL unit tests, no hardware
python -m tests.mock_scenarios.scenario_a_near_miss  # synthetic end-to-end
PYTHONPATH=$(pwd) python tools/test_backend.py        # backend + live Qwen skills
```
<img width="1729" height="1018" alt="image" src="https://github.com/user-attachments/assets/a621024c-395f-42df-808d-7b5ed963b31e" />

---

## 12. Background & license

Generalises a formal safety-monitoring framework originally developed for a semi-autonomous wheelchair (PhD research; Gazebo/ROS 2 + Jetson Orin HIL validation). The core insight — STL robustness gives a continuous, quantified safety margin rather than a binary alarm — transfers directly to fixed-infrastructure monitoring.

> Nickovic, D. et al. *RTAMT: Online Robustness Monitors from STL.* ATVA 2020.

**License:** MIT
