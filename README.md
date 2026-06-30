# SafeEdge — Formally-Verified Edge Safety Agent

Real-time pedestrian–vehicle safety monitoring built for the **Global AI Hackathon with Qwen Cloud** (Track 5: **EdgeAgent**).

A Jetson Orin NX watches a scene with an Intel RealSense D455 and continuously computes *how safe* it is — not a vague confidence score, but a mathematically rigorous **robustness value (ρ)** from Signal Temporal Logic (STL), evaluated at 30 Hz on-device. A Qwen-powered **cloud brain deployed on Alibaba Cloud** adds multimodal incident reporting, adaptive safety policy, and risk forecasting — without ever sitting in the safety-critical path.

> **For engineering teammates:** this README is the single reference for concept → architecture → deployment → current status. See the rendered system diagram in [`docs/architecture.html`](docs/architecture.html) (self-contained, opens offline).

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
10. [Current status](#10-current-status)
11. [Engineering gotchas (read before you debug)](#11-engineering-gotchas-read-before-you-debug)
12. [Known limitations & framework strategy](#12-known-limitations--framework-strategy)
13. [Testing](#13-testing)
14. [Roadmap](#14-roadmap)
15. [Background & license](#15-background--license)

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
│  D455 (RGB+depth) → YOLOv8s (GPU) → ByteTrack → SignalExtractor (metric d_min,   │
│  v_veh, d_pred) → STLMonitor (ρ₁–ρ₅, rtamt) → InterventionEngine                 │
│       │                                                  │                       │
│       └─► LocalQwen (Ollama, optional)        edge/cloud_client.py (non-blocking) │
└───────────────────────────────────────────────────────────────│────────────────┘
                                                                  │ HTTPS
┌──────────────── ALIBABA CLOUD — Function Compute 3.0 ───────────▼────────────────┐
│  FastAPI (backend/app.py)                                                        │
│    POST /api/state · /api/events · /api/policy/evaluate · GET /api/incidents …   │
│    ├─ Policy Manager   (qwen-max)     → STL param patch ──► back to edge          │
│    ├─ Incident Reporter(qwen-vl-max)  → NL safety report                         │
│    ├─ Risk Forecaster  (qwen-max)     → high-risk windows                        │
│    ├─ Tablestore (events · incidents · ρ-history · forecasts)                    │
│    └─ Live dashboard (WebSocket)                                                 │
└──────────────────────────────────────────│──────────────────────────────────────┘
                                            ▼  cloud/qwen_client.py (DashScope intl)
                              Qwen Cloud: qwen-max · qwen-vl-max · qwen-turbo
```

Full interactive/accessible diagram: **[`docs/architecture.html`](docs/architecture.html)**.

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
| `/healthz` | GET | infra | Liveness probe (reports active store backend) |
| `/ws` | WS | dashboard | Live state stream |
| `/` | GET | dashboard | UI |

Returned STL patches are applied on the edge via `STLMonitor.apply_cloud_params()`.

---

## 5. Hardware & models

| Component | Details |
|---|---|
| Edge device | NVIDIA **Jetson Orin NX 16GB**, JetPack 6.2 (R36.4.7), CUDA 12.6, TensorRT 10.3 |
| Camera | **Intel RealSense D455** (RGB + aligned depth); USB webcam / RTSP / file also supported |
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
│   ├── camera/video_source.py  # RealSenseSource · VideoSource · MockCamera
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
├── docs/architecture.html      # Accessible system diagram (self-contained)
│
├── tools/
│   ├── live_record.py          # D455 + GPU YOLO + distance annotations → MP4
│   ├── benchmark_detector.py   # Model/threshold sweep on a captured clip
│   ├── capture_video.py        # D455/webcam recorder
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

# Live D455 + GPU YOLO + distance overlays → timestamped MP4 (preview on VNC :1)
bash tools/docker_live_record.sh

# Offline model/threshold benchmark on a captured clip
bash tools/docker_benchmark.sh 450
```

Inside the container we `pip install 'numpy<2' ultralytics pyrealsense2` (see gotchas). Benchmarked result: **yolov8s @ conf 0.25 ≈ 19 fps** on the Orin GPU, chosen as the live default.

To run the full pipeline against the **live** cloud backend, set on the Jetson:

```bash
export SAFEEDGE_CLOUD_URL=https://safeedg-backend-nkmqevdhff.ap-southeast-1.fcapp.run
python -m edge.main --source realsense --model yolov8s.pt --conf 0.25
```

---

## 9. Cloud backend — Alibaba Cloud deployment

Full runbook (with the **proof-of-deployment checklist** the hackathon requires) is in **[`deploy/README.md`](deploy/README.md)**. Summary:

```bash
# 1. Build amd64 image and push to ACR
REGION=ap-southeast-1 NAMESPACE=yourns ./deploy/build_and_push.sh
#    → set vars.image in deploy/s.yaml to the printed URI

# 2. Create a Tablestore instance (tables auto-create on first run)

# 3. Deploy to Function Compute (Serverless Devs)
export DASHSCOPE_API_KEY=sk-... TABLESTORE_ENDPOINT=... TABLESTORE_INSTANCE=safeedge
export ALIBABA_CLOUD_ACCESS_KEY_ID=... ALIBABA_CLOUD_ACCESS_KEY_SECRET=...
s deploy

# 4. Verify
curl -s https://<fc-url>/healthz      # {"store":"TablestoreStore", ...}
```

Function Compute scales to zero (pay-per-request); Tablestore reserved throughput is 0 (pay-per-use).

---

## 10. Current status

| Area | Status |
|---|---|
| STL monitor + intervention (edge) | ✅ working, unit-tested |
| D455 RGB-D capture | ✅ live |
| GPU YOLO on Jetson (container) | ✅ ~19 fps, benchmarked |
| Cloud backend (FastAPI + 3 Qwen skills) | ✅ built, tested live (intl endpoint) |
| Datastore — **real Alibaba Tablestore** | ✅ live; write→read verified, durable across restarts |
| Edge → cloud wiring (`cloud_client.py`) | ✅ built, tested live |
| Backend Docker image (amd64) | ✅ builds + runs against **real Tablestore + Qwen** |
| Preflight check (`tools/check_alibaba_creds.py`) | ✅ creds + Tablestore validated |
| Serverless Devs CLI (`s`) | ✅ installed (v3.1.10) |
| Architecture diagram | ✅ `docs/architecture.html` |
| **Backend LIVE on Function Compute** | ✅ deployed; image on Docker Hub; healthz + Qwen incident path verified in production |
| Local Qwen (Ollama) on Jetson | ⏳ pending install |
| Live full-loop demo against deployed backend | ⏳ |
| Submission video + proof recording | ⏳ |

---

## 11. Engineering gotchas (read before you debug)

These cost real time; documented so you don't repeat them.

- **Qwen endpoint is `dashscope-intl`**, not mainland `dashscope`. A hackathon key fails auth against the wrong region. Code defaults to intl; override via `DASHSCOPE_BASE_URL`.
- **`openai==1.51` breaks on `httpx>=0.28`** (`Client.__init__() got unexpected keyword 'proxies'`). Only surfaces in a clean container. Pinned `httpx==0.27.2` in `backend/requirements.txt`.
- **Jetson GPU requires the container.** Host pip torch is CPU-only; CUDA-13 wheels need a newer driver than JetPack 6.2's 12.6. Use `dustynv/l4t-pytorch:r36.4.0`.
- **`numpy<2` inside the container.** torch 2.4 is compiled against NumPy 1.x; NumPy 2 breaks inference (`Numpy is not available`).
- **`pyrealsense2` in the container needs host `libusb`** — mount `/usr/lib/aarch64-linux-gnu/libusb-1.0.so.0` (the pip wheel doesn't bundle it).
- **cv2 preview from a container needs X11 auth** — mount `~/.Xauthority` and set `DISPLAY=:1` (VNC display), else `qt.qpa.xcb: could not connect`.
- **D455 over OpenCV V4L2 returns garbage** (tiny/black frames). Always use `pyrealsense2` directly.
- **`edge/main.py` cloud calls are thread-based, not asyncio.** The 30 Hz loop is synchronous — an earlier `asyncio.create_task` path was latently broken (no running loop). Cloud calls go through `cloud_client.py`'s thread pool.
- **Cloud toggle is `SAFEEDGE_CLOUD_URL`** (presence enables cloud). The old `CLOUD_REPORTING_ENABLED` flag is gone.

---

## 12. Known limitations & framework strategy

We are deliberately treating car-park as a **reference implementation of a reusable pipeline**, not necessarily the final product domain. Two honest limitations of the car-park use case:

1. **Hazard ambiguity.** Geometry alone can't tell whether a car is about to move or merely parked. Mitigations: we gate on tracked velocity + predicted clearance (a parked car gives v≈0). The remaining *intent* gap is exactly where a VLM helps — Qwen-VL can read reverse/brake lights and driver presence. (Idea, not yet built.)
2. **No actuation channel.** Car-park can detect but not warn the driver — it's monitoring/reporting, not active prevention, absent infrastructure (buzzer/bollard/V2X).

**Strategy:** the deliverable is the **domain-swappable framework** (camera → edge → Qwen → dashboard). The swap points are config-driven: detector classes (subjects/objects), signal definitions, STL specs (objectives), skill prompts, dashboard labels — plus a planned explicit **actuation interface**. Candidate higher-fit domains that resolve both limitations (active hazard + real actuation): **warehouse forklift↔pedestrian** (closest swap), construction exclusion zones, rail platform-edge, factory human-robot cells.

> Sequencing decision: complete the end-to-end pipeline with car-park first, then refactor for swappability.

---

## 13. Testing

```bash
pytest tests/test_stl_monitor.py -v                  # STL unit tests, no hardware
python -m tests.mock_scenarios.scenario_a_near_miss  # synthetic end-to-end
PYTHONPATH=$(pwd) python tools/test_backend.py        # backend + live Qwen skills
```

---

## 14. Roadmap

- [ ] Execute Alibaba deploy (ACR → FC → Tablestore) + capture proof recording
- [ ] Reposition D455 (clear line of sight) and run full live loop against deployed backend
- [ ] Install Ollama + local Qwen2.5-VL on the Jetson (3-tier story)
- [ ] Refactor swap points into a single **domain profile** config + **actuation interface**
- [ ] VLM-based intent disambiguation (reverse/brake lights)
- [ ] 3-minute submission video + architecture diagram export

---

## 15. Background & license

Generalises a formal safety-monitoring framework originally developed for a semi-autonomous wheelchair (PhD research; Gazebo/ROS 2 + Jetson Orin HIL validation). The core insight — STL robustness gives a continuous, quantified safety margin rather than a binary alarm — transfers directly to fixed-infrastructure monitoring.

> Nickovic, D. et al. *RTAMT: Online Robustness Monitors from STL.* ATVA 2020.

**License:** MIT
