# SafeEdge — Car Park Pedestrian-Vehicle Safety Monitor

A real-time safety monitor that watches a car park with a standard RGB camera and raises formal, quantified alerts when pedestrians and vehicles come too close. Built for the **Global AI Hackathon Series with Qwen Cloud** (July 2026).

---

## What it does

A static camera above a car park feeds into a pipeline running on a Jetson Orin Nano edge device. The system continuously computes how safe the scene is — not with a vague confidence score, but with a mathematically rigorous **robustness value** derived from Signal Temporal Logic (STL). When a near-miss is predicted or a violation occurs, it alerts in real time and sends a Qwen Vision report to a live dashboard.

The novel contribution is **Predictive STL**: a trajectory extrapolator feeds a predicted future clearance signal (`d_pred`) into the STL specification, so the system warns *before* a violation occurs rather than after.

---

## Architecture

```
RGB camera (USB / RTSP WiFi)
    │
    ▼  30fps
[YOLO v8-nano detection]
    │  bounding boxes
    ▼
[ByteTrack tracker]
    │  track IDs + bbox history
    ▼
[Homography signal extractor]          ← ground-plane calibration (one-time)
    │  d_min (m), v_veh_max (m/s)
    ▼
[Trajectory predictor]
    │  d_pred — predicted clearance 4s ahead
    ▼
[STL Monitor  φ1–φ5]                  ← dual-track: arithmetic + RTAMT
    │  ρ1..ρ5 robustness values
    ▼
[Intervention engine + hysteresis]
    │
    ├──► [Local Qwen2.5-3B via Ollama]  ← real-time interpretation, <100ms, offline-capable
    │
    └──► [Qwen Cloud skills]            ← async, non-blocking
              ├─ Skill 1: Policy Manager   (qwen-plus)    — hot-swap STL parameters
              ├─ Skill 2: Incident Reporter (qwen-vl-plus) — vision-based safety log entry
              └─ Skill 3: Risk Forecaster  (qwen-turbo)   — trend analysis + recommendations
                   │
                   ▼
              [FastAPI dashboard — live robustness bars + event log]
```

No depth sensor required. The homography maps image pixels to ground-plane metres from a one-time 4-point calibration.

---

## STL Specifications

Five formal safety properties run every frame:

| Spec | Type | Description |
|---|---|---|
| φ1 | Arithmetic | Minimum separation margin: `d_min - clearance_critical` |
| φ2 | Arithmetic | Speed-proximity coupling: vehicle must slow within zone |
| φ3 | Arithmetic | **Predictive** near-miss: `d_pred - warning_horizon` |
| φ4 | RTAMT (past-time) | Emergency stop compliance: `(d_min < 1m) implies once[0,2s](v < 0.1m/s)` |
| φ5 | RTAMT (past-time) | Post-alert clearance: zone must clear within 15s of alert |

Robustness ρ > 0 means the property is satisfied; ρ < 0 means violation. The magnitude quantifies *how much* margin or violation exists. All parameters are hot-swappable via Qwen Cloud policy updates at runtime.

---

## Hardware

| Component | Details |
|---|---|
| Edge device | NVIDIA Jetson Orin Nano 8GB (JetPack 7.2) |
| Camera | Any USB webcam, RTSP WiFi camera, or 4K IP camera |
| Connectivity | Tailscale VPN for remote access |
| Local AI | Qwen2.5-3B via Ollama (fits in Orin RAM budget) |
| Cloud AI | Alibaba Cloud DashScope — qwen-plus, qwen-vl-plus, qwen-turbo |

---

## Quick start (laptop / mock mode — no hardware needed)

```bash
git clone git@github.com:stanleyoz/safeEdge.git
cd safeEdge
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the synthetic near-miss scenario
python edge/main.py --mock
```

The mock generates a pedestrian + vehicle convergence scenario. Open `http://localhost:8080` to see the dashboard with live robustness bars and event log.

---

## Setup with a real camera

### 1. Calibrate the homography (one-time, ~10 minutes)

Place 4 tape markers on the ground at known positions (measure with a tape measure). Then:

```bash
python tools/calibrate_homography.py --source 0        # USB webcam
python tools/calibrate_homography.py --source "rtsp://192.168.1.x:554/stream"
```

Click each marker in order, enter the real-world coordinates when prompted. Saves `config/homography.npy`.

### 2. Configure the camera source

Edit `config/camera_config.yaml`:

```yaml
source: 0                                    # USB device index
# source: "rtsp://192.168.1.50:554/stream"  # WiFi/IP camera
target_fps: 15.0
homography_file: "config/homography.npy"
```

### 3. Set up Qwen Cloud credentials

```bash
cp .env.example .env
# Edit .env and add your DashScope API key
```

```env
DASHSCOPE_API_KEY=sk-...
LOCATION_LABEL="Level 2 Car Park"
CLOUD_REPORTING_ENABLED=true
```

Get a free trial API key at [dashscope.aliyuncs.com](https://dashscope.aliyuncs.com).

### 4. Run

```bash
python edge/main.py
```

---

## Jetson Orin deployment

```bash
# On the Orin (JetPack 7.2)
git clone git@github.com:stanleyoz/safeEdge.git
cd safeEdge
pip install -r requirements.txt

# Install Ollama and pull the local model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b

# Smoke test with mock
python edge/main.py --mock

# Run with a WiFi camera
python edge/main.py
```

YOLO will automatically use CUDA on the Orin GPU. The full pipeline (detect + track + STL + local Qwen) runs comfortably within the 8GB RAM budget.

---

## Project structure

```
safeEdge/
├── config/
│   ├── stl_specs.yaml          # STL specs + intervention params (hot-swappable)
│   ├── camera_config.yaml      # Video source + homography path
│   └── qwen_config.yaml        # Qwen Cloud + local Ollama model config
│
├── edge/
│   ├── main.py                 # Main orchestration loop
│   ├── camera/
│   │   └── video_source.py     # VideoSource (USB/RTSP/file) + MockCamera
│   ├── detection/
│   │   ├── detector.py         # YOLOv8-nano wrapper
│   │   ├── tracker.py          # ByteTrack via supervision
│   │   └── signal_extractor.py # Homography → d_min, v_veh_max, TrackedObjects
│   ├── safety/
│   │   ├── stl_monitor.py      # Dual-track STL monitor (core)
│   │   ├── trajectory.py       # Linear trajectory extrapolation → d_pred
│   │   └── intervention.py     # Intervention engine with hysteresis
│   └── local_ai/
│       └── qwen_local.py       # Local Qwen2.5-3B via Ollama
│
├── cloud/
│   ├── qwen_client.py          # DashScope OpenAI-compat client (text + vision)
│   ├── policy_manager.py       # Skill 1: adaptive STL parameter updates
│   ├── incident_reporter.py    # Skill 2: Qwen-VL vision incident reports
│   └── risk_forecaster.py      # Skill 3: trend analysis + recommendations
│
├── dashboard/
│   ├── app.py                  # FastAPI + WebSocket backend
│   └── static/index.html       # Dark-theme SPA: camera feed + ρ bars + event log
│
├── tests/
│   ├── test_stl_monitor.py             # 11 unit tests (all passing)
│   └── mock_scenarios/
│       └── scenario_a_near_miss.py     # End-to-end synthetic validation
│
└── tools/
    └── calibrate_homography.py         # Interactive 4-point ground calibration
```

---

## Running tests

```bash
pytest tests/test_stl_monitor.py -v          # 11 unit tests, no hardware required

python -m tests.mock_scenarios.scenario_a_near_miss   # end-to-end scenario
```

---

## Contributing

Contributions welcome — especially:

- **Scenario B / C** — add more mock scenarios to `tests/mock_scenarios/` (e.g. vehicle reversing blind, pedestrian running)
- **YOLO class tuning** — better confidence thresholds or class filtering for car park objects
- **Dashboard** — additional signal plots, historical ρ trend chart
- **Jetson optimisation** — TensorRT export for YOLOv8, INT8 quantisation
- **Qwen skills** — improve prompts in `cloud/policy_manager.py` and `cloud/risk_forecaster.py`

Fork, branch off `main`, open a PR. The STL monitor unit tests are the acceptance gate — keep them green.

---

## Background

This project generalises a formal safety monitoring framework originally developed for a semi-autonomous wheelchair (PhD research, Gazebo/ROS 2 + Jetson Orin HIL validation). The core insight — that STL robustness values give a continuous, quantified safety margin rather than a binary alarm — transfers directly to fixed-infrastructure monitoring.

The key academic reference for RTAMT online monitoring and past-time STL operator constraints is:

> Nickovic, D. et al. *RTAMT: Online Robustness Monitors from STL.* ATVA 2020.

---

## License

MIT
