# SafeEdge — System Architecture

SafeEdge is a **formally-verified edge safety agent** (Track 5: EdgeAgent). A
real-time Signal-Temporal-Logic monitor runs autonomously on a Jetson Orin NX,
while a **Qwen-powered cloud brain deployed on Alibaba Cloud** adds multimodal
incident reporting, adaptive safety policy, and predictive risk forecasting.

The edge loop is safety-critical and **never blocks on the cloud** — if the
network or cloud is down, the Jetson keeps protecting people. The cloud adds
intelligence when available.

---

## 1. System overview

```mermaid
flowchart LR
    subgraph EDGE["🛰️ EDGE — Jetson Orin NX 16GB (real-time, offline-capable)"]
        direction TB
        CAM["USB camera + homography<br/>ground-plane projection<br/><i>video_source.py</i>"]
        DET["YOLOv8s — GPU/TensorRT<br/>person · vehicle detect<br/><i>detector.py</i>"]
        TRK["ByteTrack tracker<br/><i>tracker.py</i>"]
        SIG["Signal Extractor<br/>metric d_min · v_veh · d_pred<br/><i>signal_extractor.py</i>"]
        STL["STL Monitor (rtamt)<br/>robustness ρ₁–ρ₅ @ 30 Hz<br/><i>stl_monitor.py</i>"]
        INT["Intervention Engine<br/>AWARENESS·WARNING·EMERGENCY<br/><i>intervention.py</i>"]
        LQ["Local Qwen2.5-VL (Ollama)<br/>3s scene interpretation<br/><i>qwen_local.py</i>"]
        CC["Cloud Client<br/>non-blocking HTTP<br/><i>cloud_client.py</i>"]

        CAM --> DET --> TRK --> SIG --> STL --> INT
        INT --> LQ
        INT --> CC
        STL --> CC
    end

    subgraph ALI["☁️ ALIBABA CLOUD — Function Compute 3.0 (serverless container)"]
        direction TB
        API["FastAPI Backend<br/>REST + WebSocket<br/><i>backend/app.py</i>"]
        PM["Policy Manager skill<br/><i>policy_manager.py</i>"]
        IR["Incident Reporter skill<br/><i>incident_reporter.py</i>"]
        RF["Risk Forecaster skill<br/><i>risk_forecaster.py</i>"]
        DB[("Tablestore<br/>events · incidents<br/>ρ-history · forecasts<br/><i>store.py</i>")]
        UI["Live Dashboard<br/><i>dashboard/static</i>"]

        API --> PM
        API --> IR
        API --> RF
        API --> DB
        API --> UI
    end

    subgraph QWEN["🧠 QWEN CLOUD — DashScope (intl endpoint)"]
        direction TB
        QMAX["qwen-max<br/>policy + forecast reasoning"]
        QVL["qwen-vl-max<br/>multimodal incident reports"]
        QT["qwen-turbo<br/>text fallback"]
    end

    CC -->|POST /api/state throttled| API
    CC -->|POST /api/events +frame| API
    CC -->|POST /api/policy/evaluate| API
    API -->|STL param patch| CC
    CC -.->|apply_cloud_params| STL

    PM --> QMAX
    RF --> QMAX
    IR --> QVL
    IR --> QT

    BROWSER["👤 Operator browser"] <-->|WebSocket live state| UI

    classDef edge fill:#0b3d2e,stroke:#1f9d6b,color:#e8fff5;
    classDef ali fill:#1a2b4a,stroke:#3a7bd5,color:#eaf2ff;
    classDef qwen fill:#3a1a4a,stroke:#a05fd5,color:#f6eaff;
    class CAM,DET,TRK,SIG,STL,INT,LQ,CC edge;
    class API,PM,IR,RF,DB,UI ali;
    class QMAX,QVL,QT qwen;
```

---

## 2. Safety event → Qwen incident report (sequence)

```mermaid
sequenceDiagram
    participant CAM as USB Camera
    participant STL as STL Monitor (edge)
    participant INT as Intervention Engine
    participant CC as Cloud Client
    participant API as FastAPI (Alibaba FC)
    participant IR as Incident Reporter
    participant QVL as qwen-vl-max
    participant DB as Tablestore
    participant UI as Dashboard

    CAM->>STL: RGB-D frame @ 30 Hz
    Note over STL: ρ = robustness of<br/>"keep pedestrians safe"
    STL->>INT: SafetyState (ρ₁–ρ₅)
    alt ρ < 0  (specification violated)
        INT->>CC: intervention event + frame
        CC-->>API: POST /api/events (non-blocking)
        API->>IR: event + JPEG
        IR->>QVL: image + structured STL data
        QVL-->>IR: natural-language incident report
        IR->>DB: persist incident
        API-->>UI: push incident (WebSocket)
    end
    Note over STL,INT: Edge keeps running at 30 Hz<br/>regardless of cloud latency
```

---

## 3. Adaptive policy loop (closed-loop, edge ⇄ cloud)

```mermaid
sequenceDiagram
    participant STL as STL Monitor (edge)
    participant CC as Cloud Client
    participant API as FastAPI (Alibaba FC)
    participant PM as Policy Manager
    participant QMAX as qwen-max

    loop every 5 min
        CC->>API: POST /api/policy/evaluate<br/>{ρ-summary, event counts, current params}
        API->>PM: evaluate
        PM->>QMAX: reason over safety/false-alarm trade-off
        QMAX-->>PM: JSON STL patch<br/>e.g. {phi2:{proximity_zone:5.5}}
        PM-->>API: patch
        API-->>CC: { patch }
        CC->>STL: apply_cloud_params(patch)
        Note over STL: STL thresholds hot-swapped live —<br/>Qwen tunes formal safety policy
    end
```

---

## 4. Why this architecture

| Concern | Design choice |
|---|---|
| **Real-time safety** | STL robustness computed on-device at 30 Hz; deterministic, formally grounded (no LLM in the safety-critical path). |
| **Autonomy / resilience** | `cloud_client.py` is fire-and-forget with hard timeouts — cloud outages never stall the edge loop. |
| **Sophisticated Qwen use** | Three distinct custom skills: multimodal (`qwen-vl-max`) reporting + reasoning (`qwen-max`) for policy & forecasting, returning structured JSON applied back to the formal monitor. |
| **Alibaba Cloud backend** | FastAPI on **Function Compute** (serverless container, scale-to-zero) + **Tablestore** datastore — this is the deployed backend the hackathon requires. |
| **Domain-portable** | Subjects/objects, signals, STL specs, and prompts are config-driven — the same pipeline retargets to other vision-safety domains (e.g. warehouse forklift–pedestrian). |

---

## 5. Technology stack

- **Edge:** Jetson Orin NX 16GB · JetPack 6.2 · YOLOv8s (Ultralytics, GPU) · ByteTrack (supervision) · `rtamt` (STL) · USB camera + homography · Ollama (local Qwen2.5-VL)
- **Cloud:** Alibaba Cloud Function Compute 3.0 · Tablestore · Container Registry (ACR) · FastAPI · Uvicorn
- **AI:** Qwen Cloud / DashScope (intl) — `qwen-max`, `qwen-vl-max`, `qwen-turbo`, local `qwen2.5-vl`
