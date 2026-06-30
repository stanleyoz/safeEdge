# SafeEdge — Alibaba Cloud Deployment

Deploys the **cloud backend** (FastAPI + 3 Qwen skills + dashboard) to
**Function Compute 3.0** with a **Tablestore** datastore. This is the component
that satisfies the hackathon "backend running on Alibaba Cloud" requirement.
The Jetson edge agent stays on-prem and calls this backend over HTTPS.

```
Jetson (edge)  ──HTTPS──►  Function Compute (this backend)  ──►  Qwen (dashscope-intl)
                                      │
                                      └──►  Tablestore (events, incidents, ρ-history)
```

## Prerequisites

- Alibaba Cloud account (international), same region throughout — we use `ap-southeast-1` (Singapore).
- CLI tools: `docker` (with buildx), Serverless Devs (`npm i -g @serverless-devs/s`).
- A RAM user AccessKey (ID + Secret) with FC + Tablestore permissions.
- Your `DASHSCOPE_API_KEY` (hackathon key, intl endpoint).
- A free **Docker Hub** account (we host the image there — Alibaba ACR Enterprise costs $23/mo, ACR Personal is unavailable in this region).

## 1. Container image (Docker Hub — free)

FC pulls the backend image from a public Docker Hub repo. The image contains
**no secrets** (injected as FC env vars at runtime), so public is safe.

1. Create a free account at hub.docker.com.
2. Log in locally (Docker Hub is docker's default registry):
   ```bash
   docker login
   ```
3. The push step below creates the repo; afterwards make it **public**
   (Docker Hub → repository → Settings → Make public) so FC can pull it.

## 2. Tablestore (datastore)

1. Console → Tablestore → create an **instance** (e.g. `safeedge`) in `ap-southeast-1`, **high-performance** or **capacity** type (serverless billing).
2. Note the **VPC/Internet endpoint**, e.g.
   `https://safeedge.ap-southeast-1.ots.aliyuncs.com`.
3. Tables are **auto-created on first run** by `backend/store.py` (events, incidents, rho, kv) — no manual schema needed.

## 3. Build & push the image

```bash
DOCKERHUB_USER=youruser ./deploy/build_and_push.sh
```
Then set `vars.image` in `deploy/s.yaml` to the printed image URI
(`docker.io/youruser/safeedge-backend:latest`). Ensure the repo is **public**.

## 4. Deploy to Function Compute

Export the secrets the `s.yaml` references, then deploy:

```bash
export DASHSCOPE_API_KEY=sk-...
export TABLESTORE_ENDPOINT=https://safeedge.ap-southeast-1.ots.aliyuncs.com
export TABLESTORE_INSTANCE=safeedge
export ALIBABA_CLOUD_ACCESS_KEY_ID=LTAI...
export ALIBABA_CLOUD_ACCESS_KEY_SECRET=...

s deploy
```

Serverless Devs prints the function's HTTP trigger URL, e.g.
`https://safeedge-backend-xxxx.ap-southeast-1.fcapp.run`.

## 5. Verify the live deployment

```bash
BASE=https://<your-fc-url>
curl -s $BASE/healthz                       # {"status":"ok","store":"TablestoreStore",...}
curl -s -X POST $BASE/api/events -H 'Content-Type: application/json' \
  -d '{"timestamp":1782795000,"level":3,"d_min":0.6,"v_veh_max":4.2,"d_pred":-0.2,"rho_min":-1.4,"message":"EMERGENCY: test"}'
sleep 6
curl -s $BASE/api/incidents                 # Qwen-generated report stored in Tablestore
```

Open `$BASE/` in a browser for the live dashboard.

## 6. Point the edge at the cloud

On the Jetson set `SAFEEDGE_CLOUD_URL=$BASE` (edge-wiring step) so intervention
events and policy requests flow to this backend.

## Proof-of-deployment artifacts (for submission)

The rules require a recording + a link to a code file using Alibaba Cloud services:

- **Code file links:**
  - `cloud/qwen_client.py` — Qwen via DashScope (Alibaba Cloud model service)
  - `backend/store.py` — `TablestoreStore` (Alibaba Cloud Tablestore SDK)
  - `deploy/s.yaml` — Function Compute deployment manifest
- **Recording:** screen-capture showing
  1. the FC console with the running `safeedge-backend` function,
  2. `curl $BASE/healthz` returning `"store":"TablestoreStore"`,
  3. the Tablestore console showing rows in `safeedge_incidents`,
  4. the live dashboard at `$BASE/`.

## Cost / free-tier notes

- Function Compute scales to zero — you pay only per request (well within hackathon credits).
- Tablestore reserved throughput set to 0 (pay-per-use).
- Qwen free tier: 1M in+out tokens / 90 days. Our calls are small (policy ~700,
  incident ~300–1500 with image, forecast ~700 tokens) — comfortably within budget.
