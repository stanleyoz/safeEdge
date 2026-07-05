# Proof of Alibaba Cloud Deployment — recording script

> Hackathon requirement: *a short recording (separate from your demo) proving
> your project's backend is running on Alibaba Cloud, plus a link to a code file
> demonstrating use of Alibaba Cloud services and APIs.*

This clip is **~75 seconds**, screen-capture only, no edd. Goal: leave zero doubt
the backend runs on Alibaba Cloud (not localhost) and uses Alibaba services.

**Live URL:** `https://safeedg-backend-nkmqevdhff.ap-southeast-1.fcapp.run`

---

## Before you record

- [ ] Redeploy the FC instance with the enhanced `/healthz` (self-reports FC identity):
      ```bash
      cd ~/projects/qwen_cloud
      DOCKERHUB_USER=stanleyoz ./deploy/build_and_push.sh
      set -a; . backend/.env; set +a
      s deploy -t deploy/s.yaml -a default
      ```
- [ ] Warm the instance: `curl -s <URL>/healthz` once (avoids a cold-start pause on camera).
- [ ] Open three tabs/windows: (1) a terminal, (2) Alibaba **Function Compute** console on the `safeedge-backend` function, (3) Alibaba **Tablestore** console on the `safeedge` instance.
- [ ] Zoom the browser to ~125% so text is legible in the capture.

---

## Shot list (≈75 s)

### 1. The function exists on Alibaba Function Compute (~15 s)
Show the **FC console** → function `safeedge-backend`:
- Region badge: **Singapore (ap-southeast-1)**
- State: **Active**
- The ARN: `acs:fc:ap-southeast-1:<account-id>:functions/safeedge-backend`
- The HTTP trigger URL (the `.fcapp.run` domain)

*Caption / say:* "The backend is a Function Compute service in Alibaba Cloud, Singapore region."

### 2. The live endpoint self-reports its Alibaba identity (~20 s)
In the terminal:
```bash
curl -s https://safeedg-backend-nkmqevdhff.ap-southeast-1.fcapp.run/healthz | python3 -m json.tool
```
Point at the response:
```json
{
  "platform": "alibaba-function-compute",
  "fc": { "region": "ap-southeast-1", "account_id": "...", "function": "safeedge-backend", "instance": "..." },
  "store": "TablestoreStore",
  "models": { "reasoning": "qwen-max", "vision": "qwen-vl-max" }
}
```
*Say:* "Hitting the public URL, the service reports its own Function Compute region, account, and instance ID — and that it's backed by Alibaba Tablestore."

### 3. End-to-end: Qwen Cloud + Tablestore, live (~25 s)
```bash
curl -s -X POST .../api/events -H 'Content-Type: application/json' \
  -d '{"timestamp":'$(date +%s)',"level":3,"d_min":0.6,"v_veh_max":4.2,"d_pred":-0.2,"rho_min":-1.5,"message":"EMERGENCY: pedestrian behind reversing vehicle"}'
curl -s .../api/incidents | python3 -m json.tool   # show the Qwen-generated report
```
*Say:* "An edge safety event posted to the cloud is sent to Qwen-VL, which writes a natural-language incident report — persisted in Alibaba Tablestore."

### 4. Tablestore holds the data (~15 s)
Switch to the **Tablestore console** → instance `safeedge` → table `safeedge_incidents` → **Query Data** → show the rows (the incident just created).

*Say:* "The reports are stored in Alibaba Tablestore, in the same region."

---

## Code-file links (paste into the submission)

The requirement also wants a link to a code file using Alibaba Cloud services/APIs.
Use the **permalink** form (commit-pinned) once merged to `main`:

- **Alibaba Tablestore SDK usage** — `backend/store.py` (class `TablestoreStore`: `OTSClient`, `create_table`, `put_row`, `get_range`)
- **Function Compute deployment manifest** — `deploy/s.yaml`
- **Qwen via DashScope (Alibaba model service)** — `cloud/qwen_client.py`

Primary link to submit: **`backend/store.py`** (unambiguous Alibaba Cloud service API calls).

---

## What NOT to show
- Do not show `backend/.env`, the AccessKey secret, or the DashScope key on screen.
- The `/healthz` masks nothing sensitive (no keys) — safe to show in full.
- The FC console env-vars panel **does** show masked secrets — fine, but don't expand/reveal them.
