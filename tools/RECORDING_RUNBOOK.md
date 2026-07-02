# SafeEdge — Live Recording Runbook

**Budget:** ~1 hour with one pedestrian (helper) + one moving car. No second take another day.
**Rule #1:** Spend the hour *only* on choreographed takes. Everything fragile is proven the day before.
**Rule #2:** The screen recorder runs **unattended** — nobody needs to sit at the dashboard during a take.

Golden path each shoot: **Real camera → Jetson pipeline → Alibaba FC → dashboard → screen recorder**, with a phone on a tripod filming the *physical* scene for the split-screen final cut.

---

## PHASE 0 — PREP (day before, or ≥2 h ahead — NO helper, no pressure)

### 0.1 Bring the pipeline up (Jetson)
```bash
ssh -i ~/.ssh/id_jetson icp@eliza-jetson.tail52ce98.ts.net
cd /home/icp/safeedge/qwen_cloud
# make sure nothing is already running (avoid double-launch webcam clash):
docker ps -q --filter ancestor=dustynv/l4t-pytorch:r36.4.0 | xargs -r docker kill
nohup bash tools/docker_live.sh --source webcam > /tmp/safeedge_live.log 2>&1 &
```

### 0.2 Preflight — one command, must be ALL GREEN
```bash
bash tools/preflight.sh          # from the laptop
```
Checks: dashboard 200 · backend healthy · state fresh (<10 s) · frame actually changing · exactly one Jetson container. Fix any FAIL now, not during the hour.

### 0.3 Camera pose + geometry (THE thing that silently ruins takes)
The homography and AOI are **camera-specific**. If the camera/tripod moved even a little since last calibration, redo them:
- **Homography:** open `…/safeEdge/homography-editor.html`, click ≥4 ground points with known metric spacing (parking bay ≈ 2.5 m × 5 m), paste correspondences back → I drop them into `config/homography.npy` and restart.
- **AOI:** open `…/safeEdge/aoi-editor.html`, trace the **lane where the car will drive + where the pedestrian will stand**. Paste back → into `config/camera_config.yaml`.
- **Frame the danger lane ACROSS the camera** (car drives left↔right, *not* toward/away). Monocular velocity underestimates radial motion — a car driving straight at the camera may not exceed the 0.8 m/s motion gate. Sideways motion reads true speed. This is the single most important framing decision.

### 0.4 Solo dry-run (be your own test pedestrian)
Watch the dashboard while you:
1. **Walk into the lane** → confirm a `person#N` box appears and **φ1 (Minimum Separation) falls** from 97 as `d_min` drops. If no person box: lighting/conf/AOI issue — fix now.
2. **Roll/drive the car slowly across the lane** → confirm `v_veh_max` climbs above **0.8 m/s** (motion gate) and ideally past **1.4 m/s** (φ2 speed limit). If it never does, the lane is too head-on → re-aim per 0.3.
3. Stand next to a **parked** car → confirm level stays **AWARENESS**, not EMERGENCY (motion-gate proof).

### 0.5 Capture a safety-net clip
```bash
# inside the Jetson docker (or via docker exec) — records an annotated MP4:
python3 tools/live_record.py --out-dir /safeedge/data/captures
```
Keep this clip. If the live camera dies mid-hour, you can still drive the dashboard from it:
```bash
python tools/replay_to_cloud.py --url $SAFEEDGE_CLOUD_URL --video data/captures/<clip>.mp4
```

### 0.6 Recorder setup + 30-second test (DON'T skip)
- **OBS (Windows, recommended):** Scene = *Window Capture* of the browser tab (full-screen the dashboard, F11). Optionally add a second source (webcam / phone-as-webcam) as a small PiP of the real scene → instant split-screen. Settings → Output → 1080p/30, MP4. Record 30 s, play it back, confirm audio+video land in the file.
- **Fallback — Xbox Game Bar:** `Win+G` → Record. Zero setup, records the active window, no PiP.
- **Bulletproof fallback:** a phone on a tripod pointed at the monitor.
- Full-screen the dashboard (**F11**) and hide the mouse cursor when not needed. Confirm the **incident panel** (Qwen reports) and **ρ bars** are both visible in frame.

### 0.7 Pre-write the shot list (see Phase 1) and tape marks on the ground
Chalk/cones for: pedestrian "stand here", car "start / closest-approach / stop" lines. Marks = repeatable takes = fewer wasted minutes.

---

## PHASE 1 — THE HOUR (helper present, camera rolling)

Start the screen recorder **once** at the top and let it run through all takes (trim later). Call each take out loud ("Take C, moving car, action") so the audio timeline marks your cuts. A phone on a tripod films the real scene in parallel.

> **Safety first, always:** low speed, wide margin, helper always facing the car and aware, car stops well short of the mark. "Within 3 m" is staged at a slow roll — never an actual near-miss.

| # | Take | Choreography | Dashboard should show | ~min |
|---|------|--------------|-----------------------|------|
| A | **Baseline SAFE** | Helper strolls through empty lane, car parked | SAFE badge, AOI outline, ρ all high | 3 |
| B | **Parked-car calm** (motion-gate proof) | Helper walks right beside a **parked** car | `d_min` low **but** level = **AWARENESS**, not emergency | 4 |
| C | **★ Money shot** | Helper stands on mark; you drive car **across** the lane, slow, passing within ~3 m | φ1 → warning → **EMERGENCY**, Qwen incident report appears | 8 |
| D | **Predictive near-miss** (φ3) | Helper and car on *converging* paths, still a few metres apart | **φ3 dips / predictive_warning fires BEFORE** they're close | 6 |
| E | **Clearance / recovery** (φ5) | Car passes and leaves, helper steps out | Level returns SAFE→ φ5 satisfied | 3 |

Do **C twice** (it's the hero shot). Budget: ~30 min of takes, leaving 30 min slack for resets and one full re-run.

**Live monitor while recording:** in a spare terminal, `watch -n1 'curl -s $SAFEEDGE_CLOUD_URL/api/state/latest | python3 -c "import sys,json;d=json.load(sys.stdin);print(d[\"level_label\"], d[\"signals\"])"'` so you can call "reset" the instant a take lands.

---

## PHASE 2 — WRAP (helper released, expensive lunch earned)

```bash
bash tools/pull_incidents.sh 20        # newest Qwen incident reports + level transitions
```
Pick the best-worded EMERGENCY report to quote on-screen in the final video. Copy the safety-net clip and the OBS recording off the machines. Then stop the pipeline to zero the cloud cost:
```bash
docker ps -q --filter ancestor=dustynv/l4t-pytorch:r36.4.0 | xargs -r docker kill
```

---

## If it goes wrong mid-hour (decision tree)
- **Dashboard frozen / no updates** → `bash tools/preflight.sh`; if state stale, restart `docker_live.sh` (single instance).
- **Person not detected** → lower `--conf`, check helper is inside the AOI, check lighting/backlight.
- **Car near person but stays SAFE** → car too head-on (radial) → drive more *across* the frame, or it's below the 0.8 m/s motion gate → drive a touch faster.
- **Everything on fire** → fall back to `replay_to_cloud.py` with the 0.5 clip, or `simulate_scenario.py --cloud --cycles 1` for a fully scripted arc. The dashboard still records a clean demo.

## Branch points (adjust to your answers)
- *Who drives while helper is pedestrian:* default assumes **you drive, recorder unattended, phone films the scene**. If a 3rd person drives, you can call shots + watch the live monitor instead.
- *Recorder:* default OBS-on-Windows; swap Section 0.6 for Game Bar / QuickTime as needed.
