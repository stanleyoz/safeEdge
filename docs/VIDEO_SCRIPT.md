# SafeEdge — 3-Minute Demo Video Script

**Target runtime: 2:55** (5s buffer under the 3:00 hard cap — judges stop watching at 3:00).

## Source footage

| Clip | File | Duration | Use for |
|---|---|---|---|
| A — Full dashboard | `Videos/Captures/…18-23-23 FINAL ACCEPTED.mp4` | 7:05 | B-roll: architecture walkthrough, Event Log, Incident Reports panel, feed-clock |
| B — Front approach | `Downloads/safeedge_front_pedestrian.mp4` | 0:26 | Cold-open only (0:00–0:12) |
| D — Main demo walkthrough | Your `finaldemo4` screen recording (session started 12:14:36 GMT+8) | 0:16 (12:15:28–12:15:44) | **Primary demo section, 1:05–1:50** — real SAFE→WARNING→EMERGENCY→WARNING arc with a live Qwen-VL report firing mid-clip |

Pull short (3–8s) segments from Clip A wherever indicated — timecodes are approximate; pick moments that visually match the narration (e.g. any WARNING badge for the "graduated response" beat, the Incident Reports panel scrolled to show 2–3 entries). Clip C (reverse/side approach) is no longer used in the main script — hold it as a spare if you want a cutaway.

**Clip D is the real, verified centerpiece.** Signal timeline (derived from live logs, GMT+8):

| Time | Badge | d_min | v_veh | ρ1 (`d_min − 3.0`) | What fired |
|---|---|---|---|---|---|
| 12:15:28 | SAFE | 100 m | 0.07–0.32 (baseline jitter) | +97 | no pair in zone |
| 12:15:32 | WARNING | 1.90 m | 0.14 | −1.10 | predictive spec (ρ2/ρ3) |
| **12:15:36** | **EMERGENCY** | **1.69 m** | **0.75 m/s** | **−1.31** | proximity + real vehicle speed (v_veh_floor) — **Qwen-VL incident report fires here** |
| 12:15:44 | WARNING | 1.2 m | 0.46 | −1.80 | pedestrian now walking among parked cars, severity eases |

Qwen incident report text generated at this exact moment (12:15:36 GMT+8): *"…a pedestrian was observed in the car park approaching a dark-colored vehicle, with a measured distance of 1.69 meters—below the critical safety threshold. The vehicle was moving at 0.75 m/s, and the predicted clearance was 0.93 meters, indicating a high-risk proximity event."*

---

## Script

| Time | Visual | Narration (read at ~150 wpm) |
|---|---|---|
| **0:00–0:12** | Cold open: Clip B, first 5s (car approaching, badge flips to EMERGENCY) — no narration yet, let it breathe, then punch in | *(on-screen text overlay, bottom-third, `SFMono-Regular` monospace matching the dashboard's own feed-clock style, off-white on translucent black — see "Title cards" section below for exact timing)* |
| **0:12–0:25** | Cut to presenter / title card: "SafeEdge — Formally-Verified Edge Safety" | "This is SafeEdge — a car-park safety system that doesn't just detect people and cars. It computes, mathematically, how much safety margin is left, in real time, on the device." |
| **0:25–0:45** | Clip A: architecture diagram (`docs/architecture.html`) or a quick screen-capture of it | "A Jetson Orin NX runs a Signal Temporal Logic safety monitor at the edge — fully offline-capable. When it detects a violation, it hands off, non-blocking, to a backend running on **Alibaba Cloud Function Compute**, backed by **Tablestore** — no LLM ever sits in the safety-critical loop." |
| **0:45–1:05** | Clip A: dashboard overview — feed panel, level badge, ρ bars all visible | "That backend calls **Qwen Cloud** three ways: `qwen-vl-max` writes real incident reports from the actual camera frame, `qwen-max` tunes the safety thresholds live, and forecasts risk windows. Everything you're about to see is real output from that live pipeline." |
| **1:05–1:14** | **Clip D**, SAFE badge, ρ bars green, feed-clock at 12:15:28 | "Watch the robustness value here — ρ-one, positive ninety-seven. Completely safe: no pedestrian-vehicle pair in the zone." |
| **1:14–1:24** | **Clip D continues** — badge flips WARNING at 12:15:32; hold a beat on the ρ readout going negative | "A pedestrian and a vehicle enter the zone. ρ turns negative — the predictive spec sees them closing, several seconds out — WARNING." |
| **1:24–1:40** | **Clip D continues** — badge flips EMERGENCY at 12:15:36; **cut to the Incident Reports panel** as the qwen-vl-max report populates live | "Vehicle speed jumps to 0.75 metres per second — a real approach, not sensor jitter — clearance falls under threshold. EMERGENCY. On Alibaba Cloud, that event triggers `qwen-vl-max`, which writes this report from the actual frame, in real time." *(let the report text be readable on screen for ~2s)* |
| **1:40–1:50** | **Clip D concludes** — badge eases to WARNING at 12:15:44, pedestrian now walking among parked cars | "As they separate, severity eases back to WARNING — a continuum, not a binary alarm." |
| **1:50–2:10** | Clip A: cut to Event Log + Incident Reports panels populated with entries | "Every one of these is logged, and the emergencies get a live `qwen-vl-max` report — real vision-language output on the actual danger frame, not a canned string. Alibaba Cloud proof is a linked code file in the repo; `/healthz` self-reports the live Function Compute instance." |
| **2:10–2:35** | Presenter or on-screen bullet text (no clip needed) | "We didn't fake a clean demo. Our first tagging rule false-fired on parked cars from bounding-box jitter. We fixed that — then operator ground-truth review caught a real *missed* emergency, so we rebuilt the rule recall-first: never miss real danger, even if it means more false alarms. We proved the fix with exact per-frame signal logs, not a re-run and a guess." |
| **2:35–2:50** | On-screen text: camera icon → multi-camera icon; GPIO/buzzer icon | "Known limit: a single monocular camera under-measures a car driving straight at it — that's geometry, not a bug. Roadmap: lateral camera framing, multi-camera stereo, and a GPIO-driven physical buzzer failsafe so a real danger still gets a response even with zero network." |
| **2:50–2:55** | Title card: repo URL + Track 5 EdgeAgent | "SafeEdge. Track 5, EdgeAgent. Formally verified at the edge, explained by Qwen Cloud, running on Alibaba Cloud." |

---

## Pacing note for the Clip D walkthrough (1:05–1:50)

The real telemetry arc is only **16 real seconds** (12:15:28→12:15:44), but the script budgets **45 seconds** to give the narration room and let judges actually read the ρ values and badge. Slow Clip D to roughly **35% speed** (~2.8×) so the real 16s stretches to fill the slot naturally — this is the same technique any product demo uses to let viewers read a UI, and it's honest as long as it's disclosed. Add a small, unobtrusive on-screen caption during this segment: `Playback slowed for clarity` (bottom corner, same monospace treatment as the other overlays) — keeps this consistent with the "we don't hide our tuning process" credibility the rest of the submission earns.

Do **not** speed up or cut between the badge transitions — the whole point of this segment is that it's one continuous, unedited real arc.

---

## Title cards — exact text & timing

**Overlay 1 — cold open (over Clip B footage, bottom-third, not full-screen):**

| Timecode | Text |
|---|---|
| 0:01–0:06 (0.3s fade in/out) | `A pedestrian. A moving vehicle.` |
| 0:06–0:11 | `Watching, every frame — before it's too late.` |
| 0:11–0:12 | *(no text — clear the frame right as the badge flips to EMERGENCY; let the real UI be the payoff)* |

Style: `SFMono-Regular, Consolas, monospace` (same family as the dashboard's own feed-clock), text `#e6edf3`, background `rgba(0,0,0,.55)`, ~3px radius — visually continuous with the product itself.

**Overlay 2 — closing card (2:50–2:55, full-screen, not an overlay):**

```
SafeEdge
Formally-Verified Edge Safety — Track 5: EdgeAgent

github.com/stanleyoz/safeEdge
```

"SafeEdge" large/bold, tagline+track medium weight fading in 0.5s later, repo URL in the monospace family, dimmer, fading in 1s later — reads as title → context → citation. The Qwen Cloud / Alibaba Cloud mention stays in the voiceover at this timecode, not repeated on the card.

---

## Notes for the edit

- **Cold open (0:00–0:12) has no voice** — let the real footage speak first; judges see the system working before they hear a claim about it.
- Keep the **feed-clock visible** in every dashboard shot — it's your proof the footage is live output, not staged.
- If time is tight, the safest cut is trimming **2:10–2:35** (the tuning story) to ~15s — it's the most valuable section for engineering credibility but the least essential if you're over time. Do not cut the two hero clips (B/C) or the architecture beat — those satisfy the "functions as depicted" and "shows the device it was built for" requirements directly.
- No third-party music/trademarks per the rules — if you want music, use royalty-free/CC0 only, or none.
- Final render: upload to YouTube/Vimeo/Youku, public, link in the submission form.
