# V-JEPA 2.1 surprise as an STL signal

Research record for adding a world-model signal to SafeEdge's STL safety
monitor, and the working folder for an arXiv preprint.

**Question:** does the *surprise* of Meta's V-JEPA 2.1 video world model (how
badly its latent predictor anticipates the next second) rise before SafeEdge's
geometric STL monitor raises WARNING?

**First answer (one clip, 8 episodes):** the ViT-g model's own predictor, with
no task-specific training, anticipates WARNING 2 s ahead with AUROC 0.70
(95% CI 0.55–0.84). A pixel frame-difference baseline is at chance (0.48).
Same-time detection is no better than frame differencing, and frame-level
pooling is dominated by people walking close to the camera. Promising, not yet
established.

## Layout

| Path | What |
| --- | --- |
| `../../tools/vjepa_surprise.py` | The tool: windows a clip, scores surprise (`predictor`, `ridge`, `persistence`, `framediff`), conformal τ, lead times, AUROCs, plots, overlays |
| `../../edge/main.py --trace` | Writes the per-frame geometric trace (signals, ρ1–ρ5, level) the tool aligns against |
| `../../tests/test_vjepa_surprise.py` | 19 unit tests of the model-free logic |
| `scripts/reproduce.sh` | End-to-end: trace → five scorers → table |
| `scripts/analyze_runs.py` | Regenerates the results table, with block-bootstrap CIs, from `results/` alone |
| `scripts/make_synthetic_clip.py` | Synthetic clip + trace for smoke tests (plumbing only) |
| `results/<run>/` | `summary.json`, `surprise.csv` (per window), `maps.npy` (24×24 surprise maps), `plot.png` |
| `results/traces/` | Geometric trace of the evaluated clip |
| `results/compare/` | Packed clip data for the STL-vs-V-JEPA replay dashboard (`../../dashboard/static/compare.html`), built by `../../tools/build_compare_data.py` |
| `results/table.md`, `table.json` | The tables below (frame-level and region-pooled), as generated |
| `session/` | Staged near-miss session: `STAGED_SESSION_PLAN.md`, randomised `run_sheet.csv`, its generator |
| `paper/` | LaTeX draft (`main.tex`, `references.bib`, `figures/`). `make` builds `main.pdf` in the `texlive/texlive` Docker image; `make arxiv` packages the upload |

## Results

Clip: `segment_0455_to_end.mp4` from the deployed unit's `data/tune/` (the
`finaldemo4` validation clip), 291 s at a true 7.7 fps (the MP4 header says 15).
Replayed locally through the unchanged edge loop (YOLOv8s, conf 0.25,
finaldemo4 configs): 8 WARNING+ episodes, 340 calm 4 s windows. Conformal
α = 0.05. Timings on an RTX 4090.

| Scorer | Concurrent AUROC | Anticipation 2 s [95% CI] | Anticipation 4 s [95% CI] | ms / window |
| --- | --- | --- | --- | --- |
| Pixel frame difference (no model) | 0.853 | 0.483 [0.34, 0.63] | 0.534 [0.39, 0.69] | 7 |
| ViT-B persistence | 0.818 | 0.454 [0.31, 0.61] | 0.490 [0.41, 0.60] | 40 |
| ViT-B ridge, fitted on this clip † | 0.955 | 0.575 (4 positives) | 0.583 (8 positives) | 40 |
| ViT-B ridge, fitted on Nov 20 live footage | 0.850 | 0.619 [0.50, 0.74] | 0.596 [0.49, 0.69] | 40 |
| **ViT-g, its own JEPA predictor** | **0.888** | **0.697 [0.55, 0.84]** | 0.647 [0.47, 0.83] | 268 |

† Fitting used up most windows just before each episode; those are excluded
from scoring, so too few positives remain and its concurrent AUROC is partly
in-sample.

- **Concurrent AUROC:** score vs "the window contains WARNING+".
- **Anticipation AUROC:** among fully calm windows, score vs "WARNING+ within
  the next H s". This is the early-warning question. Positives cluster in
  8 episodes, so CIs use a 20 s block bootstrap.
- Lead time before WARNING is reported by the tool, but on this clip ρ1 jumps
  from the no-pair sentinel straight to violation when a pedestrian is
  detected, so it mostly measures when a pedestrian appears (median 0 s).

## Findings

1. **Anticipation, not detection, is where the world model may add value.**
   Motion-only baselines are at chance for anticipation; ViT-g is the only
   scorer whose interval excludes 0.5.
2. **Frame-level p95 pooling is the wrong pooling.** The largest peaks come
   from a person walking near the camera. The ViT-g map does mark a distant
   pedestrian during a geometric warning that the frame score misses. Next:
   pool surprise around tracked pedestrian–vehicle pairs.
3. **Distilled ViT-B/L predictors can't score themselves.** They output in the
   ViT-G teacher's 1664-d space, hence the ridge forecaster for edge-sized models.
4. **Pair-local and actor pooling do not help, and the reason is
   informative.** Pooling the surprise map over the image regions of
   pedestrian–vehicle pairs (within 5 m) or of pedestrians + moving vehicles
   gives anticipation AUROC 0.50 and about 0.49 for every scorer, and an
   identical "concurrent" 0.894 even for frame differencing: the score becomes
   "is the object there", which is geometry again. On this camera no calm
   window ever contains a detected pedestrian — the AOI strip is lined with
   parked cars, so any pedestrian in it is within 4 m of one and WARNING fires
   at once; and detection runs on the AOI crop, so only 17 of 681 pedestrian
   detections fall outside the polygon. "WARNING within 2 s" therefore means
   "a pedestrian is about to appear", and anticipation can only come from cues
   before detection. In the highest-scoring pre-warning windows the ViT-g map
   sits on a person entering at the crop edge (4.8 s, 91.3 s) and on the gap
   beside a parked SUV just before someone steps out (124 s): a pedestrian the
   detector cannot yet see. Hypothesis, from four windows.
5. **Upstream hub download is broken** (`pretrained=True` points at
   `localhost:8300`); the tool loads checkpoints from `dl.fbaipublicfiles.com`.

## Reproduce

```bash
pip install torch timm einops matplotlib   # on top of requirements.txt
# table only, from committed results:
python research/vjepa-surprise/scripts/analyze_runs.py \
  --trace research/vjepa-surprise/results/traces/segment_0455_to_end.trace.jsonl \
  --fps 7.7 research/vjepa-surprise/results/{framediff,vitb_persistence,vitb_ridge_inclip,vitb_ridge_crossseason,vitg_predictor}
# everything, given the clip (see the script header):
CLIP=... FPS=7.7 bash research/vjepa-surprise/scripts/reproduce.sh
```

## Not in this repo (public)

Videos, overlay videos and heat-map frames show identifiable people in a car
park and stay off GitHub; so do model checkpoints and the fitted `ridge.pt`.
The evaluated clip and the live footage used for fitting are on the edge
device. The paper needs consent or anonymisation before any frame is published.

## Next

- [x] Pair-local and actor pooling (`--score pair|actors`; negative result, finding 4)
- [ ] Test the pre-detection hypothesis: occlusion-heavy staged runs (stepping out
      between parked cars, entering at the frame edge) vs runs in the open
- [ ] Staged near-miss session for more episodes; one public set (DoTA or Street Scene)
- [ ] ViT-B ridge latency on the Orin NX (FP16/TensorRT)
- [ ] φ6/φ7 in shadow mode on the live unit
- [ ] Verify `paper/references.bib` entries; fill paper TODOs

## Replay dashboard: STL vs V-JEPA 2.1

Full run instructions, controls and live-dashboard ports:
[`docs/REPLAY_DASHBOARD.md`](../../docs/REPLAY_DASHBOARD.md).

`dashboard/static/compare.html` is a fork of the live dashboard for replays.
For one clip it shows three channels: the STL level as the live dashboard
showed it (raw level through the edge loop's `InterventionEngine`, holds 3/23),
a V-JEPA level (WARNING when ŝ = score/τ stays above 1 for a chosen number of
windows), and the fused level max(STL, V-JEPA). V-JEPA can only tighten: it has
no notion of distance, so it never raises EMERGENCY or lowers STL. Boxes,
the 24 × 24 surprise map, a lane timeline and a merged event log (with each
V-JEPA alert's lead over the next STL WARNING+) are drawn on top.

```bash
# browsers cannot play OpenCV mp4v captures: make an H.264 copy at the true fps
ffmpeg -r 7.7 -i data/segment_0455_to_end.mp4 -c:v libx264 -crf 23 -pix_fmt yuv420p \
    -movflags +faststart -an data/segment_0455_to_end.h264.mp4
python tools/build_compare_data.py \
    --trace research/vjepa-surprise/results/traces/segment_0455_to_end.trace.jsonl \
    --fps 7.7 --video data/segment_0455_to_end.h264.mp4 --downgrade-hold 23 \
    --out research/vjepa-surprise/results/compare/segment_0455_to_end.json \
    research/vjepa-surprise/results/{vitg_predictor,vitb_ridge_crossseason,framediff}
python -m http.server 8000      # from the repo root
# open http://localhost:8000/dashboard/static/compare.html  (?t=128 jumps to a time)
```

The video is read from `data/` (git-ignored) and never leaves the machine;
without it the page still runs on a blank frame. ViT-g scored at 268 ms per
window on an RTX 4090, inside the 0.5 s stride, so a live version is plausible
on a desktop GPU but not on the Orin NX.

## Conflict forecasting: self-labelling pipeline

Next direction: a small head on frozen V-JEPA 2.1 features that predicts
*where* a pedestrian-vehicle conflict will happen 1-3 s ahead (a risk map on
the 24 x 24 patch grid, thresholded into a box), compared against
constant-velocity extrapolation of the tracks (the phi3-style forecast). The
encoder is not retrained; as with V-JEPA 2-AC's 62 h robot post-training, only
a small head learns from domain data.

`tools/conflict_labels.py` makes that domain data from footage SafeEdge has
already replayed: it reads a `--trace`, finds conflicts (pair closer than
1.8 m and converging faster than 0.5 m/s or the vehicle moving: the shape of
the STL EMERGENCY rule), and writes per-window targets aligned with a
`vjepa_surprise.py` run (`target`, `onset` vs `ongoing`, and the
constant-velocity forecast `cv_pred`) to `labels.npz`, plus `events.json`,
`summary.json`, a timeline plot and, with `--review-video`, a three-frames-per-
event sheet for a human check (kept local: it shows people).
`tests/test_conflict_labels.py` covers it on synthetic traces.

```bash
python tools/conflict_labels.py \
    --trace research/vjepa-surprise/results/traces/segment_0455_to_end.trace.jsonl \
    --fps 7.7 --windows-from research/vjepa-surprise/results/vitg_predictor \
    --plot --review-video data/segment_0455_to_end.mp4 \
    --out research/vjepa-surprise/results/labels/segment_0455_to_end
```

On the July clip it finds 12 conflict events; 41 windows have an onset within
2 s, and the constant-velocity forecast is ahead of onset on 4 of the 12.
What it taught us about labelling this camera:

- **Track-ID switches look like motion.** Per-frame ground steps are bimodal:
  real motion stays under 0.3 m per frame, while 722 car steps (and 27
  pedestrian steps) jump 1-16 m when the tracker hands an id to another car.
  Speeds now come from net displacement over 1 s within the unbroken part of a
  track; a step faster than 6 m/s breaks it.
- **Vehicle motion is not separable yet.** With the compressed homography a
  car really driving (E3, 77 s) reads about 0.5 m/s, the same as parked-car
  jitter on new or far-row tracks, and box-based pixel speed is corrupted by
  pedestrians occluding parked cars. `vehicle_moving` is therefore evidence for
  review, not ground truth, until the homography is recalibrated.
- **Nearest-pair labels can miss the real hazard.** At 74-81 s the pedestrian
  is nearest a parked car, so the moving car crossing the lane is never paired.

Labels at scale therefore need, in order: the homography recalibrated,
continuous timestamped recording, and the review sheet spot-checked.

### Data beyond our own camera

| Source | What it gives | Status |
| --- | --- | --- |
| Own footage, self-labelled | In-domain conflicts, this camera, this site | Pipeline ready; needs recording + homography |
| Staged session | Ground-truth timings by design | Held-out test set (pre-registered) |
| MEVA (Kitware/IARPA) | ~330 h of ground-camera video, persons and vehicles outdoors, scripted actors, JSON activity annotations, camera models; CC BY 4.0 (privacy and publicity rights excluded by clause 2(b)(1)); public S3 bucket `mevadata-public-01` | Verified from registry.opendata.aws and mevadata.org; parking-lot camera views and activity list not yet checked |
| VIRAT Ground 2.0 | Static cameras over parking lots and roads, person and vehicle tracks and events, reported CC BY 4.0, zipped on Kitware's data server | Reported by search results only; the Kitware pages could not be fetched here, so verify size, scenes and terms by hand |
| Simulation (CARLA) + NVIDIA Cosmos-Transfer | Thousands of scripted, varied conflicts from a fixed CCTV pose, including ones too dangerous to stage, with exact ground truth from the simulator; Cosmos-Transfer restyles them photoreal while keeping alignment (NVIDIA's smart-city recipe does CARLA to Cosmos-Transfer) | To evaluate |
| Text-to-video generation | Unlimited variety | Not for training: no ground truth, physics errors concentrated in exactly the interaction that matters, and the model would learn the generator's artefacts. At most a qualitative stress test |

Public datasets have few genuine near-misses (scripted activities, not
conflicts); their value is the variety of normal pedestrian-vehicle behaviour
around parked cars. Conflicts themselves come from our footage, the staged
session and simulation.

## Re-run with the 21 Nov camera calibration

The July homography was fitted from assumed bay sizes and is not a physical
camera (see `docs/calibration/README.md`): it compressed ground distances
about threefold. The whole pipeline was re-run unchanged except for the new
homography (`scripts/reproduce.sh`, `YOLO_MODEL` pinned to the same weights),
into `results_calib-2026-11-21/`; `results/` keeps the July numbers the
draft currently reports.

| Run | Anticipation 2 s, July homography | Anticipation 2 s, 21 Nov calibration |
| --- | --- | --- |
| framediff | 0.483 [0.34, 0.63] | 0.554 [0.41, 0.69] |
| vitb_persistence | 0.454 [0.31, 0.61] | 0.514 [0.34, 0.69] |
| vitb_ridge_inclip | 0.575 [0.53, 0.62] | 0.835 [0.81, 0.91] |
| vitb_ridge_crossseason | 0.619 [0.50, 0.74] | 0.670 [0.55, 0.79] |
| vitg_predictor | 0.697 [0.55, 0.84] | 0.689 [0.56, 0.84] |

The ViT-g result is unchanged by the recalibration. The in-clip ridge jump is
not a headline: it is fitted on this clip's own calm windows, which the new
geometry redefines, and it raises 9 false-alarm runs.

The calibration also changes the STL labels the anticipation metrics are
scored against, and it shows the July STL tuning was compensating for the
compressed geometry:

| Replay of segment_0455_to_end | July homography | 21 Nov calibration |
| --- | --- | --- |
| Time at WARNING+ (as shown) | 36.3 % | 25.8 % |
| Time at EMERGENCY | 20.7 % | 7.4 % |
| EMERGENCY onsets | 6 | 3 |
| v_veh_max with no pedestrian in view, median | 0.14 m/s | 0.72 m/s (above 0.5 in 76 % of frames) |
| Genuine conflicts (E3 car in the lane, E6 SUV pulling out) | EMERGENCY, EMERGENCY | WARNING, AWARENESS (d_min 2.0 m, 3.9 m) |
| Conflict-labeller events (pair under 1.8 m, converging) | 12 | 1 |

Two STL problems follow, both for the edge loop rather than the calibration:

- **Vehicle speed is jitter.** `v_veh_max` is the maximum over every vehicle
  in view, from a 6-frame velocity estimate with no guard against track-ID
  switches. In true metres, box jitter on far-row cars reads 0.5-1.2 m/s, so
  the EMERGENCY speed floor fires on parked cars. Fix: speed from net
  displacement over about 1 s within the unbroken part of a track (as
  `tools/conflict_labels.py` does), and only for vehicles paired with a
  pedestrian.
- **Thresholds were tuned in compressed metres.** With true distances the
  genuine conflicts sit at 2-4 m, outside the 1.8 m EMERGENCY and conflict
  bands. Thresholds should be re-derived in real units, preferably as
  time-to-collision, against the forensic ground truth (E3, E6 genuine;
  E1, E2, E5, E8 beside parked cars; E4, E7 lawn walkers).

Until then, the 21 Nov labels are interim: they fix the geometry but inherit
the speed jitter.

## STL signal fix (21 Nov 2026)

`config/stl_specs.yaml` gains a `signals:` section read by the edge loop
(`edge/detection/signal_extractor.py`), and `tools/replay_stl.py` re-runs the
safety layer (extractor, predictor, STL, hysteresis) on a recorded trace in
about a second, reproducing the recorded levels in 99.3-99.96 % of frames
(boxes are stored to 0.1 px). Each change was measured with it on the July
clip against the forensic ground truth, and on synthetic approaches in
`tests/test_signal_motion.py`:

| Setting | Original | New | Why |
| --- | --- | --- | --- |
| Vehicle speed | 6-frame estimate | net displacement over 1 s, track broken at any step faster than 6 m/s | ID switches and jitter stop reading as motion; 2 s halves jitter further but adds about 0.4 s of EMERGENCY latency for a car pulling out, so 1 s |
| `v_veh_max` scope | every vehicle in view | vehicles within 5 m of a pedestrian | a far-row car's jitter no longer speaks for the car beside the pedestrian |
| Closing for the EMERGENCY gate | mutual (includes the pedestrian walking) | the vehicle's own approach | walking up to a parked car is a WARNING, as the code always intended; mutual closing was a workaround for the July homography under-measuring vehicle approach |

Recall is unchanged on synthetic cases: a car driving at a standing
pedestrian at 1-4 m/s raises EMERGENCY on the first frame inside 1.8 m (the
original settings lagged 0.9 s at 4 m/s), and a car pulling out beside one is
flagged 0.45-0.58 s after it starts moving, as before.

Full pipeline re-run with the calibration and the new settings
(`results_calib-2026-11-21_signals/`):

| Replay of segment_0455_to_end | Calibration only | Calibration + signal fix |
| --- | --- | --- |
| Time at EMERGENCY | 7.4 % | 0 % |
| EMERGENCY onsets | 3 (E1, E2 beside parked cars; E7) | 0 |
| Pair frames with closing above 0.5 m/s | 33 % | 3 % |
| E6, SUV pulling out beside a pedestrian | AWARENESS | WARNING |
| Every other July episode | WARNING / EMERGENCY | WARNING |
| ViT-g anticipation 2 s | 0.689 [0.56, 0.84] | 0.687 [0.55, 0.84] |

No July episode was an emergency in true metres (the closest a moving vehicle
came to a pedestrian was 3.9 m), so all eight are now WARNING. The spec still
puts any pedestrian within 4 m of a car at WARNING, parked or not; see
"Parked cars as a minor concern" below for the spec decision.

## Parked cars as a minor concern (21 Nov 2026)

Spec decision: a pedestrian beside parked or stationary cars is AWARENESS,
however close; WARNING needs a vehicle within 5 m of a pedestrian to be moving
(`intervention.warning.min_vehicle_speed: 0.3` m/s). EMERGENCY keeps its own
gate.

A speed threshold alone could not separate the episodes: parked cars read
0.5-0.9 m/s because a pedestrian walking in front of one moves one edge of its
box (E5, car 32: left edge 936 to 875 px, right edge fixed at 1185), which
shifts the box centre. A moving car moves both edges together. Vehicle speed
now comes from the box translation both side edges agree on, plus the bottom
edge, with shifts up to 2 px ignored (`signals.vehicle_motion: rigid`,
`motion_deadband_px: 2`; in the far row one pixel is a quarter of a metre);
edges clipped by the frame border are ignored.

| Episode | Calibration + signal fix | Parked cars minor |
| --- | --- | --- |
| E3 car in the lane, E6 SUV pulling out | WARNING | WARNING |
| E5 | WARNING | WARNING: a car does drive past, 3.7-4.3 m away at 1 m/s (110.8-112.7 s) |
| E1, E2, E8 beside parked cars; E4, E7 lawn walkers | WARNING | AWARENESS |
| Time at WARNING+ (as shown) | 27.8 % | 7.6 % |

Any threshold from 0.2 to 0.3 m/s with a 2-4 px deadband gives the same
episodes. The deadband costs one frame (0.13 s) of EMERGENCY latency for a car
pulling out along a pedestrian; a car driving at one is unchanged.

**Anticipation benchmark.** "Calm" windows (calibration, ridge fit,
anticipation set) were those with STL at SAFE. Under this spec every WARNING
onset is preceded by AWARENESS (the pedestrian is near parked cars before a
car moves), so no SAFE window precedes any onset. The runs here use
`--calm-max-level 1`: calm means no alert shown. Results
(`results_calib-2026-11-21_moving/`, 5 WARNING+ episodes, 24 positive
windows at 2 s):

| 2 s anticipation | Calibration + signal fix | Parked cars minor |
| --- | --- | --- |
| Frame difference | 0.535 [0.39, 0.67] | 0.903 [0.81, 0.97] |
| ViT-B cross-season ridge | 0.669 [0.55, 0.79] | 0.868 [0.75, 0.97] |
| ViT-g predictor | 0.687 [0.55, 0.84] | 0.870 [0.76, 0.96] |

With WARNING now meaning "a car near a pedestrian is moving", the target is
largely motion onset, and pixel change predicts it as well as V-JEPA does.
The numbers are not comparable across the two columns (different targets and
calm sets), and five episodes give wide intervals. In-clip ridge is n/a: all
positive windows fall in the half of the calm set it is fitted on.

## EMERGENCY on time to collision with a moving car (21 Nov 2026)

Review labels three stretches of the July clip EMERGENCY: 32.5-35.7 s
(frames 250-275) and 71.4-74.7 s (frames 550-575), where car 2 drives along
the lane at about 1 m/s while a pedestrian walks into its path, and
110-112 s (E5), where a car passes a pedestrian at 2 m/s. The nearest-pair
rule could not raise any of them: d_min is measured to the centre of the
box's bottom edge (about 2 m further than the body of a side-on car), only
the nearest pair is gated (a parked car in the first two), and closing
counted only the vehicle's own approach.

With `signals.moving_speed: 0.3` the extractor pairs every pedestrian with
every vehicle moving at least that fast, measures distance to the vehicle's
body (the ground segment under its box's bottom edge) and the pair's mutual
closing over the 1 s speed window, so a pedestrian walking into a moving
car's path counts. STL raises EMERGENCY when a moving vehicle's body is
within 1.8 m, or within 5 m with a time to collision under 3 s
(`phi1.ttc_emergency`, `ttc_range`), and WARNING when it is within 4 m.
Parked cars never enter these pairs, so they stay a minor concern.

| July clip, edge loop | Parked cars minor | + moving-car TTC |
| --- | --- | --- |
| EMERGENCY onsets | none | 32.7 s, 73.8 s, 110.1 s (frames 252, 568, 848) |
| E6, SUV pulling out (1.9 m, closing 0.25 m/s) | WARNING | WARNING |
| E1, E4, E7, E8 | AWARENESS | AWARENESS |
| Time at EMERGENCY / WARNING+ | 0 % / 7.6 % | 6.4 % / 8.6 % |

Sweeping time to collision over 3-6 s and the range over 4-5 m raises the
same three and nothing else. 3 s / 5 m raises the first and third earliest
for no added alarms; the second (a slower approach, time to collision 3-7 s
until frame 562) is raised at frame 568, 2.3 s into the reviewed stretch,
and only a 5-6 s threshold raises it earlier. On synthetic approaches the
rule raises EMERGENCY 4.4 s before the nearest-pair 1.8 m rule; a car
driving at, or pulling out beside, a pedestrian is unchanged.

V-JEPA (`results_calib-2026-11-21_ttc/`; same calm definition, the targets
gain about 1 % more WARNING+ frames): frame difference 0.874 [0.76, 0.96],
ViT-B cross-season ridge 0.857 [0.73, 0.97], ViT-g predictor 0.870
[0.75, 0.96] at 2 s. Pixel change and V-JEPA remain indistinguishable on
this target.

Known gaps: `tests/test_stl_monitor.py` has four failures that predate this
change (it expects `clearance_critical: 1.5`; the config has used 3.0 since
the July tuning). `tools/conflict_labels.py` still uses mutual closing in its
own conflict rule.
