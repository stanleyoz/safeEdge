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
| `results/table.md`, `table.json` | The table below, as generated |
| `paper/` | LaTeX draft (`main.tex`, `references.bib`, `figures/`, `Makefile`) |

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
4. **Upstream hub download is broken** (`pretrained=True` points at
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

- [ ] Pair-local surprise pooling (patches around tracked pedestrian–vehicle pairs)
- [ ] Staged near-miss session for more episodes; one public set (DoTA or Street Scene)
- [ ] ViT-B ridge latency on the Orin NX (FP16/TensorRT)
- [ ] φ6/φ7 in shadow mode on the live unit
- [ ] Verify `paper/references.bib` entries; fill paper TODOs
