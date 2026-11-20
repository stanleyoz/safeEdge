#!/bin/bash
# Reproduce the V-JEPA 2.1 surprise experiment on one recorded clip.
#
# Needs: the clip (not in git; ask for it or pull from the edge device's
# data/tune/), its homography and camera/STL configs, a CUDA GPU, and
# torch + timm + einops + matplotlib on top of requirements.txt. The ViT-g run
# downloads a 16.9 GB checkpoint on first use; ViT-B is 1.7 GB.
#
# Usage:
#   CLIP=data/tune/segment_0455_to_end.mp4 FPS=7.7 \
#   CAMERA_CFG=config/camera_config.yaml STL_CFG=config/stl_specs.yaml \
#   FIT_CLIP=data/live/live_nov20.mp4 FIT_FPS=10.8 \
#   bash research/vjepa-surprise/scripts/reproduce.sh
#
# FPS is the clip's TRUE capture rate (frames / wall time); the demo clips'
# MP4 headers say 15 but were captured at 7.7. FIT_CLIP is independent calm
# footage of the same camera used to fit the cross-season ridge forecaster.
set -euo pipefail
cd "$(dirname "$0")/../../.."

: "${CLIP:?set CLIP}" "${FPS:?set FPS}"
CAMERA_CFG="${CAMERA_CFG:-config/camera_config.yaml}"
STL_CFG="${STL_CFG:-config/stl_specs.yaml}"
ALPHA="${ALPHA:-0.05}"
OUT="${OUT:-research/vjepa-surprise/results}"
stem="$(basename "$CLIP" .mp4)"
trace="$OUT/traces/$stem.trace.jsonl"
mkdir -p "$OUT/traces"

echo "[1] Geometric trace: replay through the unchanged edge loop"
SAFEEDGE_CFG_STL="$STL_CFG" SAFEEDGE_CFG_CAMERA="$CAMERA_CFG" \
  python -m edge.main --source file --file-path "$CLIP" --model yolov8s.pt \
    --conf 0.25 --fps "$FPS" --no-record --trace "$trace"

common=(--video "$CLIP" --video-fps "$FPS" --trace "$trace"
        --camera-config "$CAMERA_CFG" --alpha "$ALPHA")

echo "[2] Baselines and ViT-B"
python tools/vjepa_surprise.py "${common[@]}" --mode framediff        --out "$OUT/framediff"
python tools/vjepa_surprise.py "${common[@]}" --model base --mode persistence --out "$OUT/vitb_persistence"
python tools/vjepa_surprise.py "${common[@]}" --model base --mode ridge      --out "$OUT/vitb_ridge_inclip"

if [ -n "${FIT_CLIP:-}" ]; then
  echo "[3] Cross-season ridge: fit on independent footage, apply unchanged"
  python tools/vjepa_surprise.py --video "$FIT_CLIP" --video-fps "${FIT_FPS:?set FIT_FPS}" \
    --calib-s 999999 --camera-config "$CAMERA_CFG" --model base --mode ridge \
    --alpha "$ALPHA" --out "$OUT/live_fit_vitb_ridge"
  python tools/vjepa_surprise.py "${common[@]}" --model base --mode ridge \
    --ridge-from "$OUT/live_fit_vitb_ridge/ridge.pt" --out "$OUT/vitb_ridge_crossseason"
fi

echo "[4] ViT-g with its own JEPA predictor"
python tools/vjepa_surprise.py "${common[@]}" --model giant --mode predictor --out "$OUT/vitg_predictor"

echo "[5] Table with block-bootstrap CIs"
runs=("$OUT/framediff" "$OUT/vitb_persistence" "$OUT/vitb_ridge_inclip")
[ -d "$OUT/vitb_ridge_crossseason" ] && runs+=("$OUT/vitb_ridge_crossseason")
runs+=("$OUT/vitg_predictor")
python research/vjepa-surprise/scripts/analyze_runs.py --trace "$trace" --fps "$FPS" \
  --json "$OUT/table.json" "${runs[@]}" | tee "$OUT/table.md"
