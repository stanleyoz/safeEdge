#!/bin/bash
# SafeEdge OFFLINE tuning replay — run a saved raw clip through the FULL pipeline
# on the Jetson GPU, headless, with overridable params, and print a metrics
# summary. No camera, no cloud, no display — pure post-processing for tuning.
#
# Usage:
#   bash tools/docker_replay.sh <clip.mp4> [TAG]
# Params via env (override the live configs for THIS run only):
#   FPS=7.7           true capture fps of the clip (frames ÷ wall-duration)
#   CONF=0.25         detector confidence
#   CROP_ZOOM=false   AOI crop-zoom on/off
#   MOTION_GATE=1.2   phi1 motion gate (m/s) — vehicle must move faster to be "danger"
#
# Example sweep:
#   FPS=7.7 CROP_ZOOM=true  MOTION_GATE=0.8 bash tools/docker_replay.sh data/captures/clip.mp4 baseline
#   FPS=7.7 CROP_ZOOM=false MOTION_GATE=1.2 bash tools/docker_replay.sh data/captures/clip.mp4 fixed
set -e

CLIP="${1:?usage: docker_replay.sh <clip.mp4> [TAG]}"
TAG="${2:-run}"
PROJECT="/home/icp/safeedge/qwen_cloud"
IMAGE="dustynv/l4t-pytorch:r36.4.0"
FPS="${FPS:-7.7}"; CONF="${CONF:-0.25}"; CROP_ZOOM="${CROP_ZOOM:-false}"; MOTION_GATE="${MOTION_GATE:-1.2}"
TRACK_BUFFER="${TRACK_BUFFER:-0}"   # 0 → auto (fps×4); else frames of lost-track persistence
TRACK_ACT="${TRACK_ACT:-0.4}"       # ByteTrack activation confidence
CLOUD="${CLOUD:-0}"                 # 1 → post to dashboard + pace real-time (for screen recording)
CLOUD_URL="${SAFEEDGE_CLOUD_URL:-https://safeedg-backend-nkmqevdhff.ap-southeast-1.fcapp.run}"
REALTIME="${REALTIME:-0}"           # 1 → pace real-time even without cloud (for timestamp-accurate logs)
if [ "$CLOUD" = "1" ] || [ "$CLOUD" = "true" ]; then
  CLOUD_ENV=(-e "SAFEEDGE_CLOUD_URL=$CLOUD_URL"); RT_FLAG="--realtime"
  echo ">>> CLOUD MODE: posting to dashboard, paced real-time (~$(python3 -c "print(round(4530/${FPS:-7.7}/60,1))" 2>/dev/null)min). Open https://stanleyoz.github.io/safeEdge/ and record."
else
  CLOUD_ENV=()
  if [ "$REALTIME" = "1" ]; then RT_FLAG="--realtime"; else RT_FLAG=""; fi
fi

# clip path as seen inside the container (project is mounted at /safeedge)
case "$CLIP" in
  /safeedge/*) CLIP_IN="$CLIP" ;;
  /*)          CLIP_IN="/safeedge/${CLIP#$PROJECT/}" ;;
  *)           CLIP_IN="/safeedge/$CLIP" ;;
esac

OUT="/safeedge/data/tune/$TAG"
echo "=== SafeEdge offline replay [$TAG] ==="
echo "clip=$CLIP_IN  fps=$FPS  conf=$CONF  crop_zoom=$CROP_ZOOM  motion_gate=$MOTION_GATE  track_buffer=$TRACK_BUFFER  track_act=$TRACK_ACT"

docker run --rm --runtime nvidia \
  -e PYTHONPATH=/safeedge \
  -e SAFEEDGE_CFG_CAMERA="$OUT/camera_config.yaml" \
  -e SAFEEDGE_CFG_STL="$OUT/stl_specs.yaml" \
  -e SAFEEDGE_TRACK_BUFFER="$TRACK_BUFFER" \
  -e SAFEEDGE_TRACK_ACT="$TRACK_ACT" \
  "${CLOUD_ENV[@]}" \
  -v "${PROJECT}:/safeedge" \
  "${IMAGE}" \
  bash -c "
    pip install 'numpy<2' ultralytics supervision rtamt pyyaml python-dotenv \
      --index-url https://pypi.org/simple/ -q 2>&1 | tail -1
    mkdir -p '$OUT'
    # temp config variants: apply CROP_ZOOM + MOTION_GATE, keep everything else
    python3 - <<PY
import yaml
cam=yaml.safe_load(open('/safeedge/config/camera_config.yaml'))
cam.setdefault('aoi',{})['crop_zoom']=('$CROP_ZOOM'.lower() in ('1','true','yes'))
yaml.safe_dump(cam,open('$OUT/camera_config.yaml','w'))
stl=yaml.safe_load(open('/safeedge/config/stl_specs.yaml'))
stl['specs']['phi1']['params']['motion_gate']=float('$MOTION_GATE')
yaml.safe_dump(stl,open('$OUT/stl_specs.yaml','w'))
print('configs written to $OUT')
PY
    cd /safeedge && python3 -m edge.main --source file --file-path '$CLIP_IN' \
      --model /safeedge/yolov8s.pt --conf $CONF --fps $FPS --no-record $RT_FLAG 2>&1 | tee '$OUT/replay.log'
    echo '=== METRICS [$TAG] ==='
    python3 - <<PY
import re
log=open('$OUT/replay.log').read()
lvl=re.findall(r'\[L(\d)\]',log)
from collections import Counter
c=Counter(lvl)
flaps=sum(1 for a,b in zip(lvl,lvl[1:]) if a!=b)
ids=set(re.findall(r'person#(\d+)',log))
dmins=[float(x) for x in re.findall(r'd_min=([0-9.]+)',log)]
vmax=[float(x) for x in re.findall(r'v_veh_max=([0-9.]+)',log)]
# false-positive proxy: EMERGENCY lines whose vehicle speed is barely over gate
print(f'EMERGENCY(L3)={c.get(\"3\",0)}  WARNING(L2)={c.get(\"2\",0)}  AWARENESS(L1)={c.get(\"1\",0)}  SAFE(L0)={c.get(\"0\",0)}')
print(f'level transitions (flaps)={flaps}')
print(f'unique person track IDs={len(ids)}')
print(f'min d_min={min(dmins) if dmins else None}  max v_veh_max={max(vmax) if vmax else None}')
PY
  "
