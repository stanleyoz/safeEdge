#!/bin/bash
# SafeEdge startup — run from the LAPTOP to bring the live demo up.
# Launches the edge pipeline on the Jetson (USB webcam → Alibaba backend).
# The Alibaba backend itself needs no startup: Function Compute scales up
# on the first request automatically.
#
# Usage:
#   bash tools/startup.sh
#   JETSON=icp@eliza-jetson.tail52ce98.ts.net bash tools/startup.sh

set -u
JETSON="${JETSON:-icp@eliza-jetson.tail52ce98.ts.net}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_jetson}"
REMOTE_DIR="${REMOTE_DIR:-/home/icp/safeedge/qwen_cloud}"
FC="${SAFEEDGE_CLOUD_URL:-https://safeedg-backend-nkmqevdhff.ap-southeast-1.fcapp.run}"

ssh_c(){ ssh -i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 "$JETSON" "$@"; }

echo "=== SafeEdge startup ==="
echo "Jetson: $JETSON"
echo

echo "[1] Checking for an already-running pipeline container..."
CT=$(ssh_c 'docker ps -q --filter ancestor=dustynv/l4t-pytorch:r36.4.0 | wc -l' 2>/dev/null)
if [ "$CT" != "0" ] && [ -n "$CT" ]; then
  echo "  already running ($CT container(s)) — nothing to do. Use shutdown.sh first to restart clean."
  exit 0
fi

echo "[2] Launching docker_live.sh (--source webcam) on the Jetson..."
ssh_c "cd '$REMOTE_DIR' && nohup bash tools/docker_live.sh --source webcam > /tmp/safeedge_live.log 2>&1 < /dev/null & disown; sleep 1; echo launched"

echo "[3] Waiting for cloud state to go live (up to 60s)..."
for i in $(seq 1 12); do
  sleep 5
  S=$(curl -sS -m 10 "$FC/api/state/latest" 2>/dev/null)
  AGE=$(echo "$S" | python3 -c 'import sys,json,time;d=json.load(sys.stdin);t=d.get("timestamp") or d.get("t") or 0;print(round(time.time()-t,1))' 2>/dev/null)
  if [ -n "$AGE" ] && awk "BEGIN{exit !($AGE < 10)}"; then
    echo "  live (state age ${AGE}s)"
    echo
    echo "=== UP — dashboard: https://stanleyoz.github.io/safeEdge/ ==="
    exit 0
  fi
  echo "  ...not live yet ($i/12)"
done

echo "FAIL — pipeline did not report live state within 60s. Check: ssh $JETSON 'tail -f /tmp/safeedge_live.log'"
exit 1
