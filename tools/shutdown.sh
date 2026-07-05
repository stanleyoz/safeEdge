#!/bin/bash
# SafeEdge shutdown — run from the LAPTOP to stop the live demo and cut
# further Qwen API usage.
#
# The Alibaba side (Function Compute + Tablestore) has no idle cost — FC
# scales to zero between requests and Tablestore reserved throughput is 0
# (pay-per-use). Qwen credits are only spent when the edge pipeline posts
# incidents, so stopping the Jetson container is what actually stops spend.
# This script does NOT delete/undeploy the Alibaba backend — it stays up
# so the dashboard URL keeps resolving; it just sits idle (free) until the
# Jetson is started again.
#
# Usage:
#   bash tools/shutdown.sh
#   JETSON=icp@eliza-jetson.tail52ce98.ts.net bash tools/shutdown.sh

set -u
JETSON="${JETSON:-icp@eliza-jetson.tail52ce98.ts.net}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_jetson}"

ssh_c(){ ssh -i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 "$JETSON" "$@"; }

echo "=== SafeEdge shutdown ==="
echo "Jetson: $JETSON"
echo

echo "[1] Stopping pipeline container(s)..."
IDS=$(ssh_c 'docker ps -q --filter ancestor=dustynv/l4t-pytorch:r36.4.0' 2>/dev/null)
if [ -z "$IDS" ]; then
  echo "  none running."
else
  ssh_c "docker stop $IDS"
  echo "  stopped: $IDS"
fi

echo
echo "=== DOWN — Jetson pipeline stopped, no further Qwen calls will be made ==="
echo "(Alibaba backend left deployed; it is zero-cost while idle. Re-run tools/startup.sh to resume.)"
