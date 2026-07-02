#!/bin/bash
# SafeEdge recording preflight — run from the LAPTOP before a shoot.
# Verifies the full edge→cloud→dashboard chain is live and fresh, so the
# precious pedestrian hour is spent recording, not debugging.
#
# Usage:
#   bash tools/preflight.sh
#   JETSON=icp@eliza-jetson.tail52ce98.ts.net bash tools/preflight.sh
#
# Exit 0 = all green. Non-zero = something to fix before the helper arrives.

set -u
FC="${SAFEEDGE_CLOUD_URL:-https://safeedg-backend-nkmqevdhff.ap-southeast-1.fcapp.run}"
DASH="${SAFEEDGE_DASH_URL:-https://stanleyoz.github.io/safeEdge/}"
JETSON="${JETSON:-icp@eliza-jetson.tail52ce98.ts.net}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_jetson}"
MAX_AGE="${MAX_AGE:-10}"     # cloud state must be fresher than this many seconds
FAIL=0

ok(){   printf "  \033[32mOK\033[0m   %s\n" "$1"; }
bad(){  printf "  \033[31mFAIL\033[0m %s\n" "$1"; FAIL=1; }
info(){ printf "  ..   %s\n" "$1"; }

echo "=== SafeEdge preflight ==="
echo "FC:      $FC"
echo "Dash:    $DASH"
echo "Jetson:  $JETSON"
echo

# 1. Dashboard reachable (GitHub Pages up)
echo "[1] Dashboard (GitHub Pages)"
H=$(curl -sS -m 15 -o /dev/null -w "%{http_code}" "$DASH" 2>/dev/null)
[ "$H" = "200" ] && ok "dashboard HTTP $H" || bad "dashboard HTTP $H (Pages down? see Settings→Pages, main /docs)"

# 2. Backend healthy
echo "[2] Cloud backend /healthz"
HB=$(curl -sS -m 20 "$FC/healthz" 2>/dev/null)
echo "$HB" | grep -q '"status":"ok"' && ok "backend ok ($(echo "$HB" | grep -o '"region":"[^"]*"'))" || bad "healthz not ok: $HB"

# 3. Live state freshness (is the Jetson pipeline actually posting NOW?)
echo "[3] Cloud state freshness (< ${MAX_AGE}s = live)"
NOW=$(date +%s)
S=$(curl -sS -m 20 "$FC/api/state/latest" 2>/dev/null)
if [ -z "$S" ]; then
  bad "no state returned"
else
  AGE=$(echo "$S" | python3 -c 'import sys,json,time;d=json.load(sys.stdin);t=d.get("timestamp") or d.get("t") or 0;print(round(time.time()-t,1))' 2>/dev/null)
  LVL=$(echo "$S" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("level_label","?"))' 2>/dev/null)
  DMIN=$(echo "$S" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("signals",{}).get("d_min","?"))' 2>/dev/null)
  info "level=$LVL  d_min=$DMIN  age=${AGE}s"
  awk "BEGIN{exit !($AGE < $MAX_AGE)}" && ok "state is live (age ${AGE}s)" \
    || bad "state STALE (age ${AGE}s) — pipeline not posting; restart docker_live.sh"
fi

# 4. Frame is changing (not a frozen image)
echo "[4] Frame updating (two grabs, 4s apart)"
M1=$(curl -sS -m 20 "$FC/api/state/latest" 2>/dev/null | python3 -c 'import sys,json,hashlib;print(hashlib.md5((json.load(sys.stdin).get("frame_jpeg_b64") or "").encode()).hexdigest())' 2>/dev/null)
sleep 4
M2=$(curl -sS -m 20 "$FC/api/state/latest" 2>/dev/null | python3 -c 'import sys,json,hashlib;print(hashlib.md5((json.load(sys.stdin).get("frame_jpeg_b64") or "").encode()).hexdigest())' 2>/dev/null)
[ -n "$M1" ] && [ "$M1" != "$M2" ] && ok "frame changing (live camera)" || bad "frame NOT changing ($M1) — camera stalled or pipeline dead"

# 5. Jetson container up + only ONE (avoid double-launch webcam clash)
echo "[5] Jetson container (exactly one)"
CT=$(ssh -i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 "$JETSON" \
       'docker ps -q --filter ancestor=dustynv/l4t-pytorch:r36.4.0 | wc -l' 2>/dev/null)
case "$CT" in
  1) ok "one pipeline container running" ;;
  0) bad "no container running — start: nohup bash tools/docker_live.sh --source webcam > /tmp/safeedge_live.log 2>&1 &" ;;
  "") bad "could not SSH to Jetson (key $SSH_KEY?)" ;;
  *) bad "$CT containers running — DOUBLE LAUNCH; kill all then start one: docker ps -q --filter ancestor=dustynv/l4t-pytorch:r36.4.0 | xargs -r docker kill" ;;
esac

echo
if [ "$FAIL" = "0" ]; then
  printf "\033[32m=== ALL GREEN — safe to record ===\033[0m\n"
else
  printf "\033[31m=== NOT READY — fix the FAILs above ===\033[0m\n"
fi
exit $FAIL
