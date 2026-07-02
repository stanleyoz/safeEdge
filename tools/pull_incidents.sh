#!/bin/bash
# Pull the Qwen incident reports + events after a shoot, newest first, so you
# can quote the best one in the submission video / write-up.
#
# Usage:  bash tools/pull_incidents.sh [N]      (default 10)
N="${1:-10}"
FC="${SAFEEDGE_CLOUD_URL:-https://safeedg-backend-nkmqevdhff.ap-southeast-1.fcapp.run}"

echo "=== Last $N incidents (Qwen reasoning) ==="
curl -sS -m 30 "$FC/api/incidents?limit=$N" \
  | python3 -c '
import sys,json,datetime
d=json.load(sys.stdin)
items=d if isinstance(d,list) else d.get("items",d.get("incidents",[]))
for it in items:
    ts=it.get("timestamp") or it.get("t")
    when=datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S") if isinstance(ts,(int,float)) else ts
    lvl=it.get("level_label") or it.get("level")
    txt=it.get("report") or it.get("summary") or it.get("text") or ""
    print(f"\n[{when}] {lvl}\n{txt}")
' 2>/dev/null || echo "(no incidents or unexpected shape — raw:)"

echo
echo "=== Last $N events (level transitions) ==="
curl -sS -m 30 "$FC/api/events?limit=$N" \
  | python3 -c '
import sys,json,datetime
d=json.load(sys.stdin)
items=d if isinstance(d,list) else d.get("items",d.get("events",[]))
for e in items:
    ts=e.get("timestamp") or e.get("t")
    when=datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S") if isinstance(ts,(int,float)) else ts
    print(f"{when}  {e.get(\"level_label\",e.get(\"level\",\"?\"))}  d_min={e.get(\"signals\",{}).get(\"d_min\",\"?\")}")
' 2>/dev/null
