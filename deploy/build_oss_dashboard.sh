#!/bin/bash
# Generate an OSS-ready dashboard (deploy/oss/index.html) with the Function
# Compute API URL baked in, so the public OSS-hosted page calls the FC API
# directly (no ?api= needed). Source of truth stays dashboard/static/index.html.
#
# Usage:  FC_URL=https://<fc-url> ./deploy/build_oss_dashboard.sh
set -euo pipefail

FC_URL="${FC_URL:-https://safeedg-backend-nkmqevdhff.ap-southeast-1.fcapp.run}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/deploy/oss"
mkdir -p "$OUT"

python3 - "$ROOT/dashboard/static/index.html" "$OUT/index.html" "$FC_URL" <<'PY'
import sys
src, out, url = sys.argv[1], sys.argv[2], sys.argv[3]
html = open(src).read()
inject = f'<script>window.SAFEEDGE_API="{url}";</script>\n'
marker = "<script>\nconst LEVEL_COLOURS"
assert marker in html, "could not find main script marker"
html = html.replace(marker, inject + marker, 1)
open(out, "w").write(html)
print(f"wrote {out}\n  API base baked in -> {url}")
PY
