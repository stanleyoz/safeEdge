"""Preflight check for Alibaba Cloud credentials + Tablestore reachability.

Reads backend/.env (gitignored), never prints secrets. Verifies:
  1. all required env vars are present
  2. the AccessKey can authenticate to Tablestore (list tables)
  3. the SafeEdge tables exist or can be created

Run:  python tools/check_alibaba_creds.py
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv("backend/.env")
load_dotenv()


def mask(v: str) -> str:
    return f"{v[:4]}…{v[-2:]}" if v and len(v) > 6 else "set"


REQUIRED = [
    "ALIBABA_CLOUD_ACCESS_KEY_ID",
    "ALIBABA_CLOUD_ACCESS_KEY_SECRET",
    "TABLESTORE_ENDPOINT",
    "TABLESTORE_INSTANCE",
]

print("=== 1. Environment ===")
missing = [k for k in REQUIRED if not os.environ.get(k)]
for k in REQUIRED:
    v = os.environ.get(k, "")
    show = mask(v) if "SECRET" in k or "KEY_ID" in k else v
    print(f"  {'✅' if v else '❌'} {k} = {show or '(missing)'}")
if missing:
    print(f"\nMissing: {missing}. Add them to backend/.env and re-run.")
    sys.exit(1)

print("\n=== 2. Tablestore authentication ===")
try:
    from tablestore import OTSClient
except ImportError:
    print("  ❌ tablestore SDK not installed. Run: pip install tablestore==6.1.0")
    sys.exit(1)

endpoint = os.environ["TABLESTORE_ENDPOINT"]
instance = os.environ["TABLESTORE_INSTANCE"]
ak_id = os.environ["ALIBABA_CLOUD_ACCESS_KEY_ID"]
ak_secret = os.environ["ALIBABA_CLOUD_ACCESS_KEY_SECRET"]

try:
    client = OTSClient(endpoint, ak_id, ak_secret, instance)
    tables = client.list_table()
    print(f"  ✅ Authenticated. Instance '{instance}' reachable.")
    print(f"  Existing tables: {list(tables) or '(none yet)'}")
except Exception as exc:  # noqa: BLE001
    msg = str(exc)
    print(f"  ❌ Tablestore call failed: {msg[:200]}")
    hint = ""
    if "InvalidAccessKeyId" in msg or "SignatureDoesNotMatch" in msg:
        hint = "AccessKey ID/Secret wrong or not yet active (can take ~1 min)."
    elif "denied" in msg.lower() or "Forbidden" in msg or "Permission" in msg:
        hint = "AccessKey valid but lacks Tablestore (OTS) permission — attach AliyunOTSFullAccess to the RAM user."
    elif "could not" in msg.lower() or "resolve" in msg.lower() or "endpoint" in msg.lower():
        hint = "Endpoint/instance mismatch. Check TABLESTORE_ENDPOINT matches the instance region."
    if hint:
        print(f"  → {hint}")
    sys.exit(1)

print("\n=== 3. Table bootstrap (auto-created by backend/store.py on first run) ===")
try:
    from backend.store import TablestoreStore
    store = TablestoreStore()           # creates the 4 SafeEdge tables if absent
    after = list(store._client.list_table())
    print(f"  ✅ SafeEdge tables ready: {after}")
except Exception as exc:  # noqa: BLE001
    print(f"  ⚠️ table bootstrap issue: {str(exc)[:200]}")
    print("  (non-fatal — tables also auto-create when the backend boots)")

print("\nPREFLIGHT OK — credentials valid, Tablestore reachable.")
