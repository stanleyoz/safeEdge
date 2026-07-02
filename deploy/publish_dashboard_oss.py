"""Publish the SafeEdge dashboard to Alibaba OSS static hosting — fully scripted.

Creates (if needed) a public-read bucket in ap-southeast-1, uploads the
OSS-ready dashboard (deploy/oss/index.html, built by build_oss_dashboard.sh),
configures static-website hosting, and prints the public URL.

Credentials come from backend/.env (ALIBABA_CLOUD_ACCESS_KEY_ID/SECRET).
The RAM user needs OSS permissions (AliyunOSSFullAccess) — if it doesn't, the
script prints exactly what to add.

Usage:
  python deploy/publish_dashboard_oss.py            # default bucket name
  BUCKET=my-bucket python deploy/publish_dashboard_oss.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / "backend" / ".env")
load_dotenv()

REGION = os.environ.get("OSS_REGION", "ap-southeast-1")
BUCKET = os.environ.get("BUCKET", "safeedge-dashboard")
ENDPOINT = f"https://oss-{REGION}.aliyuncs.com"
INDEX = ROOT / "deploy" / "oss" / "index.html"

ak_id = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID")
ak_secret = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_SECRET")

if not (ak_id and ak_secret):
    sys.exit("Missing ALIBABA_CLOUD_ACCESS_KEY_ID/SECRET (set in backend/.env)")
if not INDEX.exists():
    sys.exit(f"{INDEX} not found — run deploy/build_oss_dashboard.sh first")

try:
    import oss2
except ImportError:
    sys.exit("oss2 not installed — run: pip install oss2")


def main() -> None:
    auth = oss2.Auth(ak_id, ak_secret)
    bucket = oss2.Bucket(auth, ENDPOINT, BUCKET)

    # 1. create bucket (public-read) if it doesn't exist
    try:
        bucket.create_bucket(oss2.models.BUCKET_ACL_PUBLIC_READ)
        print(f"✅ created bucket '{BUCKET}' ({REGION}, public-read)")
    except oss2.exceptions.ServerError as e:
        if e.code in ("BucketAlreadyExists", "BucketAlreadyOwnedByYou"):
            print(f"• bucket '{BUCKET}' already exists — reusing")
            try:
                bucket.put_bucket_acl(oss2.models.BUCKET_ACL_PUBLIC_READ)
            except Exception:  # noqa: BLE001
                pass
        elif "AccessDenied" in str(e) or e.code == "AccessDenied":
            sys.exit("❌ AccessDenied creating bucket — attach AliyunOSSFullAccess "
                     "to the RAM user (RAM console → Users → Permissions).")
        else:
            raise
    except oss2.exceptions.AccessDenied:
        sys.exit("❌ AccessDenied — the RAM user needs AliyunOSSFullAccess.")

    # 2. upload the dashboard, rendered inline (text/html)
    bucket.put_object_from_file(
        "index.html", str(INDEX),
        headers={"Content-Type": "text/html; charset=utf-8"},
    )
    try:
        bucket.put_object_acl("index.html", oss2.OBJECT_ACL_PUBLIC_READ)
    except Exception:  # noqa: BLE001
        pass
    print("✅ uploaded index.html (Content-Type text/html, public-read)")

    # 3. static website hosting so '/' serves index.html too
    try:
        bucket.put_bucket_website(oss2.models.BucketWebsite("index.html", "index.html"))
        print("✅ static website hosting enabled")
    except Exception as e:  # noqa: BLE001
        print(f"• website config skipped: {str(e)[:80]}")

    url = f"https://{BUCKET}.oss-{REGION}.aliyuncs.com/index.html"
    print("\n🌐 Public dashboard URL:")
    print(f"   {url}")
    print("\nIf it returns 403, disable 'Block Public Access' on the bucket "
          "(one toggle) — otherwise the public-read ACL is overridden.")


if __name__ == "__main__":
    main()
