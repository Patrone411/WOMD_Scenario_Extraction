#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Iterable, Optional, Tuple

import boto3


def iter_json_keys(
    s3_client,
    *,
    bucket: str,
    prefix: str,
) -> Iterable[str]:
    """List all *.json objects under prefix (recursively)."""
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            if key.endswith(".json") and not key.endswith(".jsonl"):
                yield key


def get_json_object_from_s3(s3_client, *, bucket: str, key: str) -> Optional[dict]:
    """Load a single JSON object from S3."""
    try:
        resp = s3_client.get_object(Bucket=bucket, Key=key)
        raw = resp["Body"].read()
        return json.loads(raw.decode("utf-8"))
    except Exception as e:
        print(f"[WARN] Failed to read/parse {key}: {e}")
        return None


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="waymo")
    ap.add_argument("--prefix", default="results/segments/")
    ap.add_argument("--endpoint_url", default="https://gif.s3.iavgroup.local")
    ap.add_argument(
        "--verify",
        default="false",
        help='TLS verify: "false"/"true" oder Pfad zum CA-Bundle (z.B. /etc/iav-ca/IAV-CA-Bundle.pem)',
    )
    ap.add_argument("--max_files", type=int, default=None, help="Optional: nur erste N Dateien (Test)")
    args = ap.parse_args()

    # verify-Parsing (wie in deinen anderen Skripten)
    v = str(args.verify).lower()
    if v in ("true", "1", "yes"):
        verify: object = True
    elif v in ("false", "0", "no"):
        verify = False
    else:
        verify = args.verify
        if isinstance(verify, str) and not os.path.exists(verify):
            print(f"[WARN] CA bundle not found: {verify} -> disabling TLS verification")
            verify = False

    s3 = boto3.client("s3", endpoint_url=args.endpoint_url, verify=verify)

    keys = list(iter_json_keys(s3, bucket=args.bucket, prefix=args.prefix))
    keys.sort()
    if args.max_files is not None:
        keys = keys[: args.max_files]

    print(f"Found {len(keys)} *.json files under s3://{args.bucket}/{args.prefix}")

    sum_covered = 0.0
    sum_total = 0.0
    min_pct: Optional[float] = None
    max_pct: Optional[float] = None
    sum_pct = 0.0
    n_pct = 0
    n_files_parsed = 0

    for key in keys:
        obj = get_json_object_from_s3(s3, bucket=args.bucket, key=key)
        if not obj:
            continue

        n_files_parsed += 1

        covered = safe_float(obj.get("covered_value"))
        total = safe_float(obj.get("total_value"))
        pct = safe_float(obj.get("percentage"))

        if covered is not None:
            sum_covered += covered
        if total is not None:
            sum_total += total

        if pct is not None:
            sum_pct += pct
            n_pct += 1
            min_pct = pct if min_pct is None else min(min_pct, pct)
            max_pct = pct if max_pct is None else max(max_pct, pct)

        if n_files_parsed % 200 == 0:
            print(f"  parsed {n_files_parsed} files...")

    print("\n=== Aggregation Result ===")
    print(f"Files parsed successfully: {n_files_parsed}")

    print(f"Sum covered_value: {sum_covered}")
    print(f"Sum total_value:   {sum_total}")

    if n_pct == 0:
        print("No percentage values found.")
    else:
        avg_pct = sum_pct / n_pct
        print(f"Min percentage: {min_pct}")
        print(f"Max percentage: {max_pct}")
        print(f"Avg percentage: {avg_pct}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
