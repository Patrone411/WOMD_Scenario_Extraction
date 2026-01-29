#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, Optional, Tuple

import boto3


NUM_FIELDS_BLOCK = [
    "bindings_enum_with_valid_windows",
    "bindings_with_hits",
    "P_win_uncond",
    "P_win_cond",
    "bindings_enum_total",
    "P_binding",
]
NUM_FIELDS_META = [
    "n_pickles_processed",
]


def iter_keys_ending_with(
    s3_client,
    *,
    bucket: str,
    prefix: str,
    filename: str,
) -> Iterable[str]:
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            if key.endswith("/" + filename) or key.endswith(filename):
                yield key


def get_json_object_from_s3(s3_client, *, bucket: str, key: str) -> Optional[dict]:
    """Lädt ein einzelnes JSON-Objekt aus S3 (kein JSONL)."""
    try:
        resp = s3_client.get_object(Bucket=bucket, Key=key)
        raw = resp["Body"].read()
        return json.loads(raw.decode("utf-8"))
    except Exception as e:
        print(f"[WARN] Failed to read/parse {key}: {e}")
        return None


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def safe_int(x) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="waymo")
    ap.add_argument("--prefix", default="results/matching_fresh/phaseB_last_4")
    ap.add_argument("--filename", default="stats_shard.json")  # falls bei dir anders, hier ändern

    ap.add_argument("--endpoint_url", default="https://gif.s3.iavgroup.local")
    ap.add_argument(
        "--verify",
        default="false",
        help='TLS verify: "false"/"true" oder Pfad zum CA-Bundle (z.B. /etc/iav-ca/IAV-CA-Bundle.pem)',
    )

    ap.add_argument(
        "--block",
        default=None,
        help="Optional: nur einen bestimmten Block aggregieren (z.B. get_close). Default: alle Blocks summieren.",
    )
    ap.add_argument("--max_files", type=int, default=None, help="Optional: nur erste N Dateien (Test)")

    args = ap.parse_args()

    # verify-Parsing
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

    keys = list(iter_keys_ending_with(s3, bucket=args.bucket, prefix=args.prefix, filename=args.filename))
    keys.sort()
    if args.max_files is not None:
        keys = keys[: args.max_files]

    print(f"Found {len(keys)} files matching '{args.filename}' under s3://{args.bucket}/{args.prefix}")

    totals: Dict[str, float] = {k: 0.0 for k in (NUM_FIELDS_BLOCK + NUM_FIELDS_META)}
    n_files = 0

    for key in keys:
        obj = get_json_object_from_s3(s3, bucket=args.bucket, key=key)
        if not obj:
            continue

        n_files += 1

        meta = obj.get("meta", {}) or {}
        totals["n_pickles_processed"] += safe_int(meta.get("n_pickles_processed"))

        blocks = obj.get("blocks", {}) or {}

        # Pro Datei: entweder 1 Block oder Summe über alle Blocks
        if args.block is not None:
            block_items = [(args.block, blocks.get(args.block, {}) or {})]
        else:
            block_items = list(blocks.items())

        for _, b in block_items:
            totals["bindings_enum_with_valid_windows"] += safe_int(b.get("bindings_enum_with_valid_windows"))
            totals["bindings_with_hits"] += safe_int(b.get("bindings_with_hits"))
            totals["bindings_enum_total"] += safe_int(b.get("bindings_enum_total"))
            totals["P_win_uncond"] += safe_float(b.get("P_win_uncond"))
            totals["P_win_cond"] += safe_float(b.get("P_win_cond"))
            totals["P_binding"] += safe_float(b.get("P_binding"))

        if n_files % 50 == 0:
            print(f"  processed {n_files} files...")

    print("\n=== Aggregation Result ===")
    print(f"Files parsed (n_files): {n_files}")
    if n_files == 0:
        print("No files parsed successfully.")
        return 1

    # Summen
    print("\n-- Sums --")
    for k in ["n_pickles_processed"] + NUM_FIELDS_BLOCK:
        print(f"{k}: {totals[k]}")

    # Averages (Summe / n_files)
    print("\n-- Averages (sum / n_files) --")
    for k in ["n_pickles_processed"] + NUM_FIELDS_BLOCK:
        print(f"{k}_avg: {totals[k] / n_files}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
