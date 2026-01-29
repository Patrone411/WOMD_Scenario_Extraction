#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Set, Tuple


def iter_example_files(root: Path):
    if root.is_file():
        yield root
        return
    # typischerweise: <osc>/<shard_tag>/example_windows.jsonl
    for p in root.rglob("example_windows.jsonl"):
        yield p


def binding_key(rec: Dict[str, Any]) -> str:
    """
    Key für Dedupe: pro (osc, block, segment_id, roles) sollte es genau 1 Beispiel geben.
    Falls du pro Binding irgendwann mehrere Beispiele erlaubst, nimm zusätzlich t0 o.ä. in den Key.
    """
    osc = rec.get("osc", "")
    block = rec.get("block", "")
    seg = rec.get("segment_id", "")
    roles = rec.get("roles") or {}
    return f"{osc}|{block}|{seg}|{json.dumps(roles, sort_keys=True)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Ordner, der die heruntergeladenen Shard-Ordner enthält.")
    ap.add_argument("--out", default="example_windows_merged.jsonl", help="Output JSONL Pfad.")
    ap.add_argument("--dedupe", action="store_true", help="Dedupe pro (osc, block, segment_id, roles).")
    ap.add_argument("--write_summary", action="store_true", help="Schreibe zusätzlich example_windows_summary.json")
    args = ap.parse_args()

    in_root = Path(args.in_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: Set[str] = set()
    n_files = 0
    n_in = 0
    n_out = 0
    n_bad = 0

    # einfache Summary: counts pro (osc, block)
    counts: Dict[Tuple[str, str], int] = {}

    with out_path.open("w", encoding="utf-8") as fout:
        for fp in iter_example_files(in_root):
            n_files += 1
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    n_in += 1
                    try:
                        rec = json.loads(line)
                    except Exception:
                        n_bad += 1
                        continue

                    if rec.get("type") != "example_window":
                        continue

                    if args.dedupe:
                        k = binding_key(rec)
                        if k in seen:
                            continue
                        seen.add(k)

                    fout.write(json.dumps(rec) + "\n")
                    n_out += 1

                    osc = rec.get("osc", "")
                    block = rec.get("block", "")
                    counts[(osc, block)] = counts.get((osc, block), 0) + 1

    print(f"[OK] merged files={n_files}  lines_in={n_in}  lines_out={n_out}  bad_json={n_bad}")
    print(f"[OUT] {out_path.resolve()}")

    if args.write_summary:
        summary = {
            "n_files": n_files,
            "lines_in": n_in,
            "lines_out": n_out,
            "bad_json": n_bad,
            "dedupe": bool(args.dedupe),
            "counts_by_osc_block": [
                {"osc": osc, "block": block, "n": n}
                for (osc, block), n in sorted(counts.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
            ],
        }
        sum_path = out_path.with_name("example_windows_summary.json")
        sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[SUM] {sum_path.resolve()}")


if __name__ == "__main__":
    main()