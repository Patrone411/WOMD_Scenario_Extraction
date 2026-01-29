#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

def _add(a: Any, b: Any) -> Any:
    """
    Recursively add mergeable structures:
      - numbers: sum
      - dict: key-wise add
      - lists: elementwise add if same length, else keep first and warn
      - None: treat as missing
    """
    if b is None:
        return a
    if a is None:
        return b

    # numbers
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a + b

    # dicts
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, vb in b.items():
            out[k] = _add(out.get(k), vb)
        return out

    # lists (hist bins sometimes stored as lists)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) == len(b) and all(isinstance(x, (int, float)) for x in a) and all(isinstance(x, (int, float)) for x in b):
            return [x + y for x, y in zip(a, b)]
        # if not trivially mergeable, keep a (you can extend later)
        return a

    # fallback: keep a (meta strings etc.)
    return a

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def iter_stats_files(root: Path) -> Iterable[Path]:
    # Accept either:
    # - a directory containing many stats_shard.json
    # - nested directories like <osc>/<shard>/stats_shard.json
    # - or a glob pattern
    if root.is_file():
        yield root
        return
    for p in root.rglob("stats_shard.json"):
        yield p
    # also allow stats.json naming
    for p in root.rglob("*.json"):
        if p.name == "stats_shard.json":
            continue
        # user may have renamed
        if p.name.startswith("stats") and p.suffix == ".json":
            yield p

def summarize(merged: Dict[str, Any]) -> str:
    lines = []
    blocks = merged.get("blocks", {}) or {}
    lines.append(f"Blocks: {len(blocks)}")
    for blk, d in blocks.items():
        bwh = d.get("bindings_with_hits", 0)
        ben = d.get("bindings_enum_with_valid_windows", 0)
        p_bind = (bwh / ben) if ben else 0.0

        pw_un = None
        hitw = d.get("hit_windows")
        poss = d.get("possible_windows_uncond")
        if isinstance(hitw, (int, float)) and isinstance(poss, (int, float)) and poss:
            pw_un = hitw / poss

        lines.append(f"- {blk}: bindings_with_hits={bwh}  denom={ben}  P_binding={p_bind:.6f}" + (f"  P_win_uncond={pw_un:.6f}" if pw_un is not None else ""))
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True, help="Local folder that contains many stats_shard.json (possibly nested).")
    ap.add_argument("--out", type=str, default="stats_merged.json", help="Output merged json path.")
    ap.add_argument("--print_summary", action="store_true")
    args = ap.parse_args()

    root = Path(args.in_dir)
    out_path = Path(args.out)

    merged: Dict[str, Any] = {}
    n = 0
    for fp in iter_stats_files(root):
        try:
            data = load_json(fp)
        except Exception as e:
            print(f"[WARN] skip {fp}: {e}")
            continue
        merged = _add(merged, data)
        n += 1

    merged.setdefault("meta", {})
    merged["meta"]["n_shards_merged"] = n

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"[OK] merged {n} shard files -> {out_path}")

    if args.print_summary:
        print("\n" + summarize(merged))

if __name__ == "__main__":
    main()
