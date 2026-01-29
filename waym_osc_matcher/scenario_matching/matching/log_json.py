# scenario_matching/matching/log_json.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Tuple, Iterable
import numpy as np
from pathlib import Path

from scenario_matching.matching.results.ops import intervals_to_mask
from scenario_matching.matching.spec import build_block_query

# Public API
__all__ = [
    "append_header_json",
    "append_batch_json",
    # Optional: granular appenders similar to pretty.py
    "append_block_results_json",
    "append_calls_in_block_json",
    "append_calls_all_blocks_json",
    "append_calls_flat_json",
]

# ---------------------------
# small helpers (unchanged-ish)
# ---------------------------


def _roles_key(roles: Mapping[str, Any]) -> Tuple[Tuple[str, str], ...]:
    """Stable key for a roles mapping."""
    return tuple(sorted((str(k), str(v)) for k, v in (roles or {}).items()))

def _pair_intervals(intervals: Iterable[Any]) -> List[Tuple[int, int]]:
    """Normalize to list of (t0, t1) ints."""
    return [(int(_iv_t0(iv)), int(_iv_t1(iv))) for iv in (intervals or [])]

def _intervals_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    """Closed-interval overlap check for integer frames."""
    return (a[0] <= b[1]) and (b[0] <= a[1])

def _any_overlap(aset: List[Tuple[int, int]], bset: List[Tuple[int, int]]) -> bool:
    for a in aset:
        for b in bset:
            if _intervals_overlap(a, b):
                return True
    return False

def preview_mask(mask: np.ndarray, width: int = 100) -> str:
    if mask.size <= width:
        return "".join("#" if b else "." for b in mask)
    idx = np.linspace(0, mask.size - 1, num=width).astype(int)
    return "".join("#" if mask[i] else "." for i in idx)

def fmt_intervals(ivs: Iterable[Any]) -> str:
    return ", ".join(f"[{_iv_t0(iv)},{_iv_t1(iv)}]" for iv in ivs)

def roles_str(roles: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(roles.items()))

def _check_labels(checks) -> List[str]:
    return [getattr(c, "_label", f"check#{i}") for i, c in enumerate(checks)]

def _iv_t0(iv) -> int:
    if isinstance(iv, (list, tuple)) and len(iv) >= 2:
        return int(iv[0])
    if hasattr(iv, "t0"):
        return int(getattr(iv, "t0"))
    # last resort: try mapping
    if isinstance(iv, dict) and "t0" in iv:
        return int(iv["t0"])
    raise TypeError(f"Unsupported interval item for t0: {type(iv)}")

def _iv_t1(iv) -> int:
    if isinstance(iv, (list, tuple)) and len(iv) >= 2:
        return int(iv[1])
    if hasattr(iv, "t1"):
        return int(getattr(iv, "t1"))
    if isinstance(iv, dict) and "t1" in iv:
        return int(iv["t1"])
    raise TypeError(f"Unsupported interval item for t1: {type(iv)}")

def _jsonable_intervals(intervals: Iterable[Any]) -> List[List[int]]:
    out: List[List[int]] = []
    for iv in intervals:
        out.append([_iv_t0(iv), _iv_t1(iv)])
    return out

def _frames_on(T: int, intervals: Iterable[Any], mask: Optional[np.ndarray]) -> int:
    if mask is not None:
        return int(np.asarray(mask, dtype=bool).sum())
    # compute from intervals
    if T is None:
        return 0
    m = intervals_to_mask(intervals, int(T))
    return int(m.sum())

def _to_py(obj: Any) -> Any:
    """
    Make objects JSON-friendly:
    - numpy scalars/arrays -> Python types/lists
    - tuples -> lists
    - fallback to str for unknowns
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    return obj

def _dump_jsonl(path: str, obj: Dict[str, Any], *, indent: Optional[int] = None) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=_to_py, indent=indent))
        f.write("\n")

# ---------------------------
# Header (once per run)
# ---------------------------

def append_header_json(
    path: str,
    *,
    osc_name: Optional[str] = None,
    calls_flat: Optional[List[Dict[str, Any]]] = None,
    meta: Optional[Dict[str, Any]] = None,
    pretty: bool = False,
) -> None:
    """
    Write a single JSON header line. Call once at the start of the run.

    Example object:
    {
      "type": "header",
      "osc_name": "foo.osc",
      "calls_flat": [...],   # optional
      "meta": {...}          # optional
    }
    """
    obj: Dict[str, Any] = {"type": "header"}
    if osc_name:
        obj["osc_name"] = osc_name
    if calls_flat is not None:
        obj["calls_flat"] = calls_flat
    if meta:
        obj["meta"] = meta
    _dump_jsonl(path, obj, indent=(2 if pretty else None))

# ---------------------------
# Batch appender (per source_uri)
# ---------------------------

def append_batch_json(
    path: str,
    *,
    source_uri: str,
    store,                 # ResultsStore
    block_hits,            # BlockHits
    plans: Mapping[str, Any],
    calls: List[Dict[str, Any]],
    include_timeline: bool = True,
    timeline_width: int = 100,
    include_check_labels: bool = False,
    fps_for_labels: int = 10,
    cfg_for_labels: Optional[Dict[str, Any]] = None,
    skip_empty: bool = True,
    pretty: bool = False,

    # NEW knobs:
    restrict_calls_to_block_hits: bool = False,   # only log calls that participate in some block hit
    require_interval_overlap: bool = True,        # require call intervals to overlap the block intervals
    drop_segments_without_block_hits: bool = True # when restricting, drop segments that have no block hits
) -> None:
    """
    Append a batch JSON object grouped per segment:
      - always include block hits (when present)
      - include call signals either:
          (a) all call signals with intervals (default), or
          (b) only call signals that belong to a block hit with same (label, segment, roles)
              and (optionally) overlapping intervals if `require_interval_overlap=True`.

    If the batch has no signals at all and skip_empty=True, nothing is written.
    """

    # --- index block hits by (label, segment, roles_key) for quick matching ---
    hit_index: Dict[Tuple[str, str, Tuple[Tuple[str, str], ...]], List[Tuple[int, int]]] = {}
    seg_blocks: Dict[str, List[Dict[str, Any]]] = {}

    for label, sigs_map in (block_hits or {}).items():
        if not sigs_map:
            continue
        for s in sigs_map.values():
            if not s.intervals:
                continue
            T = int(s.T or 0)
            iv_pairs = _pair_intervals(s.intervals)
            if not iv_pairs:
                continue
            frames_on = _frames_on(T, s.intervals, s.mask)

            # log block entry
            entry = {
                "kind": "block",
                "block_label": label,
                "segment_id": s.segment_id,
                "roles": dict(s.roles or {}),
                "intervals": [[a, b] for (a, b) in iv_pairs],
                "frames_on": frames_on,
                "T": T,
            }
            if include_timeline:
                mask = s.mask if s.mask is not None else intervals_to_mask(s.intervals, T)
                entry["timeline"] = preview_mask(mask, width=timeline_width)
            seg_blocks.setdefault(s.segment_id, []).append(entry)

            # index for filtering calls
            key = (label, s.segment_id, _roles_key(s.roles))
            hit_index.setdefault(key, []).extend(iv_pairs)

    # --- collect call signals, possibly restricting to block hits ---
    seg_calls: Dict[str, List[Dict[str, Any]]] = {}

    for (block_label, call_idx), sigs in (getattr(store, "by_call", {}) or {}).items():
        if not sigs:
            continue
        # only calls from the same block label matter when restricting
        plan_block_label = block_label

        # fetch call metadata
        call = calls[call_idx] if 0 <= call_idx < len(calls) else {}
        actor = call.get("actor")
        action = call.get("action")
        modifiers = [m.get("name") for m in (call.get("modifiers") or [])]
        call_block_label = call.get("block_label", plan_block_label)

        # optionally compile labels once
        labels: List[str] = []
        if include_check_labels:
            Q, _ = build_block_query(call, fps=fps_for_labels, cfg=(cfg_for_labels or {}))
            labels = _check_labels(Q.checks)

        for s in sigs:
            if not s.intervals:
                continue
            T = int(s.T or 0)
            iv_pairs = _pair_intervals(s.intervals)
            if not iv_pairs:
                continue

            # If restricted: keep only if there is a matching block hit on (label, segment, roles)
            if restrict_calls_to_block_hits:
                key = (call_block_label, s.segment_id, _roles_key(s.roles))
                block_iv_pairs = hit_index.get(key)
                if not block_iv_pairs:
                    # no block hit with same label/segment/roles → drop this call signal
                    continue
                if require_interval_overlap and not _any_overlap(iv_pairs, block_iv_pairs):
                    # no temporal overlap with the block intervals → drop
                    continue

            frames_on = _frames_on(T, s.intervals, s.mask)
            entry: Dict[str, Any] = {
                "kind": "call",
                "block_label": call_block_label,
                "call_index": int(call_idx),
                "segment_id": s.segment_id,
                "roles": dict(s.roles or {}),
                "actor": actor,
                "action": action,
                "modifiers": modifiers,
                "intervals": [[a, b] for (a, b) in iv_pairs],
                "frames_on": frames_on,
                "T": T,
            }
            if include_timeline:
                mask = s.mask if s.mask is not None else intervals_to_mask(s.intervals, T)
                entry["timeline"] = preview_mask(mask, width=timeline_width)
            if labels:
                entry["checks"] = labels
            seg_calls.setdefault(s.segment_id, []).append(entry)

    # --- merge per-segment, optionally dropping segments without block hits when restricted ---
    all_seg_ids = set(seg_calls.keys()) | set(seg_blocks.keys())
    segments_obj: Dict[str, Dict[str, Any]] = {}

    for seg_id in sorted(all_seg_ids):
        calls_list = seg_calls.get(seg_id, [])
        blocks_list = seg_blocks.get(seg_id, [])

        if restrict_calls_to_block_hits and drop_segments_without_block_hits:
            # keep only segments that actually have block hits
            if not blocks_list:
                continue

        # If both empty, skip
        if not calls_list and not blocks_list:
            continue

        segments_obj[seg_id] = {
            "calls": calls_list,
            "blocks": blocks_list,
        }

    # respect skip_empty
    if not segments_obj and skip_empty:
        return

    obj = {
        "type": "batch",
        "source_uri": source_uri,
        "segments": segments_obj,
    }
    _dump_jsonl(path, obj, indent=(2 if pretty else None))
# ---------------------------
# Optional: granular JSON appenders (parallels to pretty.py)
# These don’t group by segment; they just append what you ask for.
# Useful if you want smaller events instead of per-batch objects.
# ---------------------------

def append_block_results_json(
    path: str,
    *,
    plans: Mapping[str, Any],
    block_hits,
    source_uri: Optional[str] = None,
    include_timeline: bool = True,
    timeline_width: int = 100,
    pretty: bool = False,
) -> None:
    for label, plan in plans.items():
        sigs_map = block_hits.get(label, {})
        if not sigs_map:
            continue
        for s in sigs_map.values():
            T = int(s.T or 0)
            if not s.intervals:
                continue
            entry = {
                "type": "block_signal",
                "source_uri": source_uri,
                "block_label": label,
                "segment_id": s.segment_id,
                "roles": dict(s.roles or {}),
                "intervals": _jsonable_intervals(s.intervals),
                "frames_on": _frames_on(T, s.intervals, s.mask),
                "T": T,
                "plan_type": getattr(plan, "type", None),
            }
            if include_timeline:
                mask = s.mask if s.mask is not None else intervals_to_mask(s.intervals, T)
                entry["timeline"] = preview_mask(mask, width=timeline_width)
            _dump_jsonl(path, entry, indent=(2 if pretty else None))

def append_calls_in_block_json(
    path: str,
    *,
    plans: Mapping[str, Any],
    calls: List[Dict[str, Any]],
    store,
    block_label: str,
    include_checks: bool = True,
    fps: int = 10,
    cfg: Optional[Dict[str, Any]] = None,
    include_timeline: bool = True,
    timeline_width: int = 100,
    pretty: bool = False,
) -> None:
    plan = plans.get(block_label)
    if not plan:
        return
    for ci in plan.indices:
        call = calls[ci]
        ck = (call.get("block_label", block_label), ci)
        signals = store.by_call.get(ck, [])
        if not signals:
            continue
        labels = []
        if include_checks:
            Q, _ = build_block_query(call, fps=fps, cfg=(cfg or {}))
            labels = _check_labels(Q.checks)

        for s in signals:
            if not s.intervals:
                continue
            T = int(s.T or 0)
            entry = {
                "type": "call_signal",
                "block_label": call.get("block_label", block_label),
                "call_index": int(ci),
                "segment_id": s.segment_id,
                "roles": dict(s.roles or {}),
                "actor": call.get("actor"),
                "action": call.get("action"),
                "modifiers": [m.get("name") for m in (call.get("modifiers") or [])],
                "intervals": _jsonable_intervals(s.intervals),
                "frames_on": _frames_on(T, s.intervals, s.mask),
                "T": T,
            }
            if include_timeline:
                mask = s.mask if s.mask is not None else intervals_to_mask(s.intervals, T)
                entry["timeline"] = preview_mask(mask, width=timeline_width)
            if labels:
                entry["checks"] = labels
            _dump_jsonl(path, entry, indent=(2 if pretty else None))

def append_calls_all_blocks_json(
    path: str,
    *,
    plans: Mapping[str, Any],
    calls: List[Dict[str, Any]],
    store,
    include_checks: bool = True,
    fps: int = 10,
    cfg: Optional[Dict[str, Any]] = None,
    include_timeline: bool = True,
    timeline_width: int = 100,
    pretty: bool = False,
) -> None:
    for label in plans.keys():
        append_calls_in_block_json(
            path,
            plans=plans,
            calls=calls,
            store=store,
            block_label=label,
            include_checks=include_checks,
            fps=fps,
            cfg=cfg,
            include_timeline=include_timeline,
            timeline_width=timeline_width,
            pretty=pretty,
        )

def append_calls_flat_json(
    path: str,
    *,
    calls: List[Dict[str, Any]],
    store,
    include_checks: bool = True,
    fps: int = 10,
    cfg: Optional[Dict[str, Any]] = None,
    include_timeline: bool = True,
    timeline_width: int = 100,
    pretty: bool = False,
) -> None:
    for i, call in enumerate(calls):
        ck = (call.get("block_label", "<none>"), i)
        signals = store.by_call.get(ck, [])
        if not signals:
            continue
        labels = []
        if include_checks:
            Q, _ = build_block_query(call, fps=fps, cfg=(cfg or {}))
            labels = _check_labels(Q.checks)

        for s in signals:
            if not s.intervals:
                continue
            T = int(s.T or 0)
            entry = {
                "type": "call_signal",
                "block_label": call.get("block_label", "<none>"),
                "call_index": int(i),
                "segment_id": s.segment_id,
                "roles": dict(s.roles or {}),
                "actor": call.get("actor"),
                "action": call.get("action"),
                "modifiers": [m.get("name") for m in (call.get("modifiers") or [])],
                "intervals": _jsonable_intervals(s.intervals),
                "frames_on": _frames_on(T, s.intervals, s.mask),
                "T": T,
            }
            if include_timeline:
                mask = s.mask if s.mask is not None else intervals_to_mask(s.intervals, T)
                entry["timeline"] = preview_mask(mask, width=timeline_width)
            if labels:
                entry["checks"] = labels
            _dump_jsonl(path, entry, indent=(2 if pretty else None))


def write_example_windows_jsonl(
    path: str | Path,
    *,
    osc: str,
    fps: int,
    source_uri: str,
    block_hits: Mapping[str, Any],
) -> int:
    """
    Append 1 JSONL record per (block, segment, binding) that has an example_window.
    Returns number of written records.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with path.open("a", encoding="utf-8") as f:
        for block_label, segmap in (block_hits or {}).items():
            # segmap: {seg_id -> {roles_key -> BlockSignal}}
            for seg_id, bindmap in (segmap or {}).items():
                for _roles_key, bs in (bindmap or {}).items():
                    ex = getattr(bs, "example_window", None)
                    if not ex:
                        continue
                    t0, t1_first, t1_greedy = ex
                    rec = {
                        "type": "example_window",
                        "osc": osc,
                        "block": block_label,
                        "segment_id": seg_id,
                        "roles": getattr(bs, "roles", {}),
                        "T": int(getattr(bs, "T", 0) or 0),
                        "fps": int(fps),
                        "t0": int(t0),
                        "t1_first": int(t1_first),
                        "t1_greedy": int(t1_greedy),
                        "source_uri": source_uri,
                    }
                    f.write(json.dumps(rec) + "\n")
                    n += 1
    return n
