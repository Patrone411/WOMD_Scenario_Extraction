from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple, Optional

def _build_lane_lookup_min(road_segments: Dict[str, Any]) -> Dict[Any, List[Dict[str, Any]]]:
    """
    lane_id -> list of { 'segment': seg_id, 'chain_id': int, 'osc_index': 1..N }
    Works even if lane_ids are ints or strings; we store as-is.
    """
    lookup: Dict[Any, List[Dict[str, Any]]] = {}
    for seg_id, seg_data in road_segments.items():
        for osc_index, chain in enumerate(seg_data.get("chains", []), start=1):
            chain_id = chain.get("id")
            for lid in chain.get("lane_ids", []):
                lookup.setdefault(lid, []).append({
                    "segment": seg_id,
                    "chain_id": chain_id,
                    "osc_index": int(osc_index),
                })
    return lookup

def _get_candidates(lookup: Dict[Any, List[Dict[str, Any]]], lid: Optional[Any]):
    """
    Robust lookup for mixed-type lane IDs.
    Tries original key, int(key), and str(key) in that order.
    """
    if lid is None:
        return None
    c = lookup.get(lid)
    if c:
        return c
    # try int -> str fallbacks
    try:
        c = lookup.get(int(lid))
        if c:
            return c
    except Exception:
        pass
    try:
        return lookup.get(str(lid))
    except Exception:
        return None

def kept_actors_from_per_actor_minimal(
    road_segments: Dict[str, Any],
    per_actor: Dict[str, Dict[str, Any]],  # output of per_actor_minimal
    min_steps: int = 5,
):
    """
    Returns:
      - always: (per_segment_kept_ids, all_kept_ids)
      - if return_payloads=True, also:
            (per_segment_payloads, all_kept_payloads)
    """
    lane_lookup = _build_lane_lookup_min(road_segments)
    counts: Dict[str, Counter] = defaultdict(Counter)  # seg_id -> Counter(actor_id -> frames)

    for actor_id, rec in per_actor.items():
        lane_series = rec.get("lane_id", []) or []
        v0, v1 = rec.get("valid", (0, len(lane_series)-1))
        v0 = max(0, int(v0))
        v1 = min(len(lane_series) - 1, int(v1)) if lane_series else -1
        if v1 < v0:
            continue

        # count frames only in the valid window
        for t in range(v0, v1 + 1):
            lid = lane_series[t]
            cands = _get_candidates(lane_lookup, lid)
            if not cands:
                continue

            # avoid double-counting the same segment within this timestep
            seen = set()
            for m in cands:
                seg_id = m["segment"]
                if seg_id in seen:
                    continue
                seen.add(seg_id)
                counts[seg_id][actor_id] += 1

    # kept IDs per segment (sorted), and union (sorted)
    per_segment_kept_ids: Dict[str, List[str]] = {
        seg_id: sorted([a for a, c in ctr.items() if c >= min_steps])
        for seg_id, ctr in counts.items()
    }
    all_kept_ids = sorted({a for ids in per_segment_kept_ids.values() for a in ids})

    # deterministic segment order
    per_segment_payloads: Dict[str, Dict[str, Any]] = {}
    for seg_id in sorted(per_segment_kept_ids.keys()):
        ids = per_segment_kept_ids[seg_id]
        per_segment_payloads[seg_id] = {aid: per_actor[aid] for aid in ids if aid in per_actor}

    all_kept_payloads = {aid: per_actor[aid] for aid in all_kept_ids if aid in per_actor}
    return {
    "per_segment_ids": per_segment_kept_ids,
    "all_ids": all_kept_ids,
    "per_segment_payloads": per_segment_payloads,
    "actor_activities": all_kept_payloads
    }

def _actor_type(aid: str) -> str:
    return aid.split("_", 1)[0] if isinstance(aid, str) and "_" in aid else "unknown"

def _valid_len(rec: Dict[str, Any]) -> int:
    v0, v1 = rec.get("valid", (0, -1))
    try:
        v0, v1 = int(v0), int(v1)
    except Exception:
        return 0
    return max(0, v1 - v0 + 1)

def _normalize_source_to_payloads(
    source: Dict[str, Any]
) -> Tuple[Dict[str, Dict[str, Any]], Optional[Dict[str, Dict[str, Any]]]]:
    """
    Accepts either:
      - per_actor (output of per_actor_minimal): {actor_id -> payload}
      - kept bundle (output of kept_actors_from_per_actor_minimal): {
            "all_payloads": {...}, "per_segment_payloads": {...}, ... }
    Returns:
      (all_payloads, per_segment_payloads_or_None)
    """
    if "all_payloads" in source:
        return source["all_payloads"], source.get("per_segment_payloads")
    # assume it's a raw per_actor dict
    return source, None

def print_actor_type_counts(
    source: Dict[str, Any],
    min_steps: Optional[int] = None,
    show_per_segment: bool = True,
) -> Dict[str, Any]:
    """
    Prints overall type counts (and per-segment if available).
    Works with either per_actor_minimal output OR the kept bundle.
    Returns a small summary dict too.
    """
    all_payloads, per_seg_payloads = _normalize_source_to_payloads(source)

    # overall counts (optionally filter by valid length)
    if min_steps is None:
        ids: Iterable[str] = all_payloads.keys()
    else:
        ids = [aid for aid, rec in all_payloads.items() if _valid_len(rec) >= min_steps]

    overall_ctr = Counter(_actor_type(aid) for aid in ids)
    total = sum(overall_ctr.values())

    print("=== Overall kept actors ===" if "all_payloads" in source else "=== Overall actors ===")
    if min_steps is not None:
        print(f"(only counting actors with >= {min_steps} valid frames)")
    for t, n in sorted(overall_ctr.items()):
        print(f"{t:>11}: {n}")
    print(f"{'TOTAL':>11}: {total}")

    per_segment_counts = {}
    if show_per_segment and per_seg_payloads:
        print("\n=== Per-segment type counts ===")
        for seg_id in sorted(per_seg_payloads.keys()):
            payloads = per_seg_payloads[seg_id]
            # normally these are already "kept", so min_steps filtering is not needed;
            # but we keep it for symmetry:
            seg_ids = (
                payloads.keys() if min_steps is None
                else [aid for aid, rec in payloads.items() if _valid_len(rec) >= min_steps]
            )
            ctr = Counter(_actor_type(aid) for aid in seg_ids)
            per_segment_counts[seg_id] = dict(ctr)
            line = ", ".join(f"{k}:{v}" for k, v in sorted(ctr.items()))
            print(f"{seg_id}: {line or '(none)'}")

    return {
        "overall": dict(overall_ctr),
        "total": total,
        "per_segment": per_segment_counts if per_seg_payloads else None,
        "min_steps": min_steps,
        "source_type": "kept_bundle" if "all_payloads" in source else "per_actor",
    }