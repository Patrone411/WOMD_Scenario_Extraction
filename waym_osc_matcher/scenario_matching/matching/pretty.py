#/matching/pretty.py
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional
from scenario_matching.matching.spec import build_block_query

from scenario_matching.matching.results.ops import intervals_to_mask
# ---------------------------
# small helpers (unchanged)
# ---------------------------
def preview_mask(mask, width=100):
    if mask.size <= width:
        return "".join("#" if b else "." for b in mask)
    idx = np.linspace(0, mask.size - 1, num=width).astype(int)
    return "".join("#" if mask[i] else "." for i in idx)

def fmt_intervals(ivs):
    return ", ".join(f"[{iv.t0},{iv.t1}]" for iv in ivs)

def roles_str(roles: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(roles.items()))

def _check_labels(checks) -> List[str]:
    return [getattr(c, "_label", f"check#{i}") for i, c in enumerate(checks)]

# ---------------------------
# blocks (step 3) printer
# ---------------------------
def print_block_results(plans, block_hits, show_blocks=10, show_bindings=5, width=100):
    printed = 0
    for label, plan in plans.items():
        if printed >= show_blocks:
            break
        sigs_map = block_hits.get(label, {})
        sigs = list(sigs_map.values())
        if not sigs:
            continue
        print(f"\n[BLOCK] '{label}' ({plan.type}) → {len(sigs)} bindings with non-empty signals")
        for s in sigs[:show_bindings]:
            mask = s.mask if s.mask is not None else intervals_to_mask(s.intervals, s.T)
            on = int(mask.sum())
            print(f"  seg={s.segment_id}  roles={roles_str(s.roles)}")
            print(f"    intervals: {fmt_intervals(s.intervals)}  | frames_on={on}/{s.T}")
            print(f"    timeline:  {preview_mask(mask, width=width)}")
        printed += 1

# ---------------------------
# calls (step 2) printers
# ---------------------------
def print_calls_in_block(
    plans,
    calls: List[Dict],
    store,                              # ResultsStore
    block_label: str,
    *,
    show_calls: int = 10,
    show_bindings: int = 3,
    width: int = 100,
    show_checks: bool = True,
    fps: Optional[int] = None,
    cfg: Optional[Dict] = None,
):
    """
    Pretty-print calls inside a single block:
      - shows compiled check labels (so anchors like at=start/end are visible)
      - shows first N bindings per call with intervals + timeline
    """
    plan = plans.get(block_label)
    if not plan:
        print(f"[print_calls_in_block] No block named '{block_label}'")
        return

    # Lazy import to avoid circulars at module import time

    for ci in plan.indices[:show_calls]:
        call = calls[ci]
        ck = (call.get("block_label", block_label), ci)
        signals = store.by_call.get(ck, [])

        if show_checks:
            Q, _ = build_block_query(call, fps=(fps or 10), cfg=(cfg or {}))
            labels = _check_labels(Q.checks)
        else:
            labels = []

        print(f"\nCall {ci} in block '{call.get('block_label', block_label)}' → {len(signals)} bindings")
        print(f"  actor={call.get('actor')} action={call.get('action')}")
        print(f"  modifiers={[m.get('name') for m in (call.get('modifiers') or [])]}")
        if labels:
            print("  checks:", labels)

        for s in signals[:show_bindings]:
            mask = s.mask if s.mask is not None else intervals_to_mask(s.intervals, s.T)
            on = int(mask.sum())
            print(f"    seg={s.segment_id} roles={s.roles}")
            print(f"      intervals: {fmt_intervals(s.intervals)}  | frames_on={on}/{s.T}")
            print(f"      timeline:  {preview_mask(mask, width=width)}")

def print_calls_all_blocks(
    plans,
    calls: List[Dict],
    store,
    *,
    show_calls_per_block: int = 10,
    show_bindings: int = 3,
    width: int = 100,
    show_checks: bool = True,
    fps: Optional[int] = None,
    cfg: Optional[Dict] = None,
):
    """
    Convenience: iterate all blocks in plan order and print calls per block.
    """
    for label in plans.keys():
        print_calls_in_block(
            plans, calls, store, label,
            show_calls=show_calls_per_block,
            show_bindings=show_bindings,
            width=width,
            show_checks=show_checks,
            fps=fps, cfg=cfg,
        )

def print_calls_flat(
    calls: List[Dict],
    store,
    *,
    show_calls: int = 10,
    show_bindings: int = 3,
    width: int = 100,
    show_checks: bool = True,
    fps: Optional[int] = None,
    cfg: Optional[Dict] = None,
):
    """
    Old behavior: iterate calls in their raw order (ignores block structure).
    """
    for i, call in enumerate(calls[:show_calls]):
        Q, _ = build_block_query(call, fps=(fps or 10), cfg=(cfg or {})) if show_checks else (None, None)
        labels = _check_labels(Q.checks) if (show_checks and Q) else []

        ck = (call.get("block_label", "<none>"), i)
        signals = store.by_call.get(ck, [])

        print(f"\nCall {i} in block '{call.get('block_label','<none>')}' → {len(signals)} bindings")
        print(f"  actor={call.get('actor')} action={call.get('action')}")
        print(f"  modifiers={[m.get('name') for m in (call.get('modifiers') or [])]}")
        if labels:
            print("  checks:", labels)

        for s in signals[:show_bindings]:
            mask = s.mask if s.mask is not None else intervals_to_mask(s.intervals, s.T)
            on = int(mask.sum())
            print(f"    seg={s.segment_id} roles={s.roles}")
            print(f"      intervals: {fmt_intervals(s.intervals)}  | frames_on={on}/{s.T}")
            print(f"      timeline:  {preview_mask(mask, width=width)}")
