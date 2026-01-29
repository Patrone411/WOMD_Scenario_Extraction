# scenario_matching/matching/results/collect.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
import math
import time

import numpy as np

from .types import CallKey, PerCallSignal
from .store import ResultsStore
from .interval_ops import merge_windows_to_intervals, intervals_to_mask

from scenario_matching.matching.role_planning import (
    make_type_to_candidates,
    roles_used_by_call,
    role_domains_from_segment,
    prefilter_domains,
    build_overlap_matrix,
    enumerate_bindings,
)

from scenario_matching.matching.match_single_call import match_for_binding


def _dur_to_minF(call: Dict[str, Any], fps: int) -> int:
    dur = call.get("block_duration") or call.get("duration")
    if not dur:
        return 1
    if dur.get("unit", "second") != "second":
        return 1
    if "value" in dur:
        lo = float(dur["value"])
    else:
        r = dur.get("range") or [None, None]
        lo = float(r[0]) if r and r[0] is not None else None
    if lo is None:
        return 1
    return max(1, int(math.ceil(lo * float(fps))))


def _flag(obj: Any, name: str, default: bool = False) -> bool:
    return bool(getattr(obj, name, default))


def _num(obj: Any, name: str, default: int) -> int:
    v = getattr(obj, name, default)
    try:
        return int(v)
    except Exception:
        return default


def _fnum(obj: Any, name: str, default: Optional[float]) -> Optional[float]:
    v = getattr(obj, name, default)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return default


def _str(obj: Any, name: str, default: Optional[str]) -> Optional[str]:
    v = getattr(obj, name, default)
    if v is None:
        return None
    return str(v)


def collect_results(
    h,
    calls: List[Dict[str, Any]],
    *,
    max_results_per_binding: int = 10_000,
    collect_call_windows: bool = False,
    collect_modifier_stats: bool = False,
    max_windows_per_binding: int = 200_000,
) -> ResultsStore:
    """
    Collector with optional heavy debug:
      - only_call_index / only_segment_id (to inspect a pathological case)
      - slow_eval_threshold_s: only print per-binding perf when a binding exceeds threshold
    """
    setattr(h.cfg, "debug_pcs", True)

    store = ResultsStore()

    debug_domains = _flag(h.cfg, "debug_domains", False)
    debug_bindings = _flag(h.cfg, "debug_bindings", False)

    # Perf debug: print only slow bindings (prevents spam)
    slow_eval_threshold_s = _fnum(h.cfg, "slow_eval_threshold_s", None)  # e.g. 0.25
    slow_eval_max_print = _num(h.cfg, "slow_eval_max_print", 20)

    # Restrict loop for investigation (optional)
    only_call_index = getattr(h.cfg, "only_call_index", None)
    only_segment_id = _str(h.cfg, "only_segment_id", None)

    max_bindings_per_call_segment = _num(h.cfg, "max_bindings_per_call_segment", 0)

    osc_id_abs_max = _fnum(h.cfg, "osc_id_abs_max", None)
    require_consecutive_presence = _flag(h.cfg, "require_consecutive_presence", True)

    feats_by_seg = (h.feats_by_seg or {})

    # IMPORTANT: segment loop is OUTER so that block_local resets per segment
    for seg_id, feats in feats_by_seg.items():
        if only_segment_id and seg_id != only_segment_id:
            continue

        # local call index per block_label (per segment!)
        block_local: Dict[str, int] = {}

        for ci, call in enumerate(calls):
            if only_call_index is not None and int(only_call_index) != ci:
                continue

            block_label = call.get("block_label") or "<none>"

            # local call index inside this block_label, per segment
            bj = block_local.get(block_label, 0)
            block_local[block_label] = bj + 1

            call_key: CallKey = (block_label, bj)

            minF = _dur_to_minF(call, h.cfg.fps) or 10
            roles = sorted(roles_used_by_call(call))
            ego_role = call.get("actor")
            others = [r for r in roles if r != ego_role]
            require_pairs = [(ego_role, r) for r in others] if others else []

            resolver = make_type_to_candidates(feats)
            domains = role_domains_from_segment(
                h.scn_constraints, feats, roles=roles, type_to_candidates=resolver
            )

            domains = prefilter_domains(
                feats,
                domains,
                min_present_frames=minF,
                osc_id_abs_max=3,
            )

            if debug_domains:
                sizes = {r: len(domains.get(r, [])) for r in roles}
                print(f"[DOM] call={ci} bj={bj} seg={seg_id} block={block_label} sizes={sizes}", flush=True)

            overlap = None
            if require_pairs:
                actors = sorted({a for A in domains.values() for a in A})
                if len(actors) >= 2:
                    overlap = build_overlap_matrix(
                        feats,
                        actors,
                        min_overlap_frames=minF,
                        osc_id_abs_max=getattr(h.cfg, "osc_id_abs_max", None),
                        require_consecutive=require_consecutive_presence,
                    )
                    if _flag(h.cfg, "debug_overlap_stats", True):
                        tot = len(actors) * (len(actors) - 1)
                        okc = sum(1 for (a, b), v in overlap.items() if a != b and v)
                        print(f"[OVL] call={ci} bj={bj} seg={seg_id} actors={len(actors)} ok_pairs={okc}/{tot}", flush=True)

            t_bind0 = time.perf_counter()
            n_bind = 0
            slow_printed = 0

            for binding in enumerate_bindings(
                domains,
                distinct=True,
                overlap_ok=overlap,
                require_overlap_pairs=require_pairs,
                limit=max_bindings_per_call_segment or None,
            ):
                n_bind += 1

                cfg = h.cfg.to_query_cfg()
                cfg["collect_call_windows"] = bool(cfg.get("collect_call_windows", False) or collect_call_windows)
                cfg["collect_modifier_stats"] = bool(cfg.get("collect_modifier_stats", False) or collect_modifier_stats)
                cfg["max_windows_per_binding"] = int(cfg.get("max_windows_per_binding", max_windows_per_binding) or max_windows_per_binding)

                # we want strict correct end windows when SED is on
                cfg["first_window_only"] = True

                # make sure match_block can return dict (hits + endframes); safe even if you don't store heavy details
                cfg["return_details"] = True
  
                out = match_for_binding(
                    feats,
                    call,
                    binding,
                    fps=h.cfg.fps,
                    max_results=max_results_per_binding,
                    cfg=cfg,
                )

                if not out:
                    continue

                if isinstance(out, dict):
                    hits = out.get("hits") or []
                    windows_by_t0 = out.get("windows_by_t0") if cfg["collect_call_windows"] else None
                    mod_stats = out.get("mod_stats") if cfg["collect_modifier_stats"] else None
                    endframes = out.get("endframes")
                else:
                    hits = out
                    windows_by_t0 = None
                    mod_stats = None
                    endframes = None

                if not hits:
                    continue

                wins = [(hi["t_start"], hi["t_end"]) for hi in hits]
                intervals = merge_windows_to_intervals(wins)
                T = int(getattr(feats, "T", 0) or 0)
                mask = intervals_to_mask(intervals, T) if T else None

                store.add(
                    call_key,
                    PerCallSignal(
                        segment_id=seg_id,
                        roles=dict(binding),
                        T=T,
                        intervals=intervals,
                        mask=mask,
                        call_index=bj,                 # local index inside block_label for this segment
                        roles_used=tuple(roles),
                        windows_by_t0=windows_by_t0,
                        endframes=endframes,
                        mod_stats=mod_stats,
                    ),
                )

                # perf debug (optional)
                if slow_eval_threshold_s is not None and slow_printed < slow_eval_max_print:
                    dt = time.perf_counter() - t_bind0
                    if dt >= slow_eval_threshold_s:
                        slow_printed += 1
                        print(
                            f"[SLOW] seg={seg_id} call={ci} bj={bj} binding={dict(binding)} dt={dt:.3f}s",
                            flush=True,
                        )

            t_bind1 = time.perf_counter()
            if debug_bindings:
                dt = t_bind1 - t_bind0
                rate = (n_bind / dt) if dt > 0 else 0.0
                print(f"[BIND] seg={seg_id} call={ci} bj={bj} bindings={n_bind} dt={dt:.2f}s rate={rate:.1f}/s", flush=True)

    return store
