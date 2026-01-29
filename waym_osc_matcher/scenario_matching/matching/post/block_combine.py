# scenario_matching/matching/post/block_combine.py
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from scenario_matching.matching.results.types import (
    Interval,
    WindowsByT0,
    CallKey,
    BindingKey,
    PerCallSignal,
    BlockSignal,
)


def _roles_key_strict(binding: Mapping[str, str], required_roles: List[str]) -> Optional[BindingKey]:
    """
    Build a BindingKey for exactly the required_roles.
    Returns None if any required role is missing/empty.
    """
    items: List[Tuple[str, str]] = []
    for r in required_roles:
        v = binding.get(r)
        if not v:
            return None
        items.append((str(r), str(v)))
    items.sort(key=lambda kv: kv[0])
    return tuple(items)

def _coalesce_intervals(intervals: List[Interval]) -> List[Interval]:
    if not intervals:
        return []
    ivs = sorted((int(a), int(b)) for a, b in intervals)
    out: List[Interval] = []
    a0, b0 = ivs[0]
    for a, b in ivs[1:]:
        if a <= b0 + 1:
            b0 = max(b0, b)
        else:
            out.append((a0, b0))
            a0, b0 = a, b
    out.append((a0, b0))
    return out

def _intervals_from_mask(m: np.ndarray) -> List[Interval]:
    m = np.asarray(m, dtype=bool)
    if m.size == 0 or not m.any():
        return []
    idx = np.flatnonzero(m)
    starts = [int(idx[0])]
    ends: List[int] = []
    for i in range(1, len(idx)):
        if idx[i] != idx[i - 1] + 1:
            ends.append(int(idx[i - 1]))
            starts.append(int(idx[i]))
    ends.append(int(idx[-1]))
    return list(zip(starts, ends))

def _mask_from_intervals(intervals: List[Interval], T: int) -> np.ndarray:
    m = np.zeros(int(T), dtype=bool)
    for a, b in intervals:
        a = max(0, int(a))
        b = min(int(T) - 1, int(b))
        if a <= b:
            m[a : b + 1] = True
    return m

def _coerce_bucket_to_list(bucket: Any) -> List[PerCallSignal]:
    """
    bucket kann sein:
      - List[PerCallSignal]
      - Dict[BindingKey, PerCallSignal] / Dict[BindingKey, List[PerCallSignal]]
      - Set/Iterable[PerCallSignal]
    """
    if bucket is None:
        return []
    if isinstance(bucket, dict):
        vals = list(bucket.values())
        # falls Dict[BindingKey, List[PerCallSignal]]: flatten
        if vals and isinstance(vals[0], list):
            out: List[PerCallSignal] = []
            for v in vals:
                out.extend(v or [])
            return out
        return vals  # type: ignore[return-value]
    try:
        return list(bucket)
    except Exception:
        return []

def _get_store_list(store: Any, key: CallKey) -> List[PerCallSignal]:
    """
    Works with different ResultsStore implementations.
    - bevorzugt: store.get_all(key) / store.get(key)
    - fallback: store.by_call[key]
    - legacy: store.data/_data[key]
    """
    for attr in ("get_all", "get"):
        fn = getattr(store, attr, None)
        if callable(fn):
            try:
                v = fn(key)
                return _coerce_bucket_to_list(v)
            except TypeError:
                # store.get(...) erwartet evtl. (call_key, binding_key)
                pass
            except Exception:
                pass

    d = getattr(store, "by_call", None)
    if isinstance(d, dict) and key in d:
        return _coerce_bucket_to_list(d[key])

    for attr in ("data", "_data"):
        d2 = getattr(store, attr, None)
        if isinstance(d2, dict) and key in d2:
            return _coerce_bucket_to_list(d2[key])

    return []

def _max_possible_windows(T: int, minF: int, maxF: int) -> int:
    """
    Count all theoretical (t0,t1) with window length in [minF..maxF] within [0..T-1].
    Inclusive indices.
    """
    T = int(T)
    minF = max(1, int(minF))
    maxF = max(minF, int(maxF))
    total = 0
    for t0 in range(T):
        t1_min = t0 + minF - 1
        t1_max = min(T - 1, t0 + maxF - 1)
        if t1_min <= t1_max:
            total += (t1_max - t1_min + 1)
    return int(total)

def _next_true_indices(m: np.ndarray) -> np.ndarray:
    """
    next_true[t] = smallest i>=t with m[i]=True, else -1.
    """
    m = np.asarray(m, dtype=bool)
    T = m.shape[0]
    nxt = np.full(T + 1, -1, dtype=np.int32)
    last = -1
    for t in range(T - 1, -1, -1):
        if m[t]:
            last = t
        nxt[t] = last
    return nxt[:-1]

def combine_parallel_block(
    plan: Any,
    calls: List[dict],
    store: Any,
    T_by_seg: Mapping[str, int],
) -> Dict[Tuple[str, BindingKey], BlockSignal]:
    """
    Combine atomic per-call masks into a block-level signal for a PARALLEL block.

    WICHTIGER FIX:
    --------------
    Deine Calls in einem Block können unterschiedliche Role-Sets haben
    (z.B. ego-drive nutzt nur ego, npc-drive nutzt ego+npc).
    Ein "keys &= keys" auf BindingKey funktioniert dann NICHT, weil die Keys verschieden sind.

    Lösung: wähle einen "Master"-Call (mit den meisten Rollen) und projiziere dessen Binding
    auf die Rollen-Subsets der anderen Calls.
    """
    idxs = list(getattr(plan, "indices", []) or [])
    if not idxs:
        return {}

    overlap = str(getattr(plan, "overlap", "any") or "any").lower()
    collect_block_windows = bool(getattr(plan, "collect_block_windows", False))

    # Build per-call maps: (seg_id, bindingkey_for_this_call_roles) -> signal
    call_roles: List[List[str]] = []
    call_maps: List[Dict[Tuple[str, BindingKey], PerCallSignal]] = []
    call_sigs: List[List[PerCallSignal]] = []

    for ci in idxs:
        ck: CallKey = (str(calls[ci].get("block_label") or ""), int(ci))
        sigs = _get_store_list(store, ck)
        call_sigs.append(sigs)

        # roles used by this call (prefer stored roles_used)
        roles_ci: List[str] = []
        for s in sigs[:1]:
            ru = getattr(s, "roles_used", None)
            if ru:
                roles_ci = [str(x) for x in ru]
        if not roles_ci:
            # fallback: whatever roles appear in the first signal
            if sigs:
                roles_ci = sorted(str(k) for k in (getattr(sigs[0], "roles", {}) or {}).keys())
        if not roles_ci:
            # no signals -> block can't match
            return {}

        call_roles.append(roles_ci)

        m: Dict[Tuple[str, BindingKey], PerCallSignal] = {}
        for s in sigs:
            seg_id = str(getattr(s, "segment_id", ""))
            bk = _roles_key_strict(getattr(s, "roles", {}) or {}, roles_ci)
            if bk is None:
                continue
            m[(seg_id, bk)] = s
        call_maps.append(m)

    # block roles (union)
    block_roles: List[str] = sorted({r for rs in call_roles for r in rs})

    # pick master call = most roles (ties -> first)
    master_j = max(range(len(idxs)), key=lambda j: len(call_roles[j]))
    master_roles = call_roles[master_j]
    master_sigs = call_sigs[master_j]

    out: Dict[Tuple[str, BindingKey], BlockSignal] = {}

    for s_master in master_sigs:
        seg_id = str(getattr(s_master, "segment_id", ""))
        T = int(T_by_seg.get(seg_id) or getattr(s_master, "T", 0) or 0)
        if T <= 0:
            continue

        binding_full = dict(getattr(s_master, "roles", {}) or {})
        # ensure binding_full includes all master roles
        bk_master = _roles_key_strict(binding_full, master_roles)
        if bk_master is None:
            continue

        # Gather corresponding signals from other calls by projection
        sigs_for_block: List[PerCallSignal] = [None] * len(idxs)  # type: ignore[assignment]
        sigs_for_block[master_j] = s_master

        roles_union: Dict[str, str] = dict(binding_full)

        ok = True
        for j in range(len(idxs)):
            if j == master_j:
                continue
            roles_j = call_roles[j]
            bk_j = _roles_key_strict(binding_full, roles_j)
            if bk_j is None:
                ok = False
                break
            sj = call_maps[j].get((seg_id, bk_j))
            if sj is None:
                ok = False
                break
            sigs_for_block[j] = sj
            roles_union.update(dict(getattr(sj, "roles", {}) or {}))


        if not ok:
            continue

        # --- NEW: ensure exactly one signal per call_index (avoid duplicates) ---
        expected_n = len(idxs)
        by_ci: Dict[int, PerCallSignal] = {}
        for s in sigs_for_block:
            ci = getattr(s, "call_index", None)
            if ci is None:
                continue
            if ci not in by_ci:
                by_ci[ci] = s
            else:
                # prefer one that actually has endframes (more informative)
                a = by_ci[ci]
                a_has = bool(getattr(a, "endframes", None))
                s_has = bool(getattr(s, "endframes", None))
                if (not a_has) and s_has:
                    by_ci[ci] = s

        # must have all calls present, otherwise we cannot combine parallel correctly
        if len(by_ci) != expected_n:
            continue

        # rebuild in call_index order
        sigs_for_block = [by_ci[i] for i in sorted(by_ci.keys())]

        if bool(getattr(plan, "debug_block_sigs", False)):
            print(
                "[BC-SIGS]",
                [(s.call_index, len(s.endframes or []), (s.endframes[:5] if s.endframes else None)) for s in sigs_for_block],
                flush=True,
            )

        # Ensure output binding key covers the whole block (ego+npc etc.)
        bk_out = _roles_key_strict(roles_union, block_roles)
        if bk_out is None:
            continue

        # per-call masks (length T)
        per_masks: List[np.ndarray] = []
        for s in sigs_for_block:
            sm = getattr(s, "mask", None)
            if sm is None:
                sm = _mask_from_intervals(getattr(s, "intervals", []) or [], T)
            sm = np.asarray(sm, dtype=bool)
            if sm.shape[0] != T:
                sm = sm[:T] if sm.shape[0] > T else np.pad(sm, (0, T - sm.shape[0]), constant_values=False)
            per_masks.append(sm)

        minF = int(getattr(plan, "duration_min_frames", 1) or 1)
        maxF = getattr(plan, "duration_max_frames", None)
        maxF = int(maxF) if maxF is not None else T
        minF = max(1, minF)
        maxF = max(minF, min(int(maxF), T))

        # optimized scan
        m_all = per_masks[0].copy()
        for mm in per_masks[1:]:
            m_all &= mm

        nxt_all = _next_true_indices(m_all)
        nxt_calls = [_next_true_indices(mm) for mm in per_masks]

        # Prefer signal.endframes (true end-check frames); fall back to mask support if missing.
        per_true: List[np.ndarray] = []
        for s, mm in zip(sigs_for_block, per_masks):
            ef = getattr(s, "endframes", None)
            if ef is not None and len(ef) > 0:
                arr = np.asarray(ef, dtype=np.int32)
            else:
                # fallback: old behaviour if endframes not available
                arr = np.flatnonzero(mm).astype(np.int32)

            # IMPORTANT: keep sorted+unique for searchsorted + intersect1d(assume_unique=True)
            if arr.size > 1:
                arr = np.unique(arr)  # sorted unique
            per_true.append(arr)


        #per_true = [np.flatnonzero(mm).astype(np.int32) for mm in per_masks]
        m_block = np.zeros(T, dtype=bool)
        windows_by_t0: Optional[WindowsByT0] = {} if collect_block_windows else None

        n_windows = 0
        example_window: Optional[Tuple[int, int, int]] = None

        for t0 in range(T):
            t1_min = t0 + minF - 1
            t1_max = min(T - 1, t0 + maxF - 1)
            if t1_min > t1_max:
                continue

            # overlap constraint
            if overlap == "start":
                if not m_all[t0]:
                    continue
                t_all = t0
            else:  # "any"
                t_all = int(nxt_all[t0])
                if t_all < 0 or t_all > t1_max:
                    continue

            # each call at least once
            t_req = t_all
            ok2 = True
            for nxt in nxt_calls:
                t_hit = int(nxt[t0])
                if t_hit < 0 or t_hit > t1_max:
                    ok2 = False
                    break
                if t_hit > t_req:
                    t_req = t_hit
            if not ok2:
                continue
                        
            t1_first = max(int(t1_min), int(t_req))
            if t1_first > t1_max:
                continue

            # --- NEW: intersection of valid t1 across all calls ---
            # For each call, find mask-true indices within [t1_first..t1_max]
            t1_sets: List[np.ndarray] = []
            ok3 = True
            for arr in per_true:
                # slice arr to [t1_first..t1_max] using binary search
                lo = int(np.searchsorted(arr, t1_first, side="left"))
                hi = int(np.searchsorted(arr, t1_max, side="right"))
                sub = arr[lo:hi]
                if sub.size == 0:
                    ok3 = False
                    break
                t1_sets.append(sub)

            if not ok3:
                continue

            # intersect all arrays
            common = t1_sets[0]
            for sub in t1_sets[1:]:
                common = np.intersect1d(common, sub, assume_unique=True)
                if common.size == 0:
                    break
            if common.size == 0:
                continue

            t1_first_common = int(common[0])

            # greedy extent: contiguous run within common (optional)
            t1_greedy_common = t1_first_common
            if common.size > 1:
                # common is sorted
                for k in range(1, common.size):
                    if int(common[k]) == t1_greedy_common + 1:
                        t1_greedy_common += 1
                    else:
                        break

            # count windows = number of common endframes (or contiguous part, depending on what you want)
            n_windows += int(common.size)

            # union-support for this t0 is still [t0..t1_max] (mask semantics)
            m_block[t0 : t1_max + 1] = True

            if windows_by_t0 is not None:
                # only keep common valid windows
                windows_by_t0[int(t0)] = [(t1_first_common, int(t1_greedy_common))]

            if example_window is None:
                # show the first few candidate t1s per call and the intersection
                dbg = {
                    "seg_id": seg_id,
                    "bk_out": bk_out,
                    "t0": int(t0),
                    "t1_min": int(t1_min),
                    "t1_max": int(t1_max),
                    "t_req": int(t_req),
                    "t1_first": int(t1_first),
                    "per_call_first10": [sub[:10].tolist() for sub in t1_sets],
                    "common_first20": common[:20].tolist(),
                    "picked": (int(t1_first_common), int(t1_greedy_common)),
                }
                print("[BC-END]", dbg, flush=True)
                # strict: example_window uses the common end
                example_window = (int(t0), int(t1_greedy_common), int(t1_greedy_common))
                break

        if not m_block.any():
            continue

        ivs = _coalesce_intervals(_intervals_from_mask(m_block))
        n_possible = _max_possible_windows(T, minF, maxF)

        out[(seg_id, bk_out)] = BlockSignal(
            segment_id=seg_id,
            roles=dict(roles_union),
            T=T,
            intervals=ivs,
            mask=m_block,
            windows_by_t0=windows_by_t0 if collect_block_windows else None,
            n_windows=int(n_windows),
            n_possible_windows=int(n_possible),
            example_window=example_window,
        )

    return out

def chain_serial_block(
    plan: Any,
    calls: List[dict],
    store: Any,
    T_by_seg: Mapping[str, int],
    *,
    allow_overlap: bool = True,
) -> Dict[Tuple[str, BindingKey], BlockSignal]:
    """
    Minimal serial combiner (keeps compatibility with engine imports).
    Conservative behaviour: AND their masks (simultaneous).
    """
    idxs = list(getattr(plan, "indices", []) or [])
    if not idxs:
        return {}

    out: Dict[Tuple[str, BindingKey], BlockSignal] = {}

    call_roles: List[List[str]] = []
    call_maps: List[Dict[Tuple[str, BindingKey], PerCallSignal]] = []
    call_sigs: List[List[PerCallSignal]] = []

    for ci in idxs:
        ck: CallKey = (str(calls[ci].get("block_label") or ""), int(ci))
        sigs = _get_store_list(store, ck)
        call_sigs.append(sigs)

        roles_ci: List[str] = []
        for s in sigs[:1]:
            ru = getattr(s, "roles_used", None)
            if ru:
                roles_ci = [str(x) for x in ru]
        if not roles_ci:
            if sigs:
                roles_ci = sorted(str(k) for k in (getattr(sigs[0], "roles", {}) or {}).keys())
        if not roles_ci:
            return {}

        call_roles.append(roles_ci)

        m: Dict[Tuple[str, BindingKey], PerCallSignal] = {}
        for s in sigs:
            seg_id = str(getattr(s, "segment_id", ""))
            bk = _roles_key_strict(getattr(s, "roles", {}) or {}, roles_ci)
            if bk is None:
                continue
            m[(seg_id, bk)] = s
        call_maps.append(m)

    block_roles = sorted({r for rs in call_roles for r in rs})
    master_j = max(range(len(idxs)), key=lambda j: len(call_roles[j]))
    master_roles = call_roles[master_j]
    master_sigs = call_sigs[master_j]

    for s_master in master_sigs:
        seg_id = str(getattr(s_master, "segment_id", ""))
        T = int(T_by_seg.get(seg_id) or getattr(s_master, "T", 0) or 0)
        if T <= 0:
            continue

        binding_full = dict(getattr(s_master, "roles", {}) or {})
        bk_master = _roles_key_strict(binding_full, master_roles)
        if bk_master is None:
            continue

        sigs_for_block: List[PerCallSignal] = [None] * len(idxs)  # type: ignore[assignment]
        sigs_for_block[master_j] = s_master
        roles_union: Dict[str, str] = dict(binding_full)

        ok = True
        for j in range(len(idxs)):
            if j == master_j:
                continue
            roles_j = call_roles[j]
            bk_j = _roles_key_strict(binding_full, roles_j)
            if bk_j is None:
                ok = False
                break
            sj = call_maps[j].get((seg_id, bk_j))
            if sj is None:
                ok = False
                break
            sigs_for_block[j] = sj
            roles_union.update(dict(getattr(sj, "roles", {}) or {}))
        
        if not ok:
            continue

        bk_out = _roles_key_strict(roles_union, block_roles)
        if bk_out is None:
            continue

        masks: List[np.ndarray] = []
        for s in sigs_for_block:
            sm = getattr(s, "mask", None)
            if sm is None:
                sm = _mask_from_intervals(getattr(s, "intervals", []) or [], T)
            sm = np.asarray(sm, dtype=bool)
            if sm.shape[0] != T:
                sm = sm[:T] if sm.shape[0] > T else np.pad(sm, (0, T - sm.shape[0]), constant_values=False)
            masks.append(sm)

        m_ser = masks[0].copy()
        for mm in masks[1:]:
            m_ser &= mm

        if not m_ser.any():
            continue

        ivs = _coalesce_intervals(_intervals_from_mask(m_ser))
        out[(seg_id, bk_out)] = BlockSignal(
            segment_id=seg_id,
            roles=dict(roles_union),
            T=T,
            intervals=ivs,
            mask=m_ser,
            windows_by_t0=None,
            n_windows=None,
            n_possible_windows=None,
            example_window=None,
        )

    return out