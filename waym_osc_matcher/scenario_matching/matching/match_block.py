# matching/match_block.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import time  # NEW
import numpy as np

from .spec import BlockQuery


def _slice(a, t0, t1):
    """Inclusive slice [t0..t1]."""
    if a is None:
        return None
    return a[t0 : t1 + 1]


def _mask_to_intervals(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Convert boolean mask to inclusive intervals (t0,t1)."""
    T = int(mask.shape[0])
    out: List[Tuple[int, int]] = []
    t = 0
    while t < T:
        if not mask[t]:
            t += 1
            continue
        t0 = t
        while t < T and mask[t]:
            t += 1
        out.append((t0, t - 1))
    return out


def _coalesce_intervals(iv: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not iv:
        return []
    iv = sorted(iv)
    out = [iv[0]]
    for a, b in iv[1:]:
        p0, p1 = out[-1]
        if a <= p1 + 1:
            out[-1] = (p0, max(p1, b))
        else:
            out.append((a, b))
    return out


def _speed_series_mps(feats, aid: str):
    lv = getattr(feats, "long_v", {}) or {}
    if aid in lv and lv[aid] is not None:
        return lv[aid]
    sp = getattr(feats, "speed", {}) or {}
    return sp.get(aid)


def _candidate_pairs(
    feats,
    Q: BlockQuery,
    pairs: Optional[List[Tuple[str, Optional[str]]]] = None,
) -> List[Tuple[str, Optional[str]]]:
    ego = Q.ego
    actors = set(getattr(feats, "actors", []) or [])
    if pairs is not None:
        out = []
        for e, n in pairs:
            if e != ego:
                continue
            if n is not None and n not in actors:
                continue
            out.append((e, n))
        if out:
            return out

    if getattr(Q, "npc_candidates", None):
        return [(ego, n) for n in Q.npc_candidates if (not actors) or (n in actors)]

    return [(ego, None)]


def _candidate_ids(
    feats,
    Q: BlockQuery,
    ids: Optional[List[str]] = None,
) -> List[str]:
    actors = list(getattr(feats, "actors", []) or [])
    if ids:
        return [a for a in ids if (not actors) or (a in actors)]

    ego = getattr(Q, "ego", None)
    if ego and ((not actors) or (ego in actors)):
        return [ego]

    return actors


def _determine_arity(Q: BlockQuery, ids, pairs) -> int:
    arity = getattr(Q, "arity", None)
    if arity in (1, 2):
        return int(arity)
    if ids is not None:
        return 1
    if pairs is not None:
        has_real_npc = any(n is not None for _, n in pairs)
        return 2 if has_real_npc else 1
    return 2 if getattr(Q, "npc_candidates", None) else 1


def _has_sed(Q: BlockQuery) -> bool:
    if not hasattr(Q, "start_checks"):
        return False
    return bool(
        getattr(Q, "start_checks", None)
        or getattr(Q, "end_checks", None)
        or getattr(Q, "during_frame_checks", None)
    )


def _frame_mask(
    feats,
    ego_id: str,
    npc_id: Optional[str],
    checks,
    T: int,
    cfg: Dict[str, Any],
    *,
    debug_checks: bool = False,
) -> np.ndarray:
    if not checks:
        return np.ones(T, dtype=bool)
    m = np.ones(T, dtype=bool)
    for i, check in enumerate(checks):
        if not np.any(m):
            break
        label = getattr(check, "_label", f"check#{i}")
        for t in range(T):
            if not m[t]:
                continue
            try:
                ok = bool(check(feats, ego_id, npc_id, t, t, cfg))
            except Exception:
                ok = False
            if debug_checks:
                print(f"   [frame] {label} t={t}: {ok}")
            if not ok:
                m[t] = False
    return m


def match_block(
    feats,
    Q: BlockQuery,
    fps: int = 10,
    *,
    ids: Optional[List[str]] = None,
    pairs: Optional[List[Tuple[str, Optional[str]]]] = None,
    max_results: int = 5000,
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Sliding-window driver over ONE segment.

    Returns:
      - legacy: list[{"ego","npc","t_start","t_end"}]
      - if cfg["return_details"]=True: dict with
          {"hits": [...], "windows_by_t0": {...}|None, "mod_stats": {...}|None, "per_binding": [...]|None}
    """
    results: List[Dict[str, Any]] = []

    T = int(getattr(feats, "T", 0) or 0)
    if T <= 0:
        return {"hits": [], "windows_by_t0": None, "mod_stats": None, "per_binding": None} if (getattr(Q, "cfg", {}) or {}).get("return_details") else []

    # fill speed if long_v exists
    if getattr(feats, "speed", None) is None and getattr(feats, "long_v", None) is not None:
        try:
            feats.speed = {aid: np.asarray(v, dtype=float) for aid, v in feats.long_v.items()}
            setattr(feats, "speed_unit", "meters_per_second")
        except Exception:
            pass

    cfg = getattr(Q, "cfg", {}) or {}
    # NOTE: correctness of SED end constraints should not depend on return_details.
    # We still keep return_details as a payload toggle, but when use_sed=True we
    # force details on so downstream (combine) can use true endframes.
    return_details_raw = bool(cfg.get("return_details", False))
    collect_call_windows = bool(cfg.get("collect_call_windows", False))
    collect_mod_stats = bool(cfg.get("collect_modifier_stats", False))
    # If you only need *one* valid window per binding, enable this to stop the scan
    # at the first feasible (t0,t1). When SED is on, we default this to True.
    first_window_only_raw = bool(cfg.get("first_window_only", False))
    max_windows_per_binding = int(cfg.get("max_windows_per_binding", 200000))

    debug = bool(cfg.get("debug_match_block", False))
    debug_checks = bool(cfg.get("debug_checks", False))

    # --- NEW: SED scan instrumentation ---
    debug_window_scan = bool(cfg.get("debug_window_scan", False))
    sed_t0_considered = 0
    sed_t1_tried = 0
    sed_wcheck_calls = 0
    sed_dt = 0.0
    # -----------------------------------

    duration_scope = str(cfg.get("duration_scope", "block")).lower()
    allow_shorter = bool(cfg.get("allow_shorter_end", False))
    coalesce_hits = bool(cfg.get("coalesce_hits", duration_scope == "action"))

    D_default = max(1, int(getattr(Q, "duration_frames", 1)))

    min_len = cfg.get("duration_min_frames")
    max_len = cfg.get("duration_max_frames")

    if min_len is None:
        min_len = 1 if (duration_scope == "block" and allow_shorter) else D_default
    min_len = max(1, int(min_len))

    if max_len is None:
        max_len = D_default
    max_len = max(min_len, int(max_len))

    last_t0 = max(0, T - min_len)

    def _t1_bounds(t0: int) -> Tuple[int, int]:
        t1_min = min(T - 1, t0 + min_len - 1)
        if duration_scope == "action":
            t1_max = T - 1
        else:
            t1_max = min(T - 1, t0 + max_len - 1)
        return t1_min, t1_max

    # legacy window checks
    def _run_checks_legacy(ego_id, npc_id, t0, t1) -> bool:
        if debug:
            pres_map = getattr(feats, "present", {}) or {}
            pres = pres_map.get(ego_id, None)
            pwin = _slice(pres, t0, t1)
            pres_cov = None
            if pwin is not None and len(pwin):
                pres_cov = float(np.sum(np.asarray(pwin, dtype=float) > 0.5)) / float(len(pwin))
            sp_series = _speed_series_mps(feats, ego_id)
            sp_min = sp_max = None
            if sp_series is not None:
                w = _slice(sp_series, t0, t1)
                if w is not None and len(w):
                    sp_min = float(np.min(w))
                    sp_max = float(np.max(w))
            if npc_id is None:
                print(f"[win] ego={ego_id} t=[{t0},{t1}] pres_cov={pres_cov} speed_min={sp_min} speed_max={sp_max}")
            else:
                pres_n = (getattr(feats, "present", {}) or {}).get(npc_id, None)
                pwin_n = _slice(pres_n, t0, t1)
                cov_n = None
                if pwin_n is not None and len(pwin_n):
                    cov_n = float(np.sum(np.asarray(pwin_n, dtype=float) > 0.5)) / float(len(pwin_n))
                print(f"[win] ego={ego_id}, npc={npc_id} t=[{t0},{t1}] pres_cov_e={pres_cov} pres_cov_n={cov_n} "
                      f"ego_speed_min={sp_min} ego_speed_max={sp_max}")

        for i, check in enumerate(Q.checks):
            try:
                ok = bool(check(feats, ego_id, npc_id, t0, t1, cfg))
            except Exception as ex:
                if debug:
                    print(f"[match_block] check exception @({t0},{t1}) ego={ego_id} npc={npc_id}: {ex}")
                ok = False
            if debug_checks:
                label = getattr(check, "_label", f"check#{i}")
                print(f"   -> {label}: {ok}")
            if not ok:
                return False
        return True

    def _run_window_checks(ego_id, npc_id, t0, t1) -> bool:
        wchecks = getattr(Q, "window_checks", None)
        if wchecks is None:
            wchecks = getattr(Q, "checks", None) or []
        if not wchecks:
            return True
        for i, check in enumerate(wchecks):
            try:
                ok = bool(check(feats, ego_id, npc_id, t0, t1, cfg))
            except Exception:
                ok = False
            if debug_checks:
                label = getattr(check, "_label", f"wcheck#{i}")
                print(f"   -> {label}: {ok}")
            if not ok:
                return False
        return True

    def _run_end_checks_strict(ego_id, npc_id, t: int) -> bool:
        """Paranoid re-check of end_checks at exact frame t (t0=t1=t)."""
        echecks = getattr(Q, "end_checks", None) or []
        if not echecks:
            return True
        for chk in echecks:
            try:
                if not bool(chk(feats, ego_id, npc_id, t, t, cfg)):
                    return False
            except Exception:
                return False
        return True


    # S/E/D
    use_sed = bool(cfg.get("use_sed", True)) and _has_sed(Q)
    # Derive effective behavior flags
    return_details = bool(return_details_raw or use_sed)
    first_window_only = bool(first_window_only_raw or use_sed)
    anchor_slop = int(cfg.get("anchor_slop_frames", 0) or 0)

    def _dilate_bool(mask: np.ndarray, k: int) -> np.ndarray:
        if k <= 0:
            return mask
        kernel = np.ones((2 * k + 1,), dtype=np.int32)
        return np.convolve(mask.astype(np.int32), kernel, mode="same") > 0

    sed_cache: Dict[Tuple[str, Optional[str]], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def _get_sed(ego_id: str, npc_id: Optional[str]):
        key = (ego_id, npc_id)
        if key in sed_cache:
            return sed_cache[key]

        S = _frame_mask(feats, ego_id, npc_id, getattr(Q, "start_checks", []), T, cfg, debug_checks=debug_checks)
        E = _frame_mask(feats, ego_id, npc_id, getattr(Q, "end_checks", []), T, cfg, debug_checks=debug_checks)
        Dm = _frame_mask(feats, ego_id, npc_id, getattr(Q, "during_frame_checks", []), T, cfg, debug_checks=debug_checks)

        if anchor_slop > 0:
            S = _dilate_bool(S, anchor_slop)
            E = _dilate_bool(E, anchor_slop)

        pref_bad_D = np.concatenate([[0], np.cumsum((~Dm).astype(np.int32))])
        sed_cache[key] = (S, E, Dm, pref_bad_D)
        return sed_cache[key]

    def _t1_ranges_for_t0(
        ego_id: str,
        npc_id: Optional[str],
        t0: int,
        t1_min: int,
        t1_max: int,
        E: np.ndarray,
        pref_bad_D: np.ndarray,
    ) -> List[Tuple[int, int]]:
        nonlocal sed_t1_tried, sed_wcheck_calls, sed_dt  # NEW

        tscan0 = time.perf_counter() if debug_window_scan else 0.0  # NEW

        # Iterate only over candidate endframes (E==True) to avoid scanning the full range.
        end_ts = np.flatnonzero(E[t1_min : t1_max + 1])
        if end_ts.size == 0:
            if debug_window_scan:
                sed_dt += (time.perf_counter() - tscan0)  # NEW
            return []
        end_ts = (end_ts + t1_min).astype(int)

        ranges: List[Tuple[int, int]] = []
        run_lo: Optional[int] = None
        prev_ok_t1: Optional[int] = None

        # Count tried t1 candidates for instrumentation
        if debug_window_scan:
            sed_t1_tried += int(end_ts.size)  # NEW

        for t1 in end_ts:
            # Paranoid strict end re-check (protects against any mask/cfg drift)
            if not _run_end_checks_strict(ego_id, npc_id, int(t1)):
                ok = False
            else:
                # during-ok check is cheap; only then run window checks
                during_ok = (pref_bad_D[int(t1) + 1] - pref_bad_D[t0]) == 0
                if not during_ok:
                    ok = False
                else:
                    if debug_window_scan:
                        sed_wcheck_calls += 1  # NEW
                    ok = _run_window_checks(ego_id, npc_id, t0, int(t1))

            # Fast path: if we only need a single representative valid window for
            # this (t0, binding), stop scanning the remaining candidates.
            if ok and first_window_only:
                if debug_window_scan:
                    sed_dt += (time.perf_counter() - tscan0)  # NEW
                return [(int(t1), int(t1))]

            if ok:
                if run_lo is None:
                    run_lo = int(t1)
                    prev_ok_t1 = int(t1)
                else:
                    # close gap if endframes are not contiguous
                    if prev_ok_t1 is not None and int(t1) != int(prev_ok_t1) + 1:
                        ranges.append((int(run_lo), int(prev_ok_t1)))
                        run_lo = int(t1)
                    prev_ok_t1 = int(t1)
            else:
                if run_lo is not None and prev_ok_t1 is not None:
                    ranges.append((int(run_lo), int(prev_ok_t1)))
                run_lo = None
                prev_ok_t1 = None

        if run_lo is not None and prev_ok_t1 is not None:
            ranges.append((int(run_lo), int(prev_ok_t1)))

        if debug_window_scan:
            sed_dt += (time.perf_counter() - tscan0)  # NEW

        return ranges

    def _compute_mod_stats(ego_id: str, npc_id: Optional[str]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        def add(kind: str, checks):
            for i, chk in enumerate(checks or []):
                label = getattr(chk, "_label", f"{kind}#{i}")
                m = np.zeros(T, dtype=bool)
                for t in range(T):
                    try:
                        m[t] = bool(chk(feats, ego_id, npc_id, t, t, cfg))
                    except Exception:
                        m[t] = False
                runs = _coalesce_intervals(_mask_to_intervals(m))
                out[label] = {
                    "kind": kind,
                    "true_frames": int(m.sum()),
                    "true_frac": float(m.sum()) / float(T),
                    "runs": runs,
                }

        add("start", getattr(Q, "start_checks", []))
        add("end", getattr(Q, "end_checks", []))
        add("during", getattr(Q, "during_frame_checks", []))
        return out

    per_binding_details: List[Dict[str, Any]] = []

    arity = _determine_arity(Q, ids, pairs)
    seg_actors = list(getattr(feats, "actors", []) or [])

    # ---------- UNARY ----------
    if arity == 1:
        cand_ids = _candidate_ids(feats, Q, ids)
        t0 = 0
        while t0 <= last_t0:
            progressed = False
            for ego_id in cand_ids:
                if seg_actors and ego_id not in seg_actors:
                    continue

                windows_by_t0 = {} if (return_details and collect_call_windows and use_sed) else None
                mod_stats = _compute_mod_stats(ego_id, None) if (return_details and collect_mod_stats) else None
                endframes = None
                if use_sed:
                    S, E, Dm, pref_bad_D = _get_sed(ego_id, None)
                    endframes = np.flatnonzero(E).astype(int).tolist() if return_details else None
                    if not S[t0]:
                        continue
                    t1_min, t1_max = _t1_bounds(t0)
                    if debug_window_scan:
                        sed_t0_considered += 1  # NEW
                    ranges = _t1_ranges_for_t0(ego_id, None, t0, t1_min, t1_max, E, pref_bad_D)
                    if not ranges:
                        continue
                    if windows_by_t0 is not None:
                        windows_by_t0[t0] = ranges
                        nwin = sum((hi - lo + 1) for rr in windows_by_t0.values() for lo, hi in rr)
                        if nwin > max_windows_per_binding:
                            cfg["collect_call_windows"] = False
                    # representative: take the first valid end (smallest t1)
                    t1 = ranges[0][0]
                else:
                    t1_min, t1_max = _t1_bounds(t0)
                    if t1_min > t1_max:
                        continue
                    if not _run_checks_legacy(ego_id, None, t0, t1_min):
                        continue
                    t1 = t1_min
                    while t1 < t1_max and _run_checks_legacy(ego_id, None, t0, t1 + 1):
                        t1 += 1

                results.append({"ego": ego_id, "npc": None, "t_start": t0, "t_end": t1})

                if return_details:
                    per_binding_details.append({"ego": ego_id, "npc": None, "windows_by_t0": windows_by_t0, "mod_stats": mod_stats, "endframes": endframes})

                if len(results) >= max_results:
                    break

                t0 = t1 + 1 if coalesce_hits else (t0 + 1)
                progressed = True
                break

            if len(results) >= max_results:
                break
            if not progressed:
                t0 += 1

        # NEW: single summary line per match_block call (i.e., per binding in your pipeline)
        if debug_window_scan and use_sed:
            ego_hint = ids[0] if (ids and len(ids) == 1) else getattr(Q, "ego", None)
            print(
                f"[SED-SCAN] arity=1 ego={ego_hint} npc=None "
                f"t0_considered={sed_t0_considered} t1_tried={sed_t1_tried} "
                f"wchecks={sed_wcheck_calls} dt={sed_dt:.3f}s hits={len(results)}",
                flush=True,
            )

        if return_details:
            top = per_binding_details[-1] if len(per_binding_details) == 1 else None
            return {
                "hits": results,
                "windows_by_t0": None if top is None else top.get("windows_by_t0"),
                "mod_stats": None if top is None else top.get("mod_stats"),
                "endframes": None if top is None else top.get("endframes"),
                "per_binding": per_binding_details if len(per_binding_details) > 1 else None,
            }
        return results

    # ---------- BINARY ----------
    cand_pairs = _candidate_pairs(feats, Q, pairs)
    t0 = 0
    while t0 <= last_t0:
        progressed = False
        for (ego_id, npc_id) in cand_pairs:
            windows_by_t0 = {} if (return_details and collect_call_windows and use_sed) else None
            mod_stats = _compute_mod_stats(ego_id, npc_id) if (return_details and collect_mod_stats) else None
            endframes = None

            if use_sed:
                S, E, Dm, pref_bad_D = _get_sed(ego_id, npc_id)
                endframes = np.flatnonzero(E).astype(int).tolist() if return_details else None
                if not S[t0]:
                    continue
                t1_min, t1_max = _t1_bounds(t0)
                if debug_window_scan:
                    sed_t0_considered += 1  # NEW
                ranges = _t1_ranges_for_t0(ego_id, npc_id, t0, t1_min, t1_max, E, pref_bad_D)
                if not ranges:
                    continue
                if windows_by_t0 is not None:
                    windows_by_t0[t0] = ranges
                # representative: take the first valid end (smallest t1)
                t1 = ranges[0][0]
            else:
                t1_min, t1_max = _t1_bounds(t0)
                if t1_min > t1_max:
                    continue
                if not _run_checks_legacy(ego_id, npc_id, t0, t1_min):
                    continue
                t1 = t1_min
                while t1 < t1_max and _run_checks_legacy(ego_id, npc_id, t0, t1 + 1):
                    t1 += 1

            results.append({"ego": ego_id, "npc": npc_id, "t_start": t0, "t_end": t1})

            if return_details:
                per_binding_details.append({"ego": ego_id, "npc": npc_id, "windows_by_t0": windows_by_t0, "mod_stats": mod_stats, "endframes": endframes})

            if len(results) >= max_results:
                break

            t0 = t1 + 1 if coalesce_hits else (t0 + 1)
            progressed = True
            break

        if len(results) >= max_results:
            break
        if not progressed:
            t0 += 1

    if debug_window_scan and use_sed:
        pair_hint = pairs[0] if (pairs and len(pairs) == 1) else None
        print(
            f"[SED-SCAN] arity=2 pair={pair_hint} "
            f"t0_considered={sed_t0_considered} t1_tried={sed_t1_tried} "
            f"wchecks={sed_wcheck_calls} dt={sed_dt:.3f}s hits={len(results)}",
            flush=True,
        )

    if return_details:
        top = per_binding_details[-1] if len(per_binding_details) == 1 else None
        return {
            "hits": results,
            "windows_by_t0": None if top is None else top.get("windows_by_t0"),
            "mod_stats": None if top is None else top.get("mod_stats"),
            "per_binding": per_binding_details if len(per_binding_details) > 1 else None,
        }
    return results