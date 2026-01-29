
# scenario_matching/analysis_stats/stats_collector.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

# Reuse role planning helpers (uncond denoms + optional baseline sampling)
from scenario_matching.matching.role_planning import (
    make_type_to_candidates,
    roles_used_by_call,
    role_domains_from_segment,
    prefilter_domains,
    build_overlap_matrix,
    enumerate_bindings,
)

# Parameter extractor (hit vs baseline)
from scenario_matching.analysis_stats.stats_extractors_change_lane import (
    extract_params_for_window,  # expects feats, roles, t0, t1
)

# ---------------------------
# Small helpers / counters
# ---------------------------

def _roles_key(roles: Mapping[str, str]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted((str(k), str(v)) for k, v in (roles or {}).items()))

def _count_windows_by_t0(windows_by_t0: Optional[Mapping[Any, List[List[int]]]]) -> int:
    """
    windows_by_t0: {t0: [[t1_lo, t1_hi], ...], ...}
    Count total windows (t0,t1) represented.
    """
    if not windows_by_t0:
        return 0
    n = 0
    for _t0, ranges in windows_by_t0.items():
        for r in ranges or []:
            if not r or len(r) != 2:
                continue
            lo, hi = int(r[0]), int(r[1])
            if hi >= lo:
                n += (hi - lo + 1)
    return int(n)

def _count_possible_windows_full(T: int, minF: int, maxF: int) -> int:
    """
    Number of all windows (t0,t1) with length in [minF..maxF] inside 0..T-1.
    length := (t1 - t0 + 1)
    """
    T = int(T)
    minF = int(minF)
    maxF = int(maxF)
    if T <= 0 or maxF <= 0:
        return 0
    minF = max(1, minF)
    maxF = max(minF, maxF)
    total = 0
    max_len = min(maxF, T)
    for L in range(minF, max_len + 1):
        total += (T - L + 1)
    return int(total)

def _mask_runs(mask: np.ndarray) -> List[Tuple[int,int,int]]:
    """
    Return contiguous true runs as (start, end, length)
    """
    if mask.size == 0:
        return []
    m = mask.astype(bool)
    # find run boundaries
    runs = []
    i = 0
    T = m.size
    while i < T:
        if not m[i]:
            i += 1
            continue
        j = i
        while j+1 < T and m[j+1]:
            j += 1
        runs.append((i, j, j - i + 1))
        i = j + 1
    return runs

def _count_possible_windows_from_valid_mask(valid: np.ndarray, minF: int, maxF: int) -> int:
    """
    Count windows that are fully contained in valid=True frames, with length in [minF..maxF].
    """
    minF = int(minF)
    maxF = int(maxF)
    if valid.size == 0:
        return 0
    total = 0
    for (_s, _e, L) in _mask_runs(valid):
        if L < minF:
            continue
        max_len = min(maxF, L)
        for wlen in range(minF, max_len + 1):
            total += (L - wlen + 1)
    return int(total)

def _sample_uniform_window_from_valid_mask(
    rng: random.Random,
    valid: np.ndarray,
    minF: int,
    maxF: int,
) -> Optional[Tuple[int,int]]:
    """
    Sample a random valid window (t0,t1) uniformly-ish (two-step: pick a run, then pick length & start).
    Not perfectly uniform over all windows, but good enough for histogram baselines.
    """
    runs = [(s,e,L) for (s,e,L) in _mask_runs(valid) if L >= minF]
    if not runs:
        return None
    s,e,L = rng.choice(runs)
    max_len = min(maxF, L)
    wlen = rng.randint(minF, max_len)
    t0 = rng.randint(s, e - wlen + 1)
    t1 = t0 + wlen - 1
    return (int(t0), int(t1))

def _dur_to_minmaxF_for_block(plan: Any, fps: int) -> Tuple[int,int]:
    """
    Uses plan.min_frames / plan.max_frames if present. Fallback to [1..T].
    """
    minF = int(getattr(plan, "min_frames", 1) or 1)
    maxF = int(getattr(plan, "max_frames", minF) or minF)
    minF = max(1, minF)
    maxF = max(minF, maxF)
    return (minF, maxF)

# ---------------------------
# Dataclasses for accumulation
# ---------------------------

@dataclass
class BlockAgg:
    bindings_with_hits: int = 0
    hit_windows: int = 0
    possible_windows_cond: int = 0

    # unconditional denominators (grounded by valid masks)
    bindings_enum_total: int = 0
    bindings_enum_with_valid_windows: int = 0
    possible_windows_uncond: int = 0
    enum_cap_reached_segments: int = 0

@dataclass
class CallAgg:
    bindings_with_hits: int = 0
    frames_on: int = 0

@dataclass
class CheckAgg:
    kind: str
    true_frames: int = 0
    total_frames: int = 0
    n_signals: int = 0

    @property
    def true_frac(self) -> float:
        return float(self.true_frames) / float(self.total_frames) if self.total_frames else 0.0

# ---------------------------
# Main collector
# ---------------------------

class StatsCollector:
    """
    Mergeable shard stats:
      - blocks: P_win_cond / P_win_uncond / P_binding denominators
      - calls: basic hit stats
      - checks: aggregated mod_stats from PerCallSignal.mod_stats (if available)
      - params: 1D histos (hit vs baseline) from sampled windows

    IMPORTANT:
      * This file is robust to ResultsStore implementation differences:
        it prefers store.by_call if present.
      * It is robust to block_hits keying:
        it iterates BlockSignal values and reads bs.segment_id/bs.roles.
    """
    def __init__(
        self,
        *,
        osc: str,
        fps: int,
        overlap: str,
        use_sed: bool,
        calls: List[Dict[str, Any]],
        plans: Mapping[str, Any],
        scn_constraints: Any,
        # sampling knobs
        n_hit_samples_per_binding: int = 30,
        n_base_samples_per_binding: int = 30,
        seed: int = 0,
        # collection toggles
        collect_uncond_denoms: bool = True,
        collect_calls: bool = True,
        collect_checks: bool = True,
        collect_params: bool = True,
        baseline_only_for_hit_bindings: bool = False,
        # caps
        max_bindings_per_seg: int = 20_000,
        baseline_bindings_cap: int = 200,
        **_ignored: Any,
    ) -> None:
        self.osc = str(osc)
        self.fps = int(fps)
        self.overlap = str(overlap)
        self.use_sed = bool(use_sed)
        self.calls = list(calls or [])
        self.plans = dict(plans or {})
        self.scn_constraints = scn_constraints

        self.n_hit_samples_per_binding = int(n_hit_samples_per_binding)
        self.n_base_samples_per_binding = int(n_base_samples_per_binding)
        self.seed = int(seed)

        self.collect_uncond_denoms = bool(collect_uncond_denoms)
        self.collect_calls = bool(collect_calls)
        self.collect_checks = bool(collect_checks)
        self.collect_params = bool(collect_params)
        self.baseline_only_for_hit_bindings = bool(baseline_only_for_hit_bindings)

        self.max_bindings_per_seg = int(max_bindings_per_seg)
        self.baseline_bindings_cap = int(baseline_bindings_cap)

        self._rng = random.Random(self.seed)

        self.n_pickles_processed = 0
        self.total_hit_samples_landed = 0
        self.total_base_samples_landed = 0

        self.blocks: Dict[str, BlockAgg] = {}
        self.calls_agg: Dict[str, CallAgg] = {}
        self.checks: Dict[str, Dict[str, CheckAgg]] = {}  # call_key_str -> check_id -> agg

        # params: {param_name: {"bins": [...], "hit": [...], "base": [...], "n_hit":..., "n_base":..., "n_invalid":...}}
        self.params: Dict[str, Dict[str, Any]] = {}

    # --------- public API ---------

    def observe_pickle(
        self,
        *,
        batch: Any,
        feats_by_seg: Mapping[str, Any],
        source_uri: str,
    ) -> None:
        self.n_pickles_processed += 1

        # 1) block-level from block_hits
        self._observe_block_hits(batch)

        # 2) call/check-level from atomic store
        store = getattr(batch, "atomic", None)
        if store is not None and (self.collect_calls or self.collect_checks):
            self._observe_call_hits_and_checks(store)

        # 3) unconditional denominators + optional baseline sampling (more expensive)
        if self.collect_uncond_denoms:
            self._observe_uncond_denoms(feats_by_seg)

        # 4) parameter histograms (needs windows: either windows_by_t0 or fallback to example_window)
        if self.collect_params:
            self._observe_params(batch, feats_by_seg)

    def to_json(self) -> Dict[str, Any]:
        # meta
        meta = {
            "osc": self.osc,
            "fps": self.fps,
            "overlap": self.overlap,
            "use_sed": self.use_sed,
            "n_hit_samples_per_binding": self.n_hit_samples_per_binding,
            "n_base_samples_per_binding": self.n_base_samples_per_binding,
            "seed": self.seed,
            "collect_uncond_denoms": self.collect_uncond_denoms,
            "collect_calls": self.collect_calls,
            "collect_checks": self.collect_checks,
            "collect_params": self.collect_params,
            "baseline_only_for_hit_bindings": self.baseline_only_for_hit_bindings,
            "max_bindings_per_seg": self.max_bindings_per_seg,
            "baseline_bindings_cap": self.baseline_bindings_cap,
            "n_pickles_processed": self.n_pickles_processed,
            "total_hit_samples_landed": self.total_hit_samples_landed,
            "total_base_samples_landed": self.total_base_samples_landed,
        }

        # blocks with derived probabilities
        blocks_out: Dict[str, Any] = {}
        for block, agg in self.blocks.items():
            minF, maxF = _dur_to_minmaxF_for_block(self.plans.get(block), self.fps)
            pwin_cond = (agg.hit_windows / agg.possible_windows_cond) if agg.possible_windows_cond else 0.0
            pwin_uncond = (agg.hit_windows / agg.possible_windows_uncond) if agg.possible_windows_uncond else 0.0
            pbinding = (agg.bindings_with_hits / agg.bindings_enum_with_valid_windows) if agg.bindings_enum_with_valid_windows else 0.0
            blocks_out[block] = {
                "minF": minF,
                "maxF": maxF,
                **asdict(agg),
                "P_win_cond": float(pwin_cond),
                "P_win_uncond": float(pwin_uncond),
                "P_binding": float(pbinding),
            }

        # calls
        calls_out = {k: asdict(v) for k, v in self.calls_agg.items()}

        # checks
        checks_out: Dict[str, Any] = {}
        for call_key, m in self.checks.items():
            checks_out[call_key] = {
                chk: {
                    "kind": agg.kind,
                    "true_frames": agg.true_frames,
                    "total_frames": agg.total_frames,
                    "true_frac": agg.true_frac,
                    "n_signals": agg.n_signals,
                }
                for chk, agg in m.items()
            }

        return {
            "meta": meta,
            "blocks": blocks_out,
            "calls": calls_out,
            "checks": checks_out,
            "params": self.params,
        }

    # --------- internals ---------

    def _observe_block_hits(self, batch: Any) -> None:
        block_hits = getattr(batch, "block_hits", {}) or {}
        for block, hitmap in block_hits.items():
            agg = self.blocks.setdefault(block, BlockAgg())
            plan = self.plans.get(block)
            if plan is None:
                continue
            minF, maxF = _dur_to_minmaxF_for_block(plan, self.fps)

            # hitmap is {(seg_id, roles_key)->BlockSignal} but keys vary; use values to be safe
            for bs in (hitmap or {}).values():
                seg_id = getattr(bs, "segment_id", None)
                roles = getattr(bs, "roles", None) or {}
                T = int(getattr(bs, "T", 0) or 0)

                # hit bindings count
                agg.bindings_with_hits += 1

                # hit windows (prefer precomputed n_windows, else windows_by_t0)
                n_windows = int(getattr(bs, "n_windows", 0) or 0)
                if not n_windows:
                    n_windows = _count_windows_by_t0(getattr(bs, "windows_by_t0", None))
                agg.hit_windows += int(n_windows)

                # conditional denom: all windows with allowed lengths inside [0..T-1]
                n_pos = int(getattr(bs, "n_possible_windows", 0) or 0)
                if not n_pos:
                    n_pos = _count_possible_windows_full(T, minF, maxF)
                agg.possible_windows_cond += int(n_pos)

    def _observe_call_hits_and_checks(self, store: Any) -> None:
        """
        Robustly iterate ResultsStore:
          prefer store.by_call if present (dict: CallKey->list[PerCallSignal])
        """
        items: Iterable[Tuple[Any, Any]] = []
        by_call = getattr(store, "by_call", None)
        if isinstance(by_call, dict):
            items = by_call.items()
        else:
            # fallback to store.items / store.data / store._data
            if callable(getattr(store, "items", None)):
                items = store.items()
            else:
                d = getattr(store, "data", None) or getattr(store, "_data", None)
                if isinstance(d, dict):
                    items = d.items()

        for call_key, sigs in items:
            # call_key is (block_label, call_index)
            try:
                block_label, ci = call_key
            except Exception:
                block_label, ci = ("", 0)
            ck_str = f"{block_label}:{int(ci)}"
            call_agg = self.calls_agg.setdefault(ck_str, CallAgg())

            if not sigs:
                continue
            for sig in sigs:
                call_agg.bindings_with_hits += 1
                mask = getattr(sig, "mask", None)
                if mask is not None:
                    call_agg.frames_on += int(np.sum(np.asarray(mask, dtype=bool)))

                if self.collect_checks:
                    mod_stats = getattr(sig, "mod_stats", None)
                    if not mod_stats:
                        continue
                    chkmap = self.checks.setdefault(ck_str, {})
                    T = int(getattr(sig, "T", 0) or 0)
                    for chk_id, st in (mod_stats or {}).items():
                        if not isinstance(st, dict):
                            continue
                        kind = str(st.get("kind") or "unknown")
                        tf = int(st.get("true_frames") or 0)
                        # best-effort total: if writer provided it, use it; else assume T
                        tot = int(st.get("total_frames") or T or 0)
                        agg = chkmap.get(chk_id)
                        if agg is None:
                            agg = CheckAgg(kind=kind)
                            chkmap[chk_id] = agg
                        agg.true_frames += tf
                        agg.total_frames += tot
                        agg.n_signals += 1

    def _observe_uncond_denoms(self, feats_by_seg: Mapping[str, Any]) -> None:
        """
        Enumerate bindings grounded by valid masks (presence gate already inside prefilter_domains via actor_valid_mask).
        This is the expensive part, but it's still much cheaper than full matching, and mergeable by summation.
        """
        # We compute per block, per segment (roles come from plan.required_roles if present; fallback: union from calls in block)
        for block, plan in self.plans.items():
            agg = self.blocks.setdefault(block, BlockAgg())
            minF, maxF = _dur_to_minmaxF_for_block(plan, self.fps)

            # derive roles for this block:
            roles = getattr(plan, "roles", None)
            if not roles:
                # fallback: union roles used by calls of that block
                roles_set = set()
                for ci, c in enumerate(self.calls):
                    if (c.get("block_label") or "") != block:
                        continue
                    for r in roles_used_by_call(c):
                        roles_set.add(r)
                roles = sorted(roles_set)

            if not roles:
                continue

            # For each segment: enumerate bindings with overlap (ego-other) constraint if roles >=2
            for seg_id, feats in (feats_by_seg or {}).items():
                resolver = make_type_to_candidates(feats)
                domains = role_domains_from_segment(self.scn_constraints, feats, roles=roles, type_to_candidates=resolver)

                # IMPORTANT: this uses actor_valid_mask (via prefilter_domains) but call-site decides min_present_frames.
                # For denominators, we want "valid for at least minF consecutive frames":
                domains = prefilter_domains(
                    feats, domains,
                    min_present_frames=minF,
                    require_speed=True,
                    require_consecutive=True,
                    osc_id_abs_max=None,  # keep default; set in role_planning.actor_valid_mask if you want stricter lane validity
                )

                actors = sorted({a for A in domains.values() for a in A})
                overlap = None
                if len(actors) >= 2:
                    # require at least minF overlap frames (not just 10)
                    overlap = build_overlap_matrix(feats, actors, min_overlap_frames=minF)

                n_bind_seg = 0
                n_bind_valid = 0
                n_possible = 0

                # cap enumeration to keep runtime sane
                for binding in enumerate_bindings(domains, distinct=True, overlap_ok=overlap, limit=self.max_bindings_per_seg):
                    n_bind_seg += 1

                    # valid mask intersection across bound actors
                    valid_masks = []
                    for _role, actor in binding.items():
                        m = getattr(feats, "present", {}).get(actor)
                        if m is None:
                            valid_masks = []
                            break
                        valid_masks.append(np.asarray(m, dtype=float) > 0.5)

                    if not valid_masks:
                        continue

                    valid = valid_masks[0]
                    for vm in valid_masks[1:]:
                        if vm.size != valid.size:
                            # guard length mismatch
                            L = min(vm.size, valid.size)
                            valid = valid[:L] & vm[:L]
                        else:
                            valid = valid & vm

                    # require at least one run with length >= minF
                    if _count_possible_windows_from_valid_mask(valid, minF=minF, maxF=minF) <= 0:
                        continue

                    n_bind_valid += 1
                    n_possible += _count_possible_windows_from_valid_mask(valid, minF=minF, maxF=maxF)

                agg.bindings_enum_total += n_bind_seg
                agg.bindings_enum_with_valid_windows += n_bind_valid
                agg.possible_windows_uncond += n_possible
                if n_bind_seg >= self.max_bindings_per_seg:
                    agg.enum_cap_reached_segments += 1

    def _observe_params(self, batch: Any, feats_by_seg: Mapping[str, Any]) -> None:
        """
        Sample windows from hit bindings (and optionally baseline-only bindings) and update histograms.
        """
        block_hits = getattr(batch, "block_hits", {}) or {}

        # A) sample from HIT bindings (hit + baseline)
        for block, hitmap in block_hits.items():
            plan = self.plans.get(block)
            if plan is None:
                continue
            minF, maxF = _dur_to_minmaxF_for_block(plan, self.fps)

            for bs in (hitmap or {}).values():
                seg_id = getattr(bs, "segment_id", None)
                roles = getattr(bs, "roles", None) or {}
                feats = (feats_by_seg or {}).get(seg_id)
                if feats is None:
                    continue
                T = int(getattr(bs, "T", 0) or getattr(feats, "T", 0) or 0)
                if T <= 0:
                    continue

                # Choose window source:
                windows_by_t0 = getattr(bs, "windows_by_t0", None)
                ex = getattr(bs, "example_window", None)

                # Hit samples
                # Thesis mode: if example_window exists, treat it as the canonical per-binding window.
                # This avoids getting 10 different values for the same binding when windows_by_t0 is present.
                n_hit_draws = int(self.n_hit_samples_per_binding)
                if ex is not None:
                    n_hit_draws = min(1, n_hit_draws) if n_hit_draws > 0 else 0

                for _ in range(n_hit_draws):
                    w = None
                    if ex is not None:
                        # canonical: (t0, t1_first, t1_greedy)
                        t0, _t1_first, t1_greedy = map(int, ex)
                        w = (t0, max(t0, min(t1_greedy, T-1)))
                    elif windows_by_t0:
                        # sample a random hit window from compact map
                        w = _sample_hit_window_from_windows_by_t0(self._rng, windows_by_t0)
                    else:
                        # last resort: sample from intervals/mask by picking a true frame and expanding to minF
                        w = _sample_from_mask_as_window(self._rng, getattr(bs, "mask", None), minF=minF, maxF=maxF)

                    if not w:
                        continue
                    t0, t1 = w
                    rec = extract_params_for_window(feats, roles, t0=t0, t1=t1, fps=self.fps)
                    self._update_param_hists(rec, hit=True)
                    self.total_hit_samples_landed += 1

                # Baseline samples (for this binding) if allowed
                if not self.baseline_only_for_hit_bindings:
                    # We'll also sample "baseline" from general valid windows grounded by presence intersection.
                    valid = _binding_valid_mask_from_present(feats, roles)
                    if valid is not None:
                        for _ in range(self.n_base_samples_per_binding):
                            w = _sample_uniform_window_from_valid_mask(self._rng, valid, minF=minF, maxF=maxF)
                            if not w:
                                continue
                            t0, t1 = w
                            rec = extract_params_for_window(feats, roles, t0=t0, t1=t1, fps=self.fps)
                            self._update_param_hists(rec, hit=False)
                            self.total_base_samples_landed += 1
                else:
                    # baseline-only-for-hit-bindings: baseline sampled only for hit bindings, using valid mask
                    valid = _binding_valid_mask_from_present(feats, roles)
                    if valid is not None:
                        for _ in range(self.n_base_samples_per_binding):
                            w = _sample_uniform_window_from_valid_mask(self._rng, valid, minF=minF, maxF=maxF)
                            if not w:
                                continue
                            t0, t1 = w
                            rec = extract_params_for_window(feats, roles, t0=t0, t1=t1, fps=self.fps)
                            self._update_param_hists(rec, hit=False)
                            self.total_base_samples_landed += 1

        # B) optional extra baseline from NON-hit bindings is intentionally not done here
        # because it's expensive; you already get a solid baseline from valid-mask sampling above.

    def _update_param_hists(self, rec: Mapping[str, Any], *, hit: bool) -> None:
        """
        rec: param->value from extractor. We auto-create bins on first sight (simple default binning).
        """
        for k, v in (rec or {}).items():
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                continue

            name = str(k)
            entry = self.params.get(name)
            if entry is None:
                # default binning: if scalar numeric, create 30 bins between min/max on the fly (lazy)
                entry = {
                    "kind": "numeric" if isinstance(v, (int, float, np.number)) else "other",
                    "hit": {},
                    "base": {},
                    "n_hit": 0,
                    "n_base": 0,
                }
                self.params[name] = entry

            bucket = entry["hit"] if hit else entry["base"]
            key = str(int(v) if isinstance(v, (bool, np.bool_)) else v)
            bucket[key] = int(bucket.get(key, 0)) + 1
            if hit:
                entry["n_hit"] += 1
            else:
                entry["n_base"] += 1


# ---------------------------
# Sampling helpers (local to module)
# ---------------------------

def _sample_hit_window_from_windows_by_t0(rng: random.Random, windows_by_t0: Mapping[Any, List[List[int]]]) -> Optional[Tuple[int,int]]:
    """
    Sample from windows_by_t0 by:
      - choose random t0 key
      - choose random range
      - choose random t1 in that range
    Not uniform over all windows, but simple + stable.
    """
    if not windows_by_t0:
        return None
    keys = list(windows_by_t0.keys())
    if not keys:
        return None
    t0k = rng.choice(keys)
    ranges = windows_by_t0.get(t0k) or []
    if not ranges:
        return None
    lo, hi = rng.choice(ranges)
    lo = int(lo); hi = int(hi)
    if hi < lo:
        return None
    t1 = rng.randint(lo, hi)
    return (int(t0k), int(t1))

def _sample_from_mask_as_window(rng: random.Random, mask: Any, *, minF: int, maxF: int) -> Optional[Tuple[int,int]]:
    if mask is None:
        return None
    m = np.asarray(mask, dtype=bool)
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return None
    t_mid = int(rng.choice(idx))
    wlen = int(rng.randint(minF, maxF))
    t0 = max(0, t_mid - wlen + 1)
    t1 = min(m.size - 1, t0 + wlen - 1)
    return (t0, t1)

def _binding_valid_mask_from_present(feats: Any, roles: Mapping[str,str]) -> Optional[np.ndarray]:
    pres = getattr(feats, "present", {}) or {}
    masks = []
    for _r, a in (roles or {}).items():
        m = pres.get(a)
        if m is None:
            return None
        masks.append(np.asarray(m, dtype=float) > 0.5)
    if not masks:
        return None
    valid = masks[0]
    for mm in masks[1:]:
        L = min(valid.size, mm.size)
        valid = valid[:L] & mm[:L]
    return valid
