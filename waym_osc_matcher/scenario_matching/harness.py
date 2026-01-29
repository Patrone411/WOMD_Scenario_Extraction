# scenario_matching/harness.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Iterable

from scenario_matching.features.adapters import TagFeatures
from scenario_matching.features.providers.feature_provider import FeatureLoadResult, SegmentMeta

@dataclass
class HarnessConfig:
    fps: int = 10
    duration_scope: str = "action"
    allow_shorter_end: bool = False
    coalesce_hits: Optional[bool] = None
    min_lanes: int = 2
    exact_lanes: Optional[int] = None
    debug_segments: bool = False
    debug_domains: bool = False
    debug_overlap: bool = False
    debug_compile: bool = False
    debug_match: bool = False
    debug_checks: bool = False
    legacy_duration_windows=False   # ← enable legacy windows

    # --- domain/presence gating ---
    osc_id_abs_max: float | None = None
    require_consecutive_presence: bool = True
    max_bindings_per_call_segment: int = 0  # 0 = unlimited

    # --- debug switches ---
    debug_minF: bool = False
    debug_domain_ids: bool = False
    debug_bindings: bool = False
    debug_overlap_stats: bool = False

    # --- investigation filters ---
    only_call_index: int | None = None
    only_segment_id: str | None = None

    # --- perf probe (print only slow bindings) ---
    slow_eval_threshold_s: float | None = None
    slow_eval_max_print: int = 20
    debug_window_scan: bool = False

    def to_query_cfg(self) -> dict:
        out = {
            "fps": float(self.fps),
            "duration_scope": self.duration_scope,
            "allow_shorter_end": self.allow_shorter_end,
        }
        def set_if(k, v):
            if v is not None:
                out[k] = v

        set_if("presence_min_coverage", getattr(self, "presence_min_cov", None))
        set_if("speed_min_coverage",    getattr(self, "speed_min_cov", None))
        set_if("speed_value_tol",       getattr(self, "eps_speed_mps", None))
        set_if("distance_tol",          getattr(self, "eps_distance_m", None))
        set_if("yaw_reach_tol",         getattr(self, "eps_yaw_rad", None))
        set_if("lane_id_convention",    getattr(self, "lane_id_convention", None))

        set_if("anchor_slop_frames",        getattr(self, "anchor_slop_frames", None))
        set_if("lane_eq_tol",               getattr(self, "lane_eq_tol", None))
        set_if("lane_unknown_ok",           getattr(self, "lane_unknown_ok", None))
        set_if("lane_dwell_pre",            getattr(self, "lane_dwell_pre", None))
        set_if("lane_dwell_post",           getattr(self, "lane_dwell_post", None))
        set_if("change_lane_left_is_dec",   getattr(self, "change_lane_left_is_dec", None))
        set_if("vel_smooth_frames",         getattr(self, "vel_smooth_frames", None))
        set_if("min_run_frames",            getattr(self, "min_run_frames", None))
        set_if("dilate_frames",             getattr(self, "dilate_frames", None))
        set_if("erode_frames",              getattr(self, "erode_frames", None))

        if self.coalesce_hits is not None:
            out["coalesce_hits"] = bool(self.coalesce_hits)
        if self.debug_match:
            out["debug_match_block"] = True
        if self.debug_checks:
            out["debug_checks"] = True
        out["legacy_duration_windows"] = bool(self.legacy_duration_windows)
        out["debug_sed_masks"] = bool(getattr(self, "debug_sed_masks", False))
        out["use_sed"] = bool(getattr(self, "use_sed", True))
        out["first_window_only"] = bool(getattr(self,"first_window_only", True))
        return out


@dataclass
class OSCTestHarness:
    # Inputs
    osc_path: str
    entry_names: Iterable[str]
    cfg: HarnessConfig

    scn_constraints: Optional[Dict[str, Any]] = None

    # Optional: choose one loading path
    json_path: Optional[str] = None
    feature_provider: Optional[object] = None

    # Loaded artifacts
    feats_by_seg: Dict[str, TagFeatures] = field(default_factory=dict)
    seg_meta_by_id: Dict[str, SegmentMeta] = field(default_factory=dict)
    flattened_calls: List[Dict[str, Any]] = field(default_factory=list)

    # --- public API (no matching logic here!) ---
    def load_features_only(self) -> None:
        self._load_features()

    def set_calls(self, calls: List[Dict[str, Any]]) -> None:
        self.flattened_calls = list(calls)

    # --- internal feature loading ---
    def _load_features(self) -> None:
        if self.feature_provider is not None:
            feats, meta = self._load_features_via_provider(self.feature_provider, self.cfg.min_lanes)
            feats = self._apply_lane_filter(feats)
            self.feats_by_seg = feats or {}
            self.seg_meta_by_id = meta or {}
            return

        if self.json_path:
            data = TagFeatures.load_json(self.json_path)
            feats_by_seg = TagFeatures.load_all_segments(data, min_lanes=None)
            feats_by_seg = self._apply_lane_filter(feats_by_seg)
            self.feats_by_seg = feats_by_seg or {}
            self.seg_meta_by_id = {}
            return

        raise ValueError("OSCTestHarness: no feature source configured (provider or json_path required).")

    def _load_features_via_provider(self, provider, _min_lanes: int | None):
        feats_by_seg: dict[str, TagFeatures] = {}
        seg_meta_by_id: dict[str, SegmentMeta] = {}

        from collections.abc import Iterable as _Iterable

        def _ingest(res: FeatureLoadResult):
            for seg_id, feats in (res.feats_by_seg or {}).items():
                feats_by_seg[seg_id] = feats
            for seg_id, meta in (res.seg_meta_by_id or {}).items():
                seg_meta_by_id[seg_id] = meta

        payload = provider.load()

        if isinstance(payload, FeatureLoadResult):
            _ingest(payload)
        elif isinstance(payload, _Iterable):
            for item in payload:
                if not isinstance(item, FeatureLoadResult):
                    raise TypeError(f"Provider yielded unexpected item: {type(item)}")
                _ingest(item)
        else:
            raise TypeError(f"Provider returned unexpected type: {type(payload)}")

        return feats_by_seg, seg_meta_by_id

    def _apply_lane_filter(self, feats_by_seg: Dict[str, TagFeatures]) -> Dict[str, TagFeatures]:
        if feats_by_seg is None:
            return {}

        exact = getattr(self.cfg, "exact_lanes", None)
        min_req = getattr(self.cfg, "min_lanes", None)

        def _ok(f: TagFeatures) -> bool:
            n = getattr(f, "num_lanes", None)
            if n is None:
                return False
            if exact is not None:
                return int(n) == int(exact)
            if min_req is not None:
                return int(n) >= int(min_req)
            return True

        kept = {sid: f for sid, f in feats_by_seg.items() if _ok(f)}
        return kept
