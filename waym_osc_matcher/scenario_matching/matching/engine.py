# scenario_matching/matching/engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Mapping, Any

from scenario_matching.harness import OSCTestHarness, HarnessConfig
from scenario_matching.matching.results import ResultsStore, BlockSignal
from scenario_matching.matching.results.collect import collect_results
from scenario_matching.matching.post.block_combine import (
    combine_parallel_block,
    chain_serial_block,
)

__all__ = ["MatchBatchResult", "MatchEngine"]

# (seg_id, roles_key) -> BlockSignal
BlockHits = Dict[str, Dict[Tuple[str, Tuple[Tuple[str, str], ...]], BlockSignal]]


@dataclass
class MatchBatchResult:
    """
    Container for one processed batch (e.g., one pickle folder/load):
      - source_uri: provenance string for prints/auditing
      - atomic:     per-call/binding signals (ResultsStore.by_call is the main index)
      - block_hits: block-level signals keyed by block label then (segment, roles_key)
    """
    source_uri: str
    atomic: ResultsStore
    block_hits: BlockHits


class MatchEngine:
    """
    Orchestrates matching over features already loaded into an OSCTestHarness.

    Layering: this module is the ONLY place that depends on both the harness
    (features/config/calls) and the matching layer (collectors/combiner).
    The harness itself stays “thin” and has no imports from matching.
    """

    def __init__(self, cfg: HarnessConfig, scn_constraints: Mapping[str, Any], calls: List[dict]):
        self.cfg = cfg
        self.calls = list(calls)

        # A thin harness: loads/holds features + config; no matching logic inside.
        self.h = OSCTestHarness(
            osc_path="",
            entry_names={"top"},
            cfg=cfg,
            feature_provider=None,
        )
        self.h.scn_constraints = dict(scn_constraints) if scn_constraints is not None else None
        self.h.set_calls(self.calls)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def process_loaded_features_with_plans(
        self,
        *,
        plans: Mapping[str, Any],
        source_uri: str = "<unknown>",
        # optional: ask the matcher to return extra per-binding details
        collect_call_windows: bool = False,
        collect_modifier_stats: bool = False,
        max_windows_per_binding: int = 200_000,
    ) -> MatchBatchResult:
        """
        Run the full pass for the currently loaded features in the harness:
          1) collect per-call atomic signals
          2) combine them into block-level signals (parallel/serial) using 'plans'
        """
        store = self._collect_atomic(
            collect_call_windows=collect_call_windows,
            collect_modifier_stats=collect_modifier_stats,
            max_windows_per_binding=max_windows_per_binding,
        )
        T_by_seg = self._compute_T_by_seg()

        block_hits: BlockHits = {}
        for label, plan in plans.items():
            if getattr(plan, "type", None) == "parallel":
                block_hits[label] = combine_parallel_block(plan, self.calls, store, T_by_seg)

                if "31b62c6058ec8114" in source_uri:
                    hm = block_hits[label] or {}
                    keys = sorted(hm.keys(), key=lambda x: (str(x[0]), str(x[1])))
                    print(f"[COMB] {source_uri} label={label} n={len(keys)} keys={keys[:5]}", flush=True)
                    if keys:
                        bs = hm[keys[0]]
                        print(f"[COMB1] seg={getattr(bs,'segment_id',None)} roles={getattr(bs,'roles',None)} "
                            f"T={getattr(bs,'T',None)} ex={getattr(bs,'example_window',None)} "
                            f"n_windows={getattr(bs,'n_windows',None)} n_possible={getattr(bs,'n_possible_windows',None)}",
                            flush=True)
        
            elif getattr(plan, "type", None) == "serial":
                block_hits[label] = chain_serial_block(plan, self.calls, store, T_by_seg, allow_overlap=True)
            else:
                # Future types: one_of / optional / repeat / etc.
                block_hits[label] = {}

        return MatchBatchResult(source_uri=source_uri, atomic=store, block_hits=block_hits)

    # Convenience: if you ever want to expose the steps separately at call site
    def collect_atomic_only(self) -> ResultsStore:
        """Expose atomic collection as a standalone step."""
        return self._collect_atomic()

    def combine_blocks_only(
        self,
        *,
        plans: Mapping[str, Any],
        atomic: ResultsStore,
    ) -> BlockHits:
        """Given pre-collected atomic results, return combined block signals."""
        T_by_seg = self._compute_T_by_seg()
        block_hits: BlockHits = {}
        for label, plan in plans.items():
            if getattr(plan, "type", None) == "parallel":
                block_hits[label] = combine_parallel_block(plan, self.calls, atomic, T_by_seg)
            elif getattr(plan, "type", None) == "serial":
                block_hits[label] = chain_serial_block(plan, self.calls, atomic, T_by_seg, allow_overlap=True)
            else:
                block_hits[label] = {}
        return block_hits

    # Feature swapping per pickle (mirrors your previous usage)
    def set_features(self, feats_by_seg: Dict, seg_meta_by_id: Dict) -> None:
        self.h.feats_by_seg = dict(feats_by_seg or {})
        self.h.seg_meta_by_id = dict(seg_meta_by_id or {})

    def clear_features(self) -> None:
        self.h.feats_by_seg.clear()
        self.h.seg_meta_by_id.clear()

    # -------------------------------------------------------------------------
    # Internals (small, well-named helpers for readability)
    # -------------------------------------------------------------------------

    def _collect_atomic(
        self,
        *,
        collect_call_windows: bool = False,
        collect_modifier_stats: bool = False,
        max_windows_per_binding: int = 200_000,
    ) -> ResultsStore:
        """
        Uses the matching collector to build per-call/binding signals for the
        features currently loaded into the harness\.

        The extra knobs control whether match_block returns additional detail
        (call windows, per-modifier stats) needed by analysis/StatsCollector.
        """
        return collect_results(
            self.h,
            self.calls,
            collect_call_windows=collect_call_windows,
            collect_modifier_stats=collect_modifier_stats,
            max_windows_per_binding=max_windows_per_binding,
        )

    def _compute_T_by_seg(self) -> Dict[str, int]:
        """
        Return segment length (T) per segment from the currently loaded features.
        """
        return {seg: int(getattr(feats, "T", 0) or 0) for seg, feats in (self.h.feats_by_seg or {}).items()}
