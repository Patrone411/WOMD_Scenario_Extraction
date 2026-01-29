"""scenario_matching.analysis.scenario_stats

Collect baseline-vs-hit parameter distributions for blocks, using window sampling.

Designed to plug into your pipeline:
- plans: mapping label -> BlockPlan (duration_min_frames/duration_max_frames)
- batch: MatchBatchResult (block_hits[label][(seg_id, roleskey)] -> BlockSignal)
- feats_by_seg: dict seg_id -> TagFeatures

Requires plan.collect_block_windows=True so BlockSignal has windows_by_t0.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Mapping
import json
import numpy as np

from .window_sampling import sample_window_from_W, sample_window_from_H, max_possible_windows
from .online_stats import OnlineHist, OnlineCat


@dataclass
class ParamAgg:
    hit: Any
    base: Any

    def to_dict(self) -> Dict[str, Any]:
        return {"hit": self.hit.to_dict(), "baseline": self.base.to_dict()}


def _make_agg(spec: Dict[str, Any]) -> ParamAgg:
    kind = spec.get("kind", "hist")
    if kind == "hist":
        bins = spec.get("bins")
        if not bins:
            raise ValueError("hist spec requires bins")
        return ParamAgg(hit=OnlineHist.from_edges(bins), base=OnlineHist.from_edges(bins))
    if kind == "cat":
        return ParamAgg(hit=OnlineCat.empty(), base=OnlineCat.empty())
    raise ValueError(f"unknown spec kind: {kind}")


def _get_change_lane_extractor():
    from .extractors.change_lane import extract_change_lane_features, PARAM_SPECS
    return extract_change_lane_features, PARAM_SPECS


def collect_block_distributions(
    *,
    plans: Mapping[str, Any],
    batch: Any,
    feats_by_seg: Mapping[str, Any],
    block_label: str,
    n_hit_samples: int = 2000,
    n_base_samples: int = 2000,
    seed: int = 0,
    left_is_decreasing: bool = True,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)

    if block_label not in plans:
        raise KeyError(f"block_label not in plans: {block_label}")

    plan = plans[block_label]
    minF = int(getattr(plan, "duration_min_frames", 1) or 1)
    maxF_plan = getattr(plan, "duration_max_frames", None)

    if block_label != "change_lane":
        raise NotImplementedError(
            "Only block_label='change_lane' is wired in. Add more extractors in analysis/extractors."
        )

    extractor, param_specs = _get_change_lane_extractor()
    aggs: Dict[str, ParamAgg] = {k: _make_agg(spec) for k, spec in param_specs.items()}

    hitmap = (getattr(batch, "block_hits", {}) or {}).get(block_label, {}) or {}
    n_bindings = 0

    total_hit_windows = 0
    total_possible_windows = 0

    for (seg_id, _roleskey), bs in hitmap.items():
        feats = feats_by_seg.get(seg_id)
        if feats is None:
            continue
        T = int(getattr(feats, "T", 0) or 0)
        if T <= 0:
            continue

        maxF = int(maxF_plan) if maxF_plan is not None else T
        maxF = max(minF, min(maxF, T))

        w_by_t0 = getattr(bs, "windows_by_t0", None)
        if not w_by_t0:
            continue

        n_bindings += 1

        hit_windows = 0
        for ranges in w_by_t0.values():
            for lo, hi in ranges:
                hit_windows += int(hi) - int(lo) + 1
        poss_windows = max_possible_windows(T, minF, maxF)
        total_hit_windows += hit_windows
        total_possible_windows += poss_windows

        roles = getattr(bs, "roles", {}) or {}

        for _ in range(int(n_hit_samples)):
            w = sample_window_from_H(rng, w_by_t0)
            if w is None:
                break
            t0, t1 = w
            vals = extractor(feats, roles, t0, t1, left_is_decreasing=left_is_decreasing)
            if not vals:
                continue
            for name, val in vals.items():
                aggs[name].hit.add(val)

        for _ in range(int(n_base_samples)):
            w = sample_window_from_W(rng, T=T, minF=minF, maxF=maxF)
            if w is None:
                break
            t0, t1 = w
            vals = extractor(feats, roles, t0, t1, left_is_decreasing=left_is_decreasing)
            if not vals:
                continue
            for name, val in vals.items():
                aggs[name].base.add(val)

    return {
        "meta": {
            "block_label": block_label,
            "minF": minF,
            "maxF_plan": int(maxF_plan) if maxF_plan is not None else None,
            "n_bindings": n_bindings,
            "n_hit_samples_per_binding": int(n_hit_samples),
            "n_base_samples_per_binding": int(n_base_samples),
            "seed": int(seed),
            "left_is_decreasing": bool(left_is_decreasing),
            "total_hit_windows": int(total_hit_windows),
            "total_possible_windows": int(total_possible_windows),
            "P_win": (float(total_hit_windows) / float(total_possible_windows)) if total_possible_windows else None,
        },
        "params": {k: v.to_dict() for k, v in aggs.items()},
    }


def save_stats_json(path: str, stats: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
