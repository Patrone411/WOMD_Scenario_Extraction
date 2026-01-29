from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Set, Tuple, Callable


ExtractorFn = Callable[..., Dict[str, Any]]

def _clamp_int(x: int, lo: int, hi: int) -> int:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

def _jsonable(d: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure record is JSON serializable (numpy scalars, etc.)."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
            continue
        try:
            import numpy as np

            if isinstance(v, (np.integer,)):
                out[k] = int(v)
                continue
            if isinstance(v, (np.floating,)):
                out[k] = float(v)
                continue
        except Exception:
            pass
        if isinstance(v, (tuple, list)):
            out[k] = list(v)
            continue
        if isinstance(v, dict):
            out[k] = v
            continue
        out[k] = str(v)
    return out

@dataclass
class ExampleWindowWriter:
    """Append example windows (one per hit-binding) to a JSONL file."""

    path: Path

    def append_from_block_hits(
        self,
        *,
        osc: str,
        fps: int,
        source_uri: str,
        block_hits: Mapping[str, Any],
        feats_by_seg: Optional[Mapping[str, Any]] = None,
        include_counts: bool = True,
        extractor: Optional[ExtractorFn] = None,  # <<< NEU
    ) -> int:
        """Return number of records written."""

        # Fallback: altes Verhalten beibehalten
        if extractor is None:
            try:
                from scenario_matching.analysis_stats.stats_extractors_change_lane import (
                    extract_params_for_window as extractor,
                )
            except Exception:
                extractor = None

        written = 0
        seen: Set[Tuple[str, str, Tuple[Tuple[str, str], ...]]] = set()

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            for block_label, hitmap in (block_hits or {}).items():
                for _key, bs in (hitmap or {}).items():
                    seg_id = getattr(bs, "segment_id", None)
                    roles = dict(getattr(bs, "roles", {}) or {})
                    roles_key = tuple(sorted(roles.items()))
                    dedup_key = (str(block_label), str(seg_id), roles_key)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    ex = getattr(bs, "example_window", None)
                    if not ex:
                        continue
                    t0, t1_first, t1_greedy = map(int, ex)

                    T = int(getattr(bs, "T", 0) or 0)
                    if T <= 0 and feats_by_seg is not None and seg_id in feats_by_seg:
                        T = int(getattr(feats_by_seg[seg_id], "T", 0) or 0)

                    if T > 0:
                        t0 = _clamp_int(t0, 0, T - 1)
                        t1_first = _clamp_int(t1_first, t0, T - 1)
                        t1_greedy = _clamp_int(t1_greedy, t0, T - 1)

                    rec: Dict[str, Any] = {
                        "type": "example_window",
                        "osc": osc,
                        "block": str(block_label),
                        "segment_id": str(seg_id),
                        "roles": roles,
                        "T": int(T) if T else None,
                        "fps": int(fps),
                        "t0": int(t0),
                        "t1_first": int(t1_first),
                        "t1_greedy": int(t1_greedy),
                        "source_uri": source_uri,
                    }

                    if include_counts:
                        n_windows = getattr(bs, "n_windows", None)
                        n_possible = getattr(bs, "n_possible_windows", None)
                        if n_windows is not None:
                            rec["n_windows"] = int(n_windows)
                        if n_possible is not None:
                            rec["n_possible_windows"] = int(n_possible)

                    # <<< HIER der injizierte Extractor >>>
                    if extractor is not None and feats_by_seg is not None and seg_id in feats_by_seg:
                        feats = feats_by_seg[seg_id]
                        try:
                            params = extractor(
                                feats, roles, t0=t0, t1=t1_first, fps=int(fps)
                            )
                            if isinstance(params, dict):
                                for k, v in params.items():
                                    if k not in rec:
                                        rec[k] = v
                        except Exception:
                            pass

                    f.write(json.dumps(_jsonable(rec), ensure_ascii=False) + "\n")
                    written += 1

        return written


def append_example_windows_jsonl(
    *,
    path: Path,
    osc: str,
    fps: int,
    source_uri: str,
    block_hits: Mapping[str, Any],
    feats_by_seg: Optional[Mapping[str, Any]] = None,
    include_counts: bool = True,
    extractor: Optional[ExtractorFn] = None,  # <<< NEU
) -> int:
    """Convenience wrapper used from run_matching.py."""
    return ExampleWindowWriter(Path(path)).append_from_block_hits(
        osc=osc,
        fps=fps,
        source_uri=source_uri,
        block_hits=block_hits,
        feats_by_seg=feats_by_seg,
        include_counts=include_counts,
        extractor=extractor,  # <<< DURCHREICHEN
    )
