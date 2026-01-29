# osc_parser/matching/post/scenario_chain.py
#step4
#Assume top-level do serial: between blocks in their first-appearance order.
from __future__ import annotations
from typing import Dict, Tuple, List
from dataclasses import dataclass

from scenario_matching.matching.results.types import Interval
#from osc_parser.matching.results.types import Interval
from scenario_matching.matching.post.utils import coalesce_intervals
#from osc_parser.matching.post.utils import coalesce_intervals
from scenario_matching.matching.results.types import BlockSignal
RolesKey = Tuple[Tuple[str, str], ...]

@dataclass
class ScenarioTrace:
    segment_id: str
    roles: Dict[str, str]
    intervals: List[Interval]  # intervals that pass *all* blocks in order

def chain_blocks_serial(
    block_order: List[str],
    block_hits: Dict[str, Dict[Tuple[str, RolesKey], "BlockSignal"]],
    allow_overlap: bool = True,
) -> List[ScenarioTrace]:

    def _meet(a: Interval, b: Interval) -> bool:
        return (b.t0 >= a.t1) if allow_overlap else (b.t0 >= a.t1 + 1)

    if not block_order:
        return []

    # seed candidates from first block
    first = block_order[0]
    seeds: Dict[Tuple[str, RolesKey], List[Interval]] = {}
    for key, sig in block_hits.get(first, {}).items():
        seeds[key] = sig.intervals

    # chain through remaining blocks
    cur = seeds
    for lbl in block_order[1:]:
        next_hits = block_hits.get(lbl, {})
        new_cur: Dict[Tuple[str, RolesKey], List[Interval]] = {}
        for key, ivs in cur.items():
            # require same (segment, roles) in subsequent blocks
            sig = next_hits.get(key)
            if not sig:
                continue
            # chain intervals ivs (prev) -> sig.intervals (next)
            prev = sorted(ivs, key=lambda v: (v.t0, v.t1))
            nxt  = sorted(sig.intervals, key=lambda v: (v.t0, v.t1))
            out: List[Interval] = []
            j = 0
            for a in prev:
                while j < len(nxt) and nxt[j].t1 < a.t0:
                    j += 1
                k = j
                while k < len(nxt):
                    b = nxt[k]
                    if _meet(a, b):
                        out.append(Interval(a.t0, b.t1))
                    k += 1
            out = coalesce_intervals(out)
            if out:
                new_cur[key] = out
        cur = new_cur
        if not cur:
            break

    # materialize
    traces: List[ScenarioTrace] = []
    for (seg, rk), ivs in cur.items():
        traces.append(ScenarioTrace(
            segment_id=seg,
            roles={k: v for k, v in rk},
            intervals=ivs,
        ))
    return traces
