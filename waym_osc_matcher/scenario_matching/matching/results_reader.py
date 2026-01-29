# scenario_matching/matching/results_reader.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional


# --- public API ---------------------------------------------------------------

@dataclass(frozen=True)
class BlockHit:
    source_uri: str
    waymo_tf_file_id: str   # e.g. "00000"
    scene_id: str           # e.g. "2d9d7b93bc7063b6"
    block_label: str        # e.g. "person_into_ttc"
    segment_id: str         # e.g. "seg_0"
    roles: Dict[str, str]   # role -> actor_id (as logged)
    actors_to_roles: Dict[str, str]  # actor_id -> role (inverted)
    intervals: List[List[int]]
    frames_on: int
    T: int

def iter_jsonl_block_hits(
    path: str,
    *,
    block_label: Optional[str] = None,
) -> Iterator[BlockHit]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(obj, dict):
                # optionally log/debug:
                # print("Skipping non-dict JSON line:", type(obj), repr(obj))
                continue

            if obj.get("type") != "batch":
                # skip headers or any non-batch lines
                continue

            source_uri = obj.get("source_uri") or ""
            waymo_tf_file_id, scene_id = _parse_source_uri(source_uri)
            if not waymo_tf_file_id or not scene_id:
                continue

            segments = obj.get("segments") or {}
            for seg_id, seg_payload in segments.items():
                blocks = (seg_payload or {}).get("blocks") or []
                if not blocks:
                    continue

                for b in blocks:
                    lbl = b.get("block_label")
                    if block_label and lbl != block_label:
                        continue

                    roles = dict(b.get("roles") or {})
                    actors_to_roles = {actor_id: role for role, actor_id in roles.items()}

                    yield BlockHit(
                        source_uri=source_uri,
                        waymo_tf_file_id=waymo_tf_file_id,
                        scene_id=scene_id,
                        block_label=lbl or "",
                        segment_id=b.get("segment_id", seg_id),
                        roles=roles,
                        actors_to_roles=actors_to_roles,
                        intervals=list(b.get("intervals") or []),
                        frames_on=int(b.get("frames_on") or 0),
                        T=int(b.get("T") or 0),
                    )


# --- internals ---------------------------------------------------------------

# Matches: ".../<five_digits>/<alnum>.pkl" at the end of the URI/path
_SCENE_RE = re.compile(r"/(?P<tfid>\d{5})/(?P<scene>[A-Za-z0-9]+)\.pkl$")

def _parse_source_uri(uri: str) -> tuple[str, str]:
    """
    Extract the 5-digit 'waymo_tf_file_id' and the 'scene_id' from a source_uri.

    Example:
      s3://waymo/results/k8s_run-1331520/00000/2d9d7b93bc7063b6.pkl
      -> ("00000", "2d9d7b93bc7063b6")
    """
    m = _SCENE_RE.search(uri)
    if not m:
        return "", ""
    return m.group("tfid"), m.group("scene")
