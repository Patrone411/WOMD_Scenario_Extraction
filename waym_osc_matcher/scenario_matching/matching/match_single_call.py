# osc_parser/matching/match_single_call.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import copy

from .spec import build_block_query
from .match_block import match_block
from ..features.adapters import TagFeatures


def _resolve_roles_in_obj(obj: Any, binding: Dict[str, str]) -> Any:
    """Recursively replace any string equal to a role name with its bound actor_id."""
    if isinstance(obj, str):
        return binding.get(obj, obj)
    if isinstance(obj, list):
        return [_resolve_roles_in_obj(x, binding) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_resolve_roles_in_obj(x, binding) for x in obj)
    if isinstance(obj, dict):
        # keep keys, resolve values only
        return {k: _resolve_roles_in_obj(v, binding) for k, v in obj.items()}
    return obj


def _call_with_binding(call: Dict[str, Any], binding: Dict[str, str]) -> Dict[str, Any]:
    """Copy `call` and replace role strings with concrete actor ids."""
    c = copy.deepcopy(call)
    c = _resolve_roles_in_obj(c, binding)

    # Ensure the top-level 'actor' is the concrete actor id
    actor_role = call.get("actor")
    if isinstance(actor_role, str) and actor_role in binding:
        c["actor"] = binding[actor_role]
    return c


MatchOut = Union[List[Dict[str, Any]], Dict[str, Any]]


def match_for_binding(
    feats: TagFeatures,
    call: Dict[str, Any],
    binding: Dict[str, str],
    *,
    fps: int,
    max_results: int = 2000,
    cfg: Optional[Dict[str, Any]] = None,
) -> MatchOut:
    """Run matcher for a single flattened call with a specific role->actor binding.

    Notes:
      - We intentionally run UNARY: the active actor is call_resolved['actor'].
      - Any "other" references are already closed over by the built checks
        after roles are resolved to concrete actor ids.
      - `match_block` may return either:
          * list[hit] (legacy), or
          * dict with keys like: hits, endframes, windows_by_t0, mod_stats (when SED/details enabled).
        We pass the return value through unchanged.
    """
    cfg = cfg or {}

    # Resolve role names inside the call into concrete actor ids
    call_resolved = _call_with_binding(call, binding)

    # Build the query with cfg (units normalization + checks)
    Q, _pairs_hint_unused = build_block_query(call_resolved, fps=fps, cfg=cfg)

    # Force arity=1 to avoid the binary path in match_block.
    Q.arity = 1

    # Ego id is now a concrete actor id (after _call_with_binding)
    ego_id = call_resolved.get("actor")
    if not isinstance(ego_id, str):
        return []

    # Merge cfg into Q.cfg (do not clobber what build_block_query already set)
    Q.cfg = {**(getattr(Q, "cfg", {}) or {}), **cfg}

    out: MatchOut = match_block(
        feats,
        Q,
        fps=fps,
        ids=[ego_id],  # unary path
        pairs=None,
        max_results=max_results,
    )

    # Optional debug (cfg-controlled)
    if Q.cfg.get("debug_match_single_call", False):
        print(
            "[MB-OUT-KEYS]",
            type(out),
            (list(out.keys()) if isinstance(out, dict) else None),
            flush=True,
        )
        hits = out.get("hits", []) if isinstance(out, dict) else out
        for h in hits[:5]:
            print(
                f"[SEG-HIT] actor={ego_id!r} t0={h.get('t_start')} t1={h.get('t_end')} "
                f"binding_npc={binding.get('npc')!r} binding_ego={binding.get('ego_vehicle')!r}",
                flush=True,
            )

    return out
