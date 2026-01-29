# tests/test_semantics_checks.py
import numpy as np
import pytest

# import your module under test
from osc2_parser.matching.spec import (
    build_block_query,
    _check_follow_lane,
    _check_keep_lane,
    _check_keep_speed,
    _check_distance_traveled,
    _check_change_lane_action,
    _check_change_time_headway,
    _check_keep_time_headway,
    _check_change_space_gap,
    _DEFAULT_CFG,
    _check_lateral,
)

class Feats:
    """
    Minimal feature bundle matching the fields read by your checks.
    Arrays are per-actor dicts unless noted as relation dicts on (ego, npc).
    """
    def __init__(self, T):
        self.T = T
        # per-actor
        self.present     = {}
        self.speed       = {}
        self.accel       = {}
        self.yaw         = {}
        self.yaw_delta   = {}
        self.lane_idx    = {}
        self.s           = {}
        self.t           = {}
        self.s_dot       = {}
        self.t_dot       = {}
        self.x           = {}
        self.y           = {}

        # pairwise (ego, npc) -> time series or list[str]
        self.rel_position = {}
        self.lat_rel      = {}
        self.rel_distance = {}

def _const(T, val):
    return np.full(T, float(val), dtype=float)

def _ramp(T, start=0.0, step=1.0):
    return np.array([start + k*step for k in range(T)], dtype=float)

def _derive_relations(F: Feats, ego: str, npc: str):
    """Fill rel_position and lat_rel from s/t (front/back and left/right)."""
    s_e = F.s.get(ego); s_n = F.s.get(npc)
    t_e = F.t.get(ego); t_n = F.t.get(npc)
    T = F.T
    rp = []
    lr = []
    for i in range(T):
        # front/back
        if s_e is None or s_n is None or i >= len(s_e) or i >= len(s_n) \
           or not np.isfinite(s_e[i]) or not np.isfinite(s_n[i]):
            rp.append("unknown")
        else:
            rp.append("front" if (s_e[i] - s_n[i]) >= 0 else "back")
        # left/right
        if t_e is None or t_n is None or i >= len(t_e) or i >= len(t_n) \
           or not np.isfinite(t_e[i]) or not np.isfinite(t_n[i]):
            lr.append("unknown")
        else:
            lr.append("left" if (t_e[i] - t_n[i]) >= 0 else "right")
    F.rel_position[(ego, npc)] = rp
    F.lat_rel[(ego, npc)]      = lr

def _run_block_query(Q, feats, t0=0):
    """Run all compiled checks of a BlockQuery for a single window [t0..t1]."""
    t1 = t0 + Q.duration_frames - 1
    results = [chk(feats, Q.ego, None, t0, t1, Q.cfg) for chk in Q.checks]
    ok = all(bool(r) for r in results)
    return ok, results

# Default cfg with strict 'all' (matches your new default)
CFG = {**_DEFAULT_CFG, "during_mode": "all", "fps": 10.0}



def test_keep_lane_modifier_passes():
    T = 10
    F = Feats(T)
    E = "ego"

    F.present[E]  = _const(T, 1)
    F.lane_idx[E] = _const(T, 3)   # stays in lane 3

    # Direct modifier check via helper that delegates to follow_lane
    assert _check_keep_lane(F, E, None, 0, T-1, CFG)

    # Through build_block_query as a modifier on a neutral action
    call = {
        "actor": E,
        "action": "drive",  # not handled → only modifiers run
        "action_args": {"duration": {"value": 1, "unit": "s"}},
        "modifiers": [{"name": "keep_lane", "args": {}}],
    }
    Q, pairs = build_block_query(call, fps=10, cfg=CFG)
    ok, _ = _run_block_query(Q, F, t0=0)
    assert ok


def test_follow_lane_action_target_lane():
    T = 10
    F = Feats(T)
    E = "ego"

    F.present[E]  = _const(T, 1)
    F.lane_idx[E] = _const(T, 2)

    # direct call
    assert _check_follow_lane(F, E, None, 0, T-1, CFG, target_lane=2)

    # action route
    call = {
        "actor": E,
        "action": "follow_lane",
        "action_args": {"target": {"lane": 2}, "duration": {"value": 1, "unit": "s"}},
        "modifiers": [],
    }
    Q, _ = build_block_query(call, fps=10, cfg=CFG)
    ok, _ = _run_block_query(Q, F, t0=0)
    assert ok


def test_change_lane_action_left_by_1():
    T = 20
    F = Feats(T)
    E = "ego"

    F.present[E] = _const(T, 1)
    # Start in lane 2-, end in lane -1
    lanes = np.array([-2]*10 + [-1]*10, dtype=float)
    F.lane_idx[E] = lanes
    assert _check_change_lane_action(F, E, None, 0, T-1, CFG,
                                    target_lane=None, num_of_lanes=1, side="left", reference=E)

    # through build_block_query
    call = {
        "actor": E,
        "action": "change_lane",
        "action_args": {
            "num_of_lanes": 1,
            "side": "left",
            "reference": E,
            "duration": {"value": 2, "unit": "s"}
        },
        "modifiers": [],
    }
    Q, _ = build_block_query(call, fps=10, cfg=CFG)
    ok, _ = _run_block_query(Q, F, t0=0)
    assert ok

def test_change_lane_action_right_by_1_odr():
    T, E = 10, "ego"
    F = Feats(T)
    F.present[E]  = _const(T, 1)
    # RHT forward: lanes on our carriageway are negative. Right is more negative.
    F.lane_idx[E] = np.array([-1]*5 + [-2]*5, dtype=float)

    assert _check_change_lane_action(
        F, E, None, 0, T-1, CFG,
        target_lane=None, num_of_lanes=1, side="right", reference=E
    )

def test_change_lane_reference_other_actor_same_as():
    T, E, N = 10, "ego", "npc"
    F = Feats(T)
    F.present[E] = F.present[N] = _const(T, 1)
    F.lane_idx[N] = np.array([-2]*T, dtype=float)
    # Ego moves from -3 to -2 (to match NPC at end)
    F.lane_idx[E] = np.array([-3]*5 + [-2]*5, dtype=float)

    assert _check_change_lane_action(
        F, E, None, 0, T-1, CFG,
        target_lane=None, num_of_lanes=None, side="same", reference=N
    )


def test_keep_speed_modifier_all_frames():
    T = 15
    F = Feats(T)
    E = "ego"

    F.present[E] = _const(T, 1)
    F.speed[E]   = _const(T, 10.0)  # m/s constant

    # direct
    assert _check_keep_speed(F, E, None, 0, T-1, CFG)

    # now introduce one violating frame (> tol)
    bad = F.speed[E].copy()
    bad[7] += CFG.get("keep_speed_tol", 0.2) + 0.05
    F.speed[E] = bad
    # 'all' semantics → should fail
    assert not _check_keep_speed(F, E, None, 0, T-1, CFG)


def test_distance_traveled_modifier():
    T = 11
    F = Feats(T)
    E = "ego"
    F.present[E] = _const(T, 1)
    # Move +5 m each step → from s[0]=0 to s[10]=50
    F.s[E] = _ramp(T, start=0.0, step=5.0)

    # modifier: distance(50 m) during the phase
    dist = {"value": 50.0, "unit": "m"}
    assert _check_distance_traveled(F, E, None, 0, T-1, CFG, dist)

    # via build_block_query
    call = {
        "actor": E,
        "action": "drive",
        "action_args": {"duration": {"value": 1, "unit": "s"}},  # T=11 at fps 10 → window 10 frames is fine
        "modifiers": [{"name": "distance", "args": {"distance": dist}}],
    }
    Q, _ = build_block_query(call, fps=10, cfg=CFG)
    ok, _ = _run_block_query(Q, F, t0=0)  # window len=10 frames → s[10]-s[0]=50
    assert ok


def test_change_time_headway_ahead_end_anchored():
    T = 21
    F = Feats(T)
    E, N = "ego", "npc"

    # ego moves at 1 m/s, NPC is 10 m behind all the time → ego is 'front'
    F.present[E] = _const(T, 1)
    F.present[N] = _const(T, 1)
    F.s[E]       = _ramp(T, 0.0, 1.0)
    F.s[N]       = F.s[E] - 10.0
    F.s_dot[E]   = _const(T, 1.0)   # to define v_long
    _derive_relations(F, E, N)

    # End headway ≈ |Δs|/|v| = 10 / 1 = 10 s
    tgt = {"value": 10.0, "unit": "s"}

    assert _check_change_time_headway(F, E, N, 0, T-1, CFG, tgt, "ahead")

    call = {
        "actor": E,
        "action": "change_time_headway",
        "action_args": {
            "reference": N,
            "direction": "ahead",
            "target": tgt,
            "duration": {"value": 2, "unit": "s"},
        },
        "modifiers": [],
    }
    Q, _ = build_block_query(call, fps=10, cfg=CFG)
    ok, _ = _run_block_query(Q, F, t0=0)
    assert ok


def test_keep_time_headway_sampled_at_start():
    T = 30
    F = Feats(T)
    E, N = "ego", "npc"

    F.present[E] = _const(T, 1)
    F.present[N] = _const(T, 1)
    # Keep Δs ≈ 12 m and |v|= 2 m/s → headway ≈ 6 s
    F.s[E]     = _ramp(T, 0.0, 2.0)
    F.s[N]     = F.s[E] - 12.0
    F.s_dot[E] = _const(T, 2.0)
    _derive_relations(F, E, N)

    # sampled at t0, must be kept (within tol) for all frames
    assert _check_keep_time_headway(F, E, N, 0, T-1, CFG)

    call = {
        "actor": E,
        "action": "keep_time_headway",
        "action_args": {"reference": N, "duration": {"value": 2, "unit": "s"}},
        "modifiers": [],
    }
    Q, _ = build_block_query(call, fps=10, cfg=CFG)
    ok, _ = _run_block_query(Q, F, t0=0)
    assert ok


def test_change_space_gap_lateral_left():
    T = 15
    F = Feats(T)
    E, N = "ego", "npc"

    F.present[E] = _const(T, 1)
    F.present[N] = _const(T, 1)
    F.s[E] = _ramp(T, 0.0, 1.0)
    F.s[N] = _ramp(T, 0.0, 1.0)  # same longitudinal, emphasize lateral relationship
    F.t[E] = _const(T, 2.0)      # 2 m left of NPC
    F.t[N] = _const(T, 0.0)
    _derive_relations(F, E, N)

    target = {"value": 2.0, "unit": "m"}
    assert _check_change_space_gap(F, E, N, 0, T-1, CFG, target, "left")

    call = {
        "actor": E,
        "action": "change_space_gap",
        "action_args": {
            "target": target,
            "direction": "left",
            "reference": N,
            "duration": {"value": 1, "unit": "s"},
        },
        "modifiers": [],
    }
    Q, _ = build_block_query(call, fps=10, cfg=CFG)
    ok, _ = _run_block_query(Q, F, t0=0)
    assert ok


def test_keep_time_headway_low_speed_undefined():
    T, E, N = 15, "ego", "npc"
    F = Feats(T)
    F.present[E] = F.present[N] = _const(T, 1)
    F.s[E] = _ramp(T, 0.0, 0.05)      # 0.05 m/s << min_speed_for_headway (0.30)
    F.s[N] = F.s[E] - 5.0
    _derive_relations(F, E, N)

    assert not _check_keep_time_headway(F, E, N, 0, T-1, CFG)

def test_lateral_distance_missing_allows_when_configured():
    T, E, N = 8, "ego", "npc"
    F = Feats(T)
    F.present[E] = F.present[N] = _const(T, 1)
    F.lat_rel[(E,N)] = ["left"]*T
    # No t arrays at all
    cfg = {**CFG, "lateral_allow_missing": True}
    assert _check_lateral(F, E, N, 0, T-1, cfg, side="left", dist_arg={"value":2,"unit":"m"}, at="end")
