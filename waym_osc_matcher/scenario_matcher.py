from osc2_parser.parser import OSCProgram

from scenario_matching.matching.post.plan import build_block_plans
from scenario_matching.matching.engine import MatchEngine
from scenario_matching.features import S3PickleSource
from scenario_matching.matching.pretty import print_block_results, print_calls_all_blocks, print_calls_in_block
from scenario_matching.harness import HarnessConfig
from scenario_matching.matching.log_json import append_header_json, append_batch_json

# Step 1 — parse + compile OSC
osc_prefix = "osc2_parser/osc/"
#osc_file = "walker.osc" 
#osc_file = "test_ttc.osc"
#osc_file = "cut_in.osc"
#osc_file = "get_ahead.osc"
osc_file = "change_lane.osc"
osc_file = "example.osc"
#osc_file = "cross.osc"

#osc_file = "single_block_multiple_calls.osc"
osc_path = osc_prefix + osc_file
prog = OSCProgram(osc_path=osc_prefix+osc_file).compile()
print("constraints:" , prog.constraints_by_scenario["top"])
print()
print()
print("calls", prog.calls)


scn = prog.constraints_by_scenario["top"]
dur_by_label = prog.block_durations
#print('calls: ', prog.calls)

#append block durations into calls
calls = []
for c in prog.calls:
    c2 = dict(c)
    lbl = c2.get("block_label")
    if lbl in dur_by_label and c2.get("duration") is None:
        c2["block_duration"] = dur_by_label[lbl]   # <-- HIER rein
    calls.append(c2)

for c in calls:
    c.setdefault("block_overlap", "start")

plans = build_block_plans(calls,fps=10)
for p in plans.values():
    p.collect_block_windows = True
# Step 2 — set up harness/matcher
cfg = HarnessConfig(
    fps=10,
    exact_lanes=prog.min_lanes,
    debug_match=False,
    debug_segments=False,
    debug_checks=False,

)

#cfg.debug_sed_masks = True
cfg.use_sed = True

#engine = MatchEngine(cfg, scn_constraints=prog.constraints_by_scenario["top"], calls=prog.calls)
engine = MatchEngine(
    cfg=cfg,
    scn_constraints=prog.constraints_by_scenario["top"],
    calls=calls,
)
# Step 2b — feature source (folder of pickles)
src = S3PickleSource(
    bucket="waymo",
    base_prefix="results/k8s_run-1331520/00000",
    endpoint_url="https://gif.s3.iavgroup.local",
    verify="certs/IAV-CA-Bundle.pem",
    min_lanes=cfg.min_lanes,
)

log_path = f"match_run_{osc_file}_test.jsonl"
append_header_json(log_path, osc_name=osc_file, calls_flat= prog.constraints_by_scenario)


# Step 3 — process each pickle; collect atomic + block signals
for res in src:
    if batch.source_uri != "batch.source_uri" :
        continue
    #print("engine start")
    engine.set_features(res.feats_by_seg, res.seg_meta_by_id)
    meta_any = next(iter(res.seg_meta_by_id.values()), None)
    pkl_uri = getattr(meta_any, "source_uri", "<unknown>")
    #print("got data from pickle")
    batch = engine.process_loaded_features_with_plans(plans=plans, source_uri=pkl_uri)
    #print("running matching")

    print(f"\n=== {batch.source_uri} ===")
    print_block_results(plans, batch.block_hits, show_blocks=10, show_bindings=5, width=100)
    # ▼▼ NEW: show per-call results (choose one style) ▼▼


    print_calls_all_blocks(
        plans, engine.calls, batch.atomic,  # <- RICHTIG
        show_calls_per_block=10, show_bindings=3, width=100,
        fps=engine.cfg.fps, cfg=engine.cfg.to_query_cfg()
    )

    # B) Just a single block (replace with your block label)
    # first_label = next(iter(prog.plans.keys()), None)
    # if first_label:
    #     print_calls_in_block(
    #         prog.plans, prog.calls, batch.store, first_label,
    #         show_calls=10, show_bindings=3, width=100,
    #         fps=engine.cfg.fps, cfg=engine.cfg.to_query_cfg()
    #     )

    # C) Flat order (ignoring blocks)
    # print_calls_flat(
    #     prog.calls, batch.store,
    #     show_calls=10, show_bindings=3, width=100,
    #     fps=engine.cfg.fps, cfg=engine.cfg.to_query_cfg()
    # )

    append_batch_json(
        log_path,
        source_uri=pkl_uri,
        store=batch.atomic,
        block_hits=batch.block_hits,
        plans=plans,
        calls=engine.calls,
        restrict_calls_to_block_hits=True,     # ← only calls that belong to a block hit
        require_interval_overlap=True,         # ← and they must overlap the block intervals
        drop_segments_without_block_hits=True, # ← optional, default True when restricting
        skip_empty=True,
        pretty=False
    )

    engine.clear_features()