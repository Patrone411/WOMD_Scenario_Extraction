#!/usr/bin/env python3
import argparse, json, os

from osc2_parser.parser import OSCProgram
from scenario_matching.matching.post.plan import build_block_plans
from scenario_matching.matching.engine import MatchEngine
from scenario_matching.features import S3PickleSource
from scenario_matching.harness import HarnessConfig

from scenario_matching.analysis_stats.stats_collector import StatsCollector
from scenario_matching.analysis_stats.stats_extractors_change_lane import (
    PARAM_SPECS,
    extract_change_lane_features,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--osc_prefix", default="osc2_parser/osc/")
    ap.add_argument("--osc_file", default="change_lane.osc")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--overlap", default="start")
    ap.add_argument("--use_sed", action="store_true", default=True)

    ap.add_argument("--bucket", default="waymo")
    ap.add_argument("--base_prefix", default="results/k8s_run-1331520/00000")
    ap.add_argument("--endpoint_url", default="https://gif.s3.iavgroup.local")
    ap.add_argument("--verify", default="certs/IAV-CA-Bundle.pem")
    ap.add_argument("--n_pickles", type=int, default=10)

    ap.add_argument("--out", default="out_local/stats_shard.json")
    ap.add_argument("--n_hit_samples", type=int, default=30)
    ap.add_argument("--n_base_samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_bindings_per_seg", type=int, default=20000)

    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    prog = OSCProgram(osc_path=args.osc_prefix + args.osc_file).compile()
    scn = prog.constraints_by_scenario["top"]

    calls = []
    dur_by_label = getattr(prog, "block_durations", {}) or {}
    for c in prog.calls:
        c2 = dict(c)
        lbl = c2.get("block_label")
        if lbl in dur_by_label and c2.get("duration") is None:
            c2["block_duration"] = dur_by_label[lbl]
        c2.setdefault("block_overlap", args.overlap)
        calls.append(c2)

    plans = build_block_plans(calls, fps=args.fps)
    for p in plans.values():
        p.collect_block_windows = True

    cfg = HarnessConfig(
        fps=args.fps,
        exact_lanes=prog.min_lanes,
        debug_match=False,
        debug_segments=False,
        debug_checks=False,
    )
    cfg.use_sed = bool(args.use_sed)
    engine = MatchEngine(cfg=cfg, scn_constraints=scn, calls=calls)

    collector = StatsCollector(
        osc_name=args.osc_file,
        fps=args.fps,
        overlap=args.overlap,
        use_sed=bool(args.use_sed),
        scn_constraints=scn,
        calls=calls,
        plans=plans,
        param_specs=PARAM_SPECS,
        extractor=lambda feats, roles, t0, t1: extract_change_lane_features(
            feats, roles, t0, t1, left_is_decreasing=True
        ),
        n_hit_samples=args.n_hit_samples,
        n_base_samples=args.n_base_samples,
        seed=args.seed,
        max_bindings_per_seg=args.max_bindings_per_seg,
    )

    src = S3PickleSource(
        bucket=args.bucket,
        base_prefix=args.base_prefix,
        endpoint_url=args.endpoint_url,
        verify=args.verify,
        min_lanes=cfg.min_lanes,
    )

    processed = 0
    for res in src:
        if processed >= args.n_pickles:
            break
        engine.set_features(res.feats_by_seg, res.seg_meta_by_id)
        meta_any = next(iter(res.seg_meta_by_id.values()), None)
        source_uri = getattr(meta_any, "source_uri", "<unknown>")

        batch = engine.process_loaded_features_with_plans(plans=plans, source_uri=source_uri)

        collector.observe_pickle(
            feats_by_seg=res.feats_by_seg,
            batch=batch,
            source_uri=source_uri,
        )

        engine.clear_features()
        processed += 1

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(collector.to_json(), f, indent=2, ensure_ascii=False)

    print("[OK] wrote", args.out)

if __name__ == "__main__":
    main()
