#!/usr/bin/env python3
import argparse, json, math, sys, re
from typing import Any, Dict, List
import numpy as np
import pandas as pd

def is_num(x): return isinstance(x, (int, float)) and not isinstance(x, bool)
def is_finite_num(x): return is_num(x) and math.isfinite(x)

def scenario_id_of(key: str) -> str:
    m = re.match(r"^(.*)_(general_actor_activities|actor_per_segment)$", key)
    return m.group(1) if m else key

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--thresholds", default="0.25,0.5,1.0")
    args = ap.parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]

    # Pair the two blocks that share the same scenario id
    general, perseg = {}, {}
    for k, v in data.items():
        if k.endswith("general_actor_activities"):
            general[scenario_id_of(k)] = v
        elif k.endswith("actor_per_segment"):
            perseg[scenario_id_of(k)] = v

    all_rows, actor_rows = [], []
    for scen_id, gen in general.items():
        all_payloads = gen.get("all_payloads", {})
        seg_root = perseg.get(scen_id, {})
        seg_keys = [k for k in getattr(seg_root, "keys", lambda: [])() if k.startswith("seg_")]

        for seg_id in sorted(seg_keys):
            actors = seg_root.get(seg_id, {}) or {}
            for actor_id, rec in actors.items():
                s_dot = rec.get("s_dot", [])
                valid = rec.get("valid")
                long_v = (all_payloads.get(actor_id, {}) or {}).get("long_v", [])
                L = min(len(s_dot), len(long_v))
                if L == 0: 
                    continue

                if isinstance(valid, (list, tuple)) and len(valid) == 2:
                    v0 = max(0, int(valid[0])); v1 = min(int(valid[1]), L-1)
                    idxs = list(range(v0, v1+1)) if v1 >= v0 else []
                else:
                    idxs = list(range(L))

                diffs, lv_vals, sd_vals = [], [], []
                for t in idxs:
                    lv, sd = long_v[t], s_dot[t]
                    if is_finite_num(lv) and is_finite_num(sd):
                        diffs.append(float(lv) - float(sd))
                        lv_vals.append(float(lv)); sd_vals.append(float(sd))

                row = dict(scenario=scen_id, segment=seg_id, actor=actor_id,
                           comparable_timesteps=len(diffs))
                if len(diffs) == 0:
                    row.update(dict(mae=None, bias=None, rmse=None, corr=None))
                    for thr in thresholds: row[f"within_{thr}_ms_rate"] = None
                else:
                    d = np.asarray(diffs, float)
                    row.update({
                        "mae": float(np.mean(np.abs(d))),
                        "bias": float(np.mean(d)),
                        "rmse": float(np.sqrt(np.mean(d**2))),
                        "corr": (None if len(lv_vals) < 2 else
                                 float(np.corrcoef(lv_vals, sd_vals)[0,1])
                                 if (np.std(lv_vals) > 0 and np.std(sd_vals) > 0) else None)
                    })
                    for thr in thresholds:
                        row[f"within_{thr}_ms_rate"] = float((np.abs(d) <= thr).mean())
                all_rows.append(row)

        # per-actor weighted aggregation for this scenario
        df_scen = pd.DataFrame([r for r in all_rows if r["scenario"] == scen_id])
        dfc = df_scen[df_scen["comparable_timesteps"] > 0]
        if not dfc.empty:
            w = dfc["comparable_timesteps"]
            g = dfc.groupby(["scenario", "actor"], as_index=False).apply(
                lambda gdf: pd.Series({
                    "segments": int(len(gdf)),
                    "comp_steps_total": int(gdf["comparable_timesteps"].sum()),
                    "mae_weighted": float((gdf["mae"] * gdf["comparable_timesteps"]).sum() / w.loc[gdf.index].sum()),
                    "bias_weighted": float((gdf["bias"] * gdf["comparable_timesteps"]).sum() / w.loc[gdf.index].sum()),
                    "rmse_weighted": float((gdf["rmse"] * gdf["comparable_timesteps"]).sum() / w.loc[gdf.index].sum()),
                    "corr_avg": float(np.nanmean(gdf["corr"])) if (gdf["corr"].notna().any()) else None,
                    **{f"within_{thr}_ms_rate_weighted":
                       float((gdf[f'within_{thr}_ms_rate'] * gdf['comparable_timesteps']).sum() /
                             w.loc[gdf.index].sum()) for thr in thresholds},
                })
            ).reset_index(drop=True)
            actor_rows.append(g)

    seg_out = pd.DataFrame(all_rows)
    act_out = pd.concat(actor_rows, ignore_index=True) if actor_rows else pd.DataFrame()

    # overall summary
    if not seg_out.empty:
        dfc = seg_out[seg_out["comparable_timesteps"] > 0]
        w = dfc["comparable_timesteps"]
        summary = {
            "pairs_with_data": int(len(dfc)),
            "comp_steps_total": int(w.sum()),
            "overall_mae": float((dfc["mae"] * w).sum() / w.sum()),
            "overall_bias": float((dfc["bias"] * w).sum() / w.sum()),
            "overall_rmse": float((dfc["rmse"] * w).sum() / w.sum()),
        }
        for thr in thresholds:
            summary[f"within_{thr}_ms_rate_overall"] = float((dfc[f"within_{thr}_ms_rate"] * w).sum() / w.sum())
        sum_out = pd.DataFrame([summary])
    else:
        sum_out = pd.DataFrame([{"pairs_with_data": 0, "comp_steps_total": 0}])

    seg_out.to_csv(f"{args.outdir.rstrip('/')}/longv_vs_sdot_by_segment.csv", index=False)
    act_out.to_csv(f"{args.outdir.rstrip('/')}/longv_vs_sdot_by_actor.csv", index=False)
    sum_out.to_csv(f"{args.outdir.rstrip('/')}/longv_vs_sdot_summary.csv", index=False)

if __name__ == "__main__":
    main()
