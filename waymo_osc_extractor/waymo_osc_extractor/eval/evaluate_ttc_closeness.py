#!/usr/bin/env python3
import argparse, json, math, re, sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

def parse_key(k: str) -> Tuple[str, str]:
    m = re.match(r"^(woe|wsm)_(.+?)_iar$", k)
    return (m.group(1), m.group(2)) if m else (None, None)

def is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)

def is_finite_num(x: Any) -> bool:
    return is_num(x) and math.isfinite(x)

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def group_by_scenario(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    scenarios: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for k, v in data.items():
        kind, sid = parse_key(k)
        if kind and sid:
            scenarios[sid][kind] = v
    return scenarios

def per_pair_metrics(scenarios: Dict[str, Dict[str, Any]], thresholds: Iterable[float]) -> pd.DataFrame:
    thr_list = list(thresholds)
    rows: List[Dict[str, Any]] = []

    for sid, packs in scenarios.items():
        if "woe" not in packs or "wsm" not in packs:
            continue
        woe = packs["woe"]; wsm = packs["wsm"]
        for ego in sorted(set(woe.keys()) & set(wsm.keys())):
            for inter in sorted(set(woe[ego].keys()) & set(wsm[ego].keys())):
                woe_pair = woe[ego][inter]
                wsm_pair = wsm[ego][inter]

                woe_ttc = woe_pair.get("ttc", [])
                wsm_ttc = wsm_pair.get("ttc_pred", wsm_pair.get("ttc", []))

                L = min(len(woe_ttc), len(wsm_ttc))
                if L == 0:
                    continue

                diffs: List[float] = []
                woe_vals: List[float] = []
                wsm_vals: List[float] = []

                n_wsm_finite = 0
                n_wsm_finite_woe_infinite = 0
                hits = {thr: 0 for thr in thr_list}

                for t in range(L):
                    wsm_t = wsm_ttc[t]
                    if is_finite_num(wsm_t):
                        n_wsm_finite += 1
                        woe_t = woe_ttc[t]
                        if is_finite_num(woe_t):
                            d = float(woe_t) - float(wsm_t)
                            diffs.append(d)
                            woe_vals.append(float(woe_t))
                            wsm_vals.append(float(wsm_t))
                            ad = abs(d)
                            for thr in thr_list:
                                if ad <= thr:
                                    hits[thr] += 1
                        else:
                            n_wsm_finite_woe_infinite += 1

                n_comp = len(diffs)
                if n_comp > 0:
                    mae = float(np.mean(np.abs(diffs)))
                    medae = float(np.median(np.abs(diffs)))
                    bias = float(np.mean(diffs))
                    rmse = float(np.sqrt(np.mean(np.square(diffs))))
                    corr = float(np.corrcoef(np.array(woe_vals), np.array(wsm_vals))[0,1]) if n_comp >= 2 else None
                else:
                    mae = medae = bias = rmse = corr = None

                row = {
                    "scenario_id": sid,
                    "ego_actor": ego,
                    "interactor": inter,
                    "wsm_finite_timesteps": n_wsm_finite,
                    "comparable_pairs": n_comp,
                    "wsm_finite_woe_infinite": n_wsm_finite_woe_infinite,
                    "mae": mae,
                    "median_abs_error": medae,
                    "bias_woe_minus_wsm": bias,
                    "rmse": rmse,
                    "corr_woe_vs_wsm": corr,
                }
                for thr in thr_list:
                    row[f"within_{thr}s_count"] = hits[thr]
                    row[f"within_{thr}s_rate"] = (hits[thr] / n_comp) if n_comp > 0 else None
                rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["scenario_id", "ego_actor", "interactor"]).reset_index(drop=True)
    return df

def agg_weighted(df: pd.DataFrame, group_cols: List[str], thresholds: Iterable[float]) -> pd.DataFrame:
    thr_list = list(thresholds)

    def _agg(g: pd.DataFrame) -> pd.Series:
        n = g["comparable_pairs"].sum()
        out = {"pairs": int(len(g)), "comparable_pairs_total": int(n)}
        if n == 0:
            out.update({
                "mae_weighted": None, "median_abs_error_weighted": None,
                "bias_weighted": None, "rmse_weighted": None, "corr_weighted": None,
                **{f"within_{thr}s_rate_weighted": None for thr in thr_list},
            })
            return pd.Series(out)
        out.update({
            "mae_weighted": float((g["mae"] * g["comparable_pairs"]).sum() / n),
            "bias_weighted": float((g["bias_woe_minus_wsm"] * g["comparable_pairs"]).sum() / n),
            "rmse_weighted": float((g["rmse"] * g["comparable_pairs"]).sum() / n),
            "corr_weighted": None,
        })
        for thr in thr_list:
            out[f"within_{thr}s_rate_weighted"] = float(g[f"within_{thr}s_count"].sum() / n)
        out["median_abs_error_weighted"] = None
        return pd.Series(out)

    if df.empty:
        return pd.DataFrame(columns=group_cols + ["pairs","comparable_pairs_total"])
    return df.groupby(group_cols, dropna=False).apply(_agg).reset_index()

def global_summary(pair_df: pd.DataFrame, thresholds: Iterable[float]) -> pd.DataFrame:
    if pair_df.empty:
        return pd.DataFrame([{
            "pairs_with_data": 0, "total_pairs": 0, "comparable_timesteps_total": 0,
            "overall_mae": None, "overall_bias": None, "overall_rmse": None
        }])
    dfc = pair_df[pair_df["comparable_pairs"] > 0].copy()
    w = dfc["comparable_pairs"]
    out = {
        "pairs_with_data": int((pair_df["comparable_pairs"] > 0).sum()),
        "total_pairs": int(len(pair_df)),
        "comparable_timesteps_total": int(dfc["comparable_pairs"].sum()),
        "overall_mae": float((dfc["mae"] * w).sum() / w.sum()) if w.sum() > 0 else None,
        "overall_bias": float((dfc["bias_woe_minus_wsm"] * w).sum() / w.sum()) if w.sum() > 0 else None,
        "overall_rmse": float((dfc["rmse"] * w).sum() / w.sum()) if w.sum() > 0 else None,
    }
    thr_list = list(thresholds)
    total_comp = dfc["comparable_pairs"].sum()
    for thr in thr_list:
        out[f"within_{thr}s_rate_overall"] = float(dfc[f"within_{thr}s_count"].sum() / total_comp) if total_comp > 0 else None
    mae_series = dfc["mae"].dropna()
    out["mae_median"] = float(mae_series.median()) if not mae_series.empty else None
    out["mae_p10"] = float(mae_series.quantile(0.10)) if not mae_series.empty else None
    out["mae_p90"] = float(mae_series.quantile(0.90)) if not mae_series.empty else None
    return pd.DataFrame([out])

def main():
    ap = argparse.ArgumentParser(description="Compare WOE[ttc] vs WSM[ttc_pred]/WSM[ttc].")
    ap.add_argument("json_path",default="inter_actors.json", help="Path to inter_actors.json")
    ap.add_argument("--outdir", default="./results/ttc", help="Directory to write CSVs")
    ap.add_argument("--thresholds", default="0.25,0.5,1,2,5", help="Comma-separated tolerance thresholds in seconds")
    args = ap.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip() != ""]
    data = load_json(args.json_path)
    scenarios = group_by_scenario(data)

    pair_df = per_pair_metrics(scenarios, thresholds)
    actor_agg = agg_weighted(pair_df, ["scenario_id", "ego_actor"], thresholds)
    scenario_agg = agg_weighted(pair_df, ["scenario_id"], thresholds)
    glob = global_summary(pair_df, thresholds)

    pair_csv = f"{args.outdir.rstrip('/')}/iar_ttc_closeness_by_pair.csv"
    actor_csv = f"{args.outdir.rstrip('/')}/iar_ttc_closeness_by_actor.csv"
    scenario_csv = f"{args.outdir.rstrip('/')}/iar_ttc_closeness_by_scenario.csv"
    summary_csv = f"{args.outdir.rstrip('/')}/iar_ttc_closeness_summary.csv"

    pair_df.to_csv(pair_csv, index=False)
    actor_agg.to_csv(actor_csv, index=False)
    scenario_agg.to_csv(scenario_csv, index=False)
    glob.to_csv(summary_csv, index=False)

    print("Wrote:")
    print("  ", pair_csv)
    print("  ", actor_csv)
    print("  ", scenario_csv)
    print("  ", summary_csv)

if __name__ == "__main__":
    sys.exit(main())
