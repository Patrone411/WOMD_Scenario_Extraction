import json, os, glob, argparse
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

def deep_add(dst, src):
    for k, v in (src or {}).items():
        if isinstance(v, dict):
            dst.setdefault(k, {})
            deep_add(dst[k], v)
        elif isinstance(v, (int, float)):
            dst[k] = dst.get(k, 0) + v
        else:
            dst.setdefault(k, v)

def merge_stats(paths):
    out = {"meta": {}, "blocks": {}, "calls": {}, "checks": {}, "params": {}}
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            s = json.load(f)

        if not out["meta"]:
            out["meta"] = s.get("meta", {})
        else:
            for kk in ("n_pickles_processed", "total_hit_samples_landed", "total_base_samples_landed"):
                if kk in s.get("meta", {}):
                    out["meta"][kk] = out["meta"].get(kk, 0) + s["meta"].get(kk, 0)

        deep_add(out["blocks"], s.get("blocks", {}))
        deep_add(out["calls"],  s.get("calls", {}))
        deep_add(out["checks"], s.get("checks", {}))
        deep_add(out["params"], s.get("params", {}))
    return out

def safe_div(a, b):
    return float(a)/float(b) if b else float("nan")

def plot_blocks(merged, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    blocks = merged.get("blocks", {})
    if not blocks:
        print("no blocks in stats")
        return

    labels, p_bind, p_win_un, p_win_co = [], [], [], []
    for lbl, d in blocks.items():
        labels.append(lbl)
        p_bind.append(safe_div(d.get("bindings_with_hits", 0), d.get("bindings_enum_with_valid_windows", 0)))
        p_win_un.append(safe_div(d.get("hit_windows", 0), d.get("possible_windows_uncond", 0)))
        p_win_co.append(safe_div(d.get("hit_windows", 0), d.get("possible_windows_cond", 0)))

    x = np.arange(len(labels))
    plt.figure()
    plt.bar(x, p_win_un)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("P_win_uncond")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "blocks_pwin_uncond.png"))
    plt.close()

    plt.figure()
    plt.bar(x, p_bind)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("P_binding")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "blocks_p_binding.png"))
    plt.close()

def plot_checks(merged, out_dir, topk=30):
    os.makedirs(out_dir, exist_ok=True)
    checks = merged.get("checks", {})
    if not checks:
        print("no checks in stats")
        return

    rows = []
    for key, d in checks.items():
        if not isinstance(d, dict):
            continue
        st = safe_div(d.get("start_true", 0), d.get("start_total", 0))
        du = safe_div(d.get("during_true", 0), d.get("during_total", 0))
        en = safe_div(d.get("end_true", 0), d.get("end_total", 0))
        fracs = [v for v in (st, du, en) if not np.isnan(v)]
        tight = min(fracs) if fracs else np.nan
        rows.append((str(key), tight))

    rows.sort(key=lambda r: (np.isnan(r[1]), r[1]))
    rows = rows[:topk]

    labels = [r[0] for r in rows]
    tight = [r[1] for r in rows]

    y = np.arange(len(labels))
    plt.figure(figsize=(10, max(4, 0.25*len(labels))))
    plt.barh(y, tight)
    plt.yticks(y, labels)
    plt.xlabel("tightness (min of start/during/end true_frac)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "checks_tightness.png"))
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True,
                    help="Path to stats_shard.json OR a directory containing stats_shard.json files")
    ap.add_argument("--out", dest="out_dir", required=True,
                    help="Output directory for merged_stats.json and plots/")
    args = ap.parse_args()

    if os.path.isdir(args.in_path):
        paths = sorted(glob.glob(os.path.join(args.in_path, "**", "*.json"), recursive=True))
    else:
        paths = [args.in_path]

    merged = merge_stats(paths)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "merged_stats.json"), "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    plot_dir = os.path.join(args.out_dir, "plots")
    plot_blocks(merged, plot_dir)
    plot_checks(merged, plot_dir)

    print("\n=== Block probabilities ===")
    for lbl, d in merged.get("blocks", {}).items():
        print(
            lbl,
            "P_binding=", safe_div(d.get("bindings_with_hits",0), d.get("bindings_enum_with_valid_windows",0)),
            "P_win_uncond=", safe_div(d.get("hit_windows",0), d.get("possible_windows_uncond",0)),
            "P_win_cond=", safe_div(d.get("hit_windows",0), d.get("possible_windows_cond",0)),
        )

    print(f"\nWrote: {os.path.join(args.out_dir, 'merged_stats.json')}")
    print(f"Wrote plots in: {plot_dir}")

if __name__ == "__main__":
    main()
