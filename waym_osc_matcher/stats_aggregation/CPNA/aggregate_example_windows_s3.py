
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import boto3
import matplotlib.pyplot as plt


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def iter_keys_ending_with(s3_client, *, bucket: str, prefix: str, filename: str) -> Iterable[str]:
    """Listet alle S3-Keys unter prefix, die auf filename enden."""
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            if key.endswith("/" + filename) or key.endswith(filename):
                yield key


def stream_jsonl_from_s3(s3_client, *, bucket: str, key: str) -> Iterable[dict]:
    """Streamt JSONL aus S3 (funktioniert auch für große Dateien)."""
    resp = s3_client.get_object(Bucket=bucket, Key=key)
    body = resp["Body"]  # StreamingBody
    for raw in body.iter_lines():
        if not raw:
            continue
        try:
            line = raw.decode("utf-8").strip()
        except UnicodeDecodeError:
            continue
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def compute_histogram(values: List[float], bins: int, vmin: Optional[float], vmax: Optional[float]) -> Tuple[List[float], List[int]]:
    """Gibt (bin_edges, counts) zurück."""
    if not values:
        return [], []

    lo = min(values) if vmin is None else vmin
    hi = max(values) if vmax is None else vmax
    if hi <= lo:
        return [lo, hi], [len(values)]

    width = (hi - lo) / bins
    edges = [lo + i * width for i in range(bins + 1)]
    counts = [0] * bins

    for v in values:
        if v < lo or v > hi:
            continue
        idx = bins - 1 if v == hi else int((v - lo) / width)
        if 0 <= idx < bins:
            counts[idx] += 1

    return edges, counts


def plot_hist_grid_3x2(
    *,
    fields: Dict[str, Tuple[str, str, Optional[Tuple[float, float]]]],
    hist_data: Dict[str, Tuple[List[float], List[int]]],
    xlims: Dict[str, Optional[Tuple[float, float]]],
    save_path: Optional[str],
    show: bool,
    fig_w_in: float,
    fig_h_in: float,
):
    """Erzeugt 6 Histogramme als 3x2 Subplots."""
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )

    fig, axes = plt.subplots(3, 2, figsize=(fig_w_in, fig_h_in))
    axes_list = axes.flatten()

    for i, (fname, (title, xlabel, _fixed_range)) in enumerate(fields.items()):
        ax = axes_list[i]
        edges, counts = hist_data[fname]

        if not counts:
            ax.set_title(title)
            ax.text(0.5, 0.5, "keine Daten", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue

        centers = [(edges[j] + edges[j + 1]) / 2 for j in range(len(counts))]
        widths = [(edges[j + 1] - edges[j]) for j in range(len(counts))]

        ax.bar(centers, counts, width=widths, align="center", facecolor="blue", alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Anzahl")

        lim = xlims.get(fname)
        if lim is not None:
            ax.set_xlim(lim[0], lim[1])

    for ax in axes_list[len(fields) :]:
        ax.axis("off")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[OK] saved plot grid: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="waymo")
    ap.add_argument("--prefix", default="results/matching_fresh/phaseB/cross.osc/")
    ap.add_argument("--filename", default="example_windows.jsonl")

    ap.add_argument("--endpoint_url", default="https://gif.s3.iavgroup.local")
    ap.add_argument(
        "--verify",
        default="false",
        help='TLS verify: "false"/"true" oder Pfad zum CA-Bundle (z.B. /etc/iav-ca/IAV-CA-Bundle.pem)',
    )

    ap.add_argument("--bins", type=int, default=30)
    ap.add_argument("--min", dest="range_min", type=float, default=None)
    ap.add_argument("--max", dest="range_max", type=float, default=None)

    ap.add_argument("--max_files", type=int, default=None, help="Optional: nur erste N Dateien (Test)")
    ap.add_argument("--plot", action="store_true", help="Plot anzeigen (GUI nötig)")
    ap.add_argument("--save_plots_dir", default=None, help="Optional: Ordner zum Speichern")
    ap.add_argument("--out_name", default="hist_grid.pdf", help="Dateiname für das Grid (pdf/png)")

    # Grid-Größe (inches)
    ap.add_argument("--fig_w", type=float, default=7.2, help="Breite in inches (gut für \\textwidth)")
    ap.add_argument("--fig_h", type=float, default=7.8, help="Höhe in inches (3 Zeilen -> höher)")

    # optional: nur Zeilen mit type == "example_window"
    ap.add_argument("--only_type_example_window", action="store_true")

    args = ap.parse_args()

    # verify-Parsing
    v = str(args.verify).lower()
    if v in ("true", "1", "yes"):
        verify: object = True
    elif v in ("false", "0", "no"):
        verify = False
    else:
        verify = args.verify
        if isinstance(verify, str) and not os.path.exists(verify):
            print(f"[WARN] CA bundle not found: {verify} -> disabling TLS verification")
            verify = False

    s3 = boto3.client("s3", endpoint_url=args.endpoint_url, verify=verify)

    # 6 Felder + Labels + (optional) feste Ranges
    fields: Dict[str, Tuple[str, str, Optional[Tuple[float, float]]]] = {
        "window_dur_s": ("Szenariodauer", "Dauer [s]", None),
        "min_st_dist_m": ("Minimalabstand", "Minimaler Abstand [m]", None),
        "delta_s_start_m": ("Anfangsabstand", "Abstand zu Beginn [m]", None),
        "delta_s_end_m": ("Endabstand", "Abstand am Ende [m]", None),
        "ego_speed_mean_kph": ("Ø Ego-Geschwindigkeit", "Geschwindigkeit [km/h]", (0.0, 120.0)),
        "npc_speed_mean_kph": ("Ø NPC-Geschwindigkeit", "Geschwindigkeit [km/h]", (0.0, 30.0)),
    }

    # Transform-Regeln: Abstände als abs()
    abs_fields = {"min_st_dist_m", "delta_s_start_m", "delta_s_end_m"}

    values: Dict[str, List[float]] = defaultdict(list)

    keys = list(iter_keys_ending_with(s3, bucket=args.bucket, prefix=args.prefix, filename=args.filename))
    keys.sort()

    if args.max_files is not None:
        keys = keys[: args.max_files]

    print(f"Found {len(keys)} files matching '{args.filename}' under s3://{args.bucket}/{args.prefix}")

    total_lines = 0
    files_scanned = 0

    for key in keys:
        files_scanned += 1

        for row in stream_jsonl_from_s3(s3, bucket=args.bucket, key=key):
            if args.only_type_example_window and row.get("type") != "example_window":
                continue

            total_lines += 1

            for fname in fields.keys():
                val = safe_float(row.get(fname))
                if val is None:
                    continue
                if fname in abs_fields:
                    val = abs(val)
                values[fname].append(val)

        if files_scanned % 50 == 0:
            print(f"  scanned {files_scanned}/{len(keys)} files...")

    print("\n=== Aggregation Result ===")
    print(f"Root: s3://{args.bucket}/{args.prefix}")
    print(f"Files scanned: {files_scanned}")
    print(f"Total JSONL lines (counted): {total_lines}\n")

    # Histogramme + xlims bestimmen
    hist_data: Dict[str, Tuple[List[float], List[int]]] = {}
    xlims: Dict[str, Optional[Tuple[float, float]]] = {}

    for fname, (title, _xlabel, fixed_range) in fields.items():
        if fixed_range is not None:
            vmin, vmax = fixed_range
        else:
            # falls user global min/max setzt, gilt das – sonst None => min/max aus Daten
            vmin, vmax = args.range_min, args.range_max

        edges, counts = compute_histogram(values[fname], args.bins, vmin, vmax)
        hist_data[fname] = (edges, counts)

        if fixed_range is not None:
            xlims[fname] = fixed_range
        elif len(edges) >= 2:
            # automatische xlims auf Daten-/Histogramm-Bereich
            xlims[fname] = (edges[0], edges[-1])
        else:
            xlims[fname] = None

        print(f"{title} ({fname}): n={len(values[fname])}")
        if counts:
            for i, c in enumerate(counts):
                lo = edges[i]
                hi = edges[i + 1]
                print(f"  [{lo:10.3f}, {hi:10.3f}) : {c}")
        else:
            print("  (keine Daten)")
        print()

    save_path = None
    if args.save_plots_dir:
        os.makedirs(args.save_plots_dir, exist_ok=True)
        save_path = os.path.join(args.save_plots_dir, args.out_name)

    plot_hist_grid_3x2(
        fields=fields,
        hist_data=hist_data,
        xlims=xlims,
        save_path=save_path,
        show=args.plot,
        fig_w_in=args.fig_w,
        fig_h_in=args.fig_h,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
