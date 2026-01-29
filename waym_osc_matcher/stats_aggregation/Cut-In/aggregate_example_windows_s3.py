#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

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


def compute_xlim_from_edges(edges: List[float], fallback: Optional[Tuple[float, float]] = None) -> Optional[Tuple[float, float]]:
    """Nimmt den sichtbaren Bereich aus den Histogramm-Kanten (min/max) und gibt (xmin, xmax) zurück."""
    if len(edges) >= 2:
        return (edges[0], edges[-1])
    return fallback


def save_grid_pdf_4x2(
    *,
    fields: Dict[str, Tuple[str, str, Optional[Tuple[float, float]]]],
    hist_data: Dict[str, Tuple[List[float], List[int]]],
    out_path: str,
    fig_w_in: float,
    fig_h_in: float,
    force_xlim_from_data: bool,
):
    """Speichert Histogramme als 4x2 Grid in einer PDF (LaTeX-freundlich).
    Funktioniert auch, wenn weniger als 8 Felder vorhanden sind (restliche Achsen werden ausgeblendet).
    """
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )

    rows, cols = 4, 2
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w_in, fig_h_in))
    axes_list = axes.flatten()

    items = list(fields.items())

    for i, (fname, (title, xlabel, fixed_range)) in enumerate(items):
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

        if fixed_range is not None:
            ax.set_xlim(fixed_range[0], fixed_range[1])
        elif force_xlim_from_data:
            xlim = compute_xlim_from_edges(edges)
            if xlim is not None:
                ax.set_xlim(xlim[0], xlim[1])

    for j in range(len(items), rows * cols):
        axes_list[j].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="waymo")
    ap.add_argument("--prefix", default="results/matching_fresh/phaseB_last_4/change_lane.osc/")
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
    ap.add_argument("--plot", action="store_true", help="Grid anzeigen (GUI nötig)")
    ap.add_argument("--save_plots_dir", default=None, help="Ordner zum Speichern der Plots")
    ap.add_argument("--only_type_example_window", action="store_true")

    ap.add_argument("--grid_name", default="hists_grid.pdf", help="Dateiname für Grid (PDF empfohlen)")
    ap.add_argument("--fig_w", type=float, default=7.2, help="Grid-Breite in inches (gut für \\textwidth)")
    ap.add_argument("--fig_h", type=float, default=9.6, help="Grid-Höhe in inches (4 Zeilen -> höher)")
    ap.add_argument("--auto_xlim", action="store_true", help="x-Achse pro Plot an min/max der Daten anpassen (wenn kein fixed_range).")

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

    # Abstands-Felder -> abs()
    abs_fields = {"dist_start_m", "dist_end_m", "dist_lc_m"}

    fields = {
        "window_dur_s": ("Szenariodauer", "Dauer [s]", None),

        "ego_speed_mean_kph": ("Ø Ego-Geschwindigkeit", "Geschwindigkeit [km/h]", (0.0, 200.0)),
        "npc_speed_mean_kph": ("Ø NPC-Geschwindigkeit", "Geschwindigkeit [km/h]", (0.0, 200.0)),

        "dist_start_m": ("Anfangsabstand", "Abstand zu Beginn [m]", None),
        "dist_end_m": ("Endabstand", "Abstand am Ende [m]", None),
        "dist_lc_m": ("Abstand nach Lane Change", "Abstand nach LC [m]", None),

        # TTC nach LC: feste Range 0..60s
        "ttc_lc_2_s": ("TTC nach Lane Change", "TTC nach LC [s]", (0.0, 60.0)),
    }

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


    # Histogramme berechnen + Textausgabe
    hist_data: Dict[str, Tuple[List[float], List[int]]] = {}
    for fname, (title, _xlabel, fixed_range) in fields.items():
        if fixed_range is not None:
            vmin, vmax = fixed_range
        else:
            vmin, vmax = args.range_min, args.range_max

        edges, counts = compute_histogram(values[fname], args.bins, vmin, vmax)
        hist_data[fname] = (edges, counts)

        print(f"{title} ({fname}): n={len(values[fname])}")
        if counts:
            for i, c in enumerate(counts):
                lo = edges[i]
                hi = edges[i + 1]
                print(f"  [{lo:10.3f}, {hi:10.3f}) : {c}")
        else:
            print("  (keine Daten)")
        print()

    if args.save_plots_dir:
        os.makedirs(args.save_plots_dir, exist_ok=True)
        out_path = os.path.join(args.save_plots_dir, args.grid_name)

        save_grid_pdf_4x2(
            fields=fields,
            hist_data=hist_data,
            out_path=out_path,
            fig_w_in=args.fig_w,
            fig_h_in=args.fig_h,
            force_xlim_from_data=args.auto_xlim,
        )
        print(f"[OK] saved grid plot: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
