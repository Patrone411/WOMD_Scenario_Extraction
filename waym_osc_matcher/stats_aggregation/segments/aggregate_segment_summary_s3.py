#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import os
import boto3
import matplotlib.pyplot as plt


def safe_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except (TypeError, ValueError):
        return None


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


@dataclass
class Hist:
    bins: int
    vmin: Optional[float] = None
    vmax: Optional[float] = None

    def build(self, values: List[float]) -> Tuple[List[float], List[int]]:
        if not values:
            return [], []

        vmin = min(values) if self.vmin is None else self.vmin
        vmax = max(values) if self.vmax is None else self.vmax

        if vmax <= vmin:
            return [vmin, vmax], [len(values)]

        width = (vmax - vmin) / self.bins
        edges = [vmin + i * width for i in range(self.bins + 1)]
        counts = [0] * self.bins

        for v in values:
            if v < vmin or v > vmax:
                continue
            idx = self.bins - 1 if v == vmax else int((v - vmin) / width)
            if 0 <= idx < self.bins:
                counts[idx] += 1

        return edges, counts


def iter_segment_summary_keys(
    s3_client,
    *,
    bucket: str,
    prefix: str,
    filename: str = "segment_summary.jsonl",
) -> Iterable[str]:
    """Listet alle S3-Keys unter prefix, die auf filename enden."""
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            if key.endswith("/" + filename) or key.endswith(filename):
                yield key


def stream_jsonl_from_s3_object(s3_client, *, bucket: str, key: str) -> Iterable[dict]:
    """Streamt JSONL aus S3. Funktioniert auch für große Dateien."""
    resp = s3_client.get_object(Bucket=bucket, Key=key)
    body = resp["Body"]  # StreamingBody
    for raw_line in body.iter_lines():
        if not raw_line:
            continue
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError:
            continue
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def compute_xlim_from_edges(edges: List[float], fallback: Optional[Tuple[float, float]] = None) -> Optional[Tuple[float, float]]:
    """Wie in deinem Grid-Skript: xlim aus edges."""
    if len(edges) >= 2:
        return (edges[0], edges[-1])
    return fallback


def plot_hist_ax(
    ax,
    edges: List[float],
    counts: List[int],
    title: str,
    xlabel: str,
    fixed_range: Optional[Tuple[float, float]] = None,
    force_xlim_from_data: bool = True,
):
    """Histogramm in ein gegebenes Axes-Objekt, im selben Stil wie dein Grid-Code."""
    if not counts:
        ax.set_title(title)
        ax.text(0.5, 0.5, "keine Daten", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
    widths = [(edges[i + 1] - edges[i]) for i in range(len(counts))]

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="waymo")
    ap.add_argument("--prefix", default="results/matching_fresh/phaseA_wide/ttc.osc/")
    ap.add_argument("--filename", default="segment_summary.jsonl")

    ap.add_argument("--endpoint_url", default="https://gif.s3.iavgroup.local")
    ap.add_argument(
        "--verify",
        default="/etc/iav-ca/IAV-CA-Bundle.pem",
        help='TLS verify: "false"/"true" oder Pfad zum CA-Bundle',
    )

    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--min", dest="range_min", type=float, default=None)
    ap.add_argument("--max", dest="range_max", type=float, default=None)
    ap.add_argument("--max_files", type=int, default=None)

    ap.add_argument("--plot", action="store_true", help="Plot anzeigen (GUI nötig)")
    ap.add_argument("--save_plots_dir", default=None, help="Ordner zum Speichern")
    ap.add_argument("--out_name", default="length_m_by_num_lanes.pdf", help="Dateiname (pdf/png)")

    # Größe/Style wie dein Grid-Skript
    ap.add_argument("--fig_w", type=float, default=7.2, help="Breite in inches (gut für \\textwidth)")
    ap.add_argument("--fig_h", type=float, default=3.2, help="Höhe in inches (1x2 Layout)")
    ap.add_argument("--auto_xlim", action="store_true", help="x-Achse aus Daten-min/max (edges) setzen")

    args = ap.parse_args()

    # boto3 verify: True/False oder Pfad
    v = str(args.verify).lower()
    if v in ("true", "1", "yes"):
        verify: object = True
    elif v in ("false", "0", "no"):
        verify = False
    else:
        verify = args.verify

    s3 = boto3.client("s3", endpoint_url=args.endpoint_url, verify=verify)

    count_by_lanes = Counter()
    lengths_by_lanes: Dict[int, List[float]] = {2: [], 3: []}

    keys = list(iter_segment_summary_keys(s3, bucket=args.bucket, prefix=args.prefix, filename=args.filename))
    keys.sort()

    if args.max_files is not None:
        keys = keys[: args.max_files]

    print(f"Found {len(keys)} files matching '{args.filename}' under s3://{args.bucket}/{args.prefix}")

    files_scanned = 0
    lines_read = 0

    for key in keys:
        files_scanned += 1
        for row in stream_jsonl_from_s3_object(s3, bucket=args.bucket, key=key):
            lines_read += 1
            num_lanes = safe_int(row.get("num_lanes"))
            length_m = safe_float(row.get("length_m"))

            if num_lanes in (2, 3):
                count_by_lanes[num_lanes] += 1
                if length_m is not None:
                    lengths_by_lanes[num_lanes].append(length_m)

        if files_scanned % 50 == 0:
            print(f"  scanned {files_scanned}/{len(keys)} files...")

    hist = Hist(bins=args.bins, vmin=args.range_min, vmax=args.range_max)
    edges2, counts2 = hist.build(lengths_by_lanes[2])
    edges3, counts3 = hist.build(lengths_by_lanes[3])

    print("\n=== Aggregation Result ===")
    print(f"S3 root: s3://{args.bucket}/{args.prefix}")
    print(f"Files scanned: {files_scanned}")
    print(f"Lines read:   {lines_read}")
    print()
    print(f"Segments num_lanes=2: {count_by_lanes[2]}")
    print(f"Segments num_lanes=3: {count_by_lanes[3]}")
    print()

    # Stil identisch zu deinem Grid-Skript
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(args.fig_w, args.fig_h))

    plot_hist_ax(
        axes[0],
        edges2,
        counts2,
        title="Roadsegmente nach Länge (2 Fahrstreifen)",
        xlabel="Länge [m]",
        fixed_range=None,
        force_xlim_from_data=args.auto_xlim,
    )
    plot_hist_ax(
        axes[1],
        edges3,
        counts3,
        title="Roadsegmente nach Länge (3 Fahrstreifen)",
        xlabel="Länge [m]",
        fixed_range=None,
        force_xlim_from_data=args.auto_xlim,
    )

    fig.tight_layout()

    if args.save_plots_dir:
        os.makedirs(args.save_plots_dir, exist_ok=True)
        out_path = os.path.join(args.save_plots_dir, args.out_name)
        fig.savefig(out_path, bbox_inches="tight")
        print(f"[OK] saved: {out_path}")

    if args.plot or not args.save_plots_dir:
        plt.show()

    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
