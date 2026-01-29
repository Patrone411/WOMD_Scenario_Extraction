"""scenario_matching.analysis.plotting

Simple matplotlib plots for baseline vs hit distributions (PNG).
"""
from __future__ import annotations
from typing import Dict, Any
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_hist_compare(bin_edges, base_counts, hit_counts, *, title: str, out_png: str) -> None:
    be = np.asarray(bin_edges, dtype=float)
    base = np.asarray(base_counts, dtype=float)
    hit = np.asarray(hit_counts, dtype=float)

    if base.sum() > 0:
        base = base / base.sum()
    if hit.sum() > 0:
        hit = hit / hit.sum()

    x = 0.5 * (be[:-1] + be[1:])

    plt.figure()
    plt.plot(x, base, label="baseline")
    plt.plot(x, hit, label="hit")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_cat_compare(base_counts: Dict[str, int], hit_counts: Dict[str, int], *, title: str, out_png: str) -> None:
    keys = sorted(set(base_counts.keys()) | set(hit_counts.keys()))
    b = np.array([base_counts.get(k, 0) for k in keys], dtype=float)
    h = np.array([hit_counts.get(k, 0) for k in keys], dtype=float)

    if b.sum() > 0:
        b = b / b.sum()
    if h.sum() > 0:
        h = h / h.sum()

    x = np.arange(len(keys))

    plt.figure(figsize=(max(6, len(keys) * 0.6), 4))
    plt.bar(x - 0.2, b, width=0.4, label="baseline")
    plt.bar(x + 0.2, h, width=0.4, label="hit")
    plt.xticks(x, keys, rotation=45, ha="right")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_stats_bundle(stats: Dict[str, Any], *, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    block = stats.get("meta", {}).get("block_label", "block")
    params = stats.get("params", {}) or {}
    for name, bundle in params.items():
        hit = bundle.get("hit", {}) or {}
        base = bundle.get("baseline", {}) or {}
        if hit.get("type") == "hist" and base.get("type") == "hist":
            plot_hist_compare(
                hit["bin_edges"],
                base["counts"],
                hit["counts"],
                title=f"{block} :: {name}",
                out_png=os.path.join(out_dir, f"{block}__{name}.png"),
            )
        elif hit.get("type") == "cat" and base.get("type") == "cat":
            plot_cat_compare(
                base.get("counts", {}),
                hit.get("counts", {}),
                title=f"{block} :: {name}",
                out_png=os.path.join(out_dir, f"{block}__{name}.png"),
            )
