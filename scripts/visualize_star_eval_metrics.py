#!/usr/bin/env python3
"""
Visualize STAR closed-vocab eval output (metrics.json from eval_star_closed_predictions.py).

Generates PNG figures under --out_dir:
  - macro_metrics.png          — macro averages (single run) or grouped bars (multi run)
  - per_image_hist.png         — distributions of per-image metrics
  - coverage_vs_recall.png     — scatter: object_recall vs triplet_recall per image (optional)

Examples:
  python scripts/visualize_star_eval_metrics.py \\
    --metrics_json /path/to/preds_dir/metrics.json

  python scripts/visualize_star_eval_metrics.py \\
    --metrics_json baseline/metrics.json sft/metrics.json \\
    --run_labels BL SFT \\
    --out_dir experiments/viz_runs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_metrics(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = ("macro_object_recall", "macro_triplet_recall", "macro_mean_iou_matched")
    for k in required:
        if k not in data:
            raise KeyError(f"{path}: missing key {k!r}")
    return data


def _setup_font() -> None:
    """Use first available font from candidates (avoids missing-font spam)."""
    from matplotlib import font_manager

    plt.rcParams["axes.unicode_minus"] = False
    names = {f.name for f in font_manager.fontManager.ttflist}
    for fam in ("Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"):
        if fam in names:
            plt.rcParams["font.sans-serif"] = [fam]
            return
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


def plot_macro_single(m: dict, out_path: Path, title: str | None = None) -> None:
    labels = ["Object recall", "Triplet recall", "Mean IoU (matched)"]
    vals = [
        m["macro_object_recall"],
        m["macro_triplet_recall"],
        m["macro_mean_iou_matched"],
    ]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    for i, b in enumerate(bars):
        h = b.get_height()
        ax.annotate(
            f"{vals[i]:.3f}",
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    t = title or "STAR closed-vocab — macro metrics"
    ax.set_title(t)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_macro_compare(metrics_list: list[dict], run_labels: list[str], out_path: Path) -> None:
    categories = ["Object recall", "Triplet recall", "Mean IoU"]
    n_runs = len(metrics_list)
    n_cat = len(categories)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.2 + n_cat * 1.6), 4.5))
    width = 0.8 / n_runs
    x = np.arange(n_cat)
    colors = plt.cm.tab10(np.linspace(0, 0.85, n_runs))
    for r, (m, lab, c) in enumerate(zip(metrics_list, run_labels, colors)):
        vals = [
            m["macro_object_recall"],
            m["macro_triplet_recall"],
            m["macro_mean_iou_matched"],
        ]
        offset = -0.4 + width / 2 + r * width
        ax.bar(x + offset, vals, width, label=lab, color=c)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=12, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("STAR closed-vocab — macro metrics (comparison)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_per_image_histograms(m: dict, out_path: Path) -> None:
    per = m.get("per_image") or []
    if not per:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No per_image in metrics.json", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return

    obj_r = [float(p["object_recall"]) for p in per if "error" not in p]
    trip_r = [float(p["triplet_recall"]) for p in per if "error" not in p]
    miou = [float(p["mean_iou_matched"]) for p in per if "error" not in p]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    bins = min(25, max(8, len(per) // 5))
    for ax, data, title in zip(
        axes,
        (obj_r, trip_r, miou),
        ("Per-image object recall", "Per-image triplet recall", "Per-image mean IoU (matched)"),
    ):
        ax.hist(data, bins=bins, color="#4C78A8", edgecolor="white", alpha=0.9)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 1.05)
    fig.suptitle("Per-image score distributions", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_extended_macro(m: dict, out_path: Path, title: str | None = None) -> None:
    """Extra bars when eval_star_closed_predictions.py wrote macro_object_precision / macro_f1_object."""
    if "macro_object_precision" not in m:
        return
    labels = ["Object recall", "Object precision", "Object F1", "Triplet recall", "Mean IoU (matched)"]
    vals = [
        float(m["macro_object_recall"]),
        float(m["macro_object_precision"]),
        float(m.get("macro_f1_object", 0.0)),
        float(m["macro_triplet_recall"]),
        float(m["macro_mean_iou_matched"]),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x = np.arange(len(labels))
    colors = ["#4C78A8", "#72B7B2", "#B279A2", "#F58518", "#54A24B"]
    bars = ax.bar(x, vals, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    for i, b in enumerate(bars):
        h = b.get_height()
        ax.annotate(
            f"{vals[i]:.3f}",
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    psr = m.get("parse_success_rate")
    if psr is not None:
        ax.text(
            0.02,
            0.98,
            f"Parse success: {float(psr):.1%}  |  mean GT objs: {m.get('mean_gt_objects', 0):.1f}  pred: {m.get('mean_pred_objects', 0):.1f}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            color="#333333",
        )
    t = title or "STAR closed-vocab — extended macro metrics"
    ax.set_title(t)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_per_image_precision_hist(m: dict, out_path: Path) -> None:
    per = m.get("per_image") or []
    if not per or "macro_object_precision" not in m:
        return
    data = [float(p["object_precision"]) for p in per if "error" not in p and "object_precision" in p]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    bins = min(20, max(8, len(data) // 4))
    ax.hist(data, bins=bins, color="#72B7B2", edgecolor="white", alpha=0.9)
    ax.set_title("Per-image object precision (matched / pred count)")
    ax.set_xlim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_scatter_obj_vs_triplet(m: dict, out_path: Path) -> None:
    per = m.get("per_image") or []
    xs, ys = [], []
    for p in per:
        if "error" in p:
            continue
        xs.append(float(p["object_recall"]))
        ys.append(float(p["triplet_recall"]))
    if len(xs) < 2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, "Not enough points for scatter", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(xs, ys, alpha=0.45, s=22, c="#4C78A8")
    ax.set_xlabel("Object recall (per image)")
    ax.set_ylabel("Triplet recall (per image)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Object vs triplet recall")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot STAR eval metrics.json")
    ap.add_argument(
        "--metrics_json",
        nargs="+",
        required=True,
        help="One or more metrics.json paths from eval_star_closed_predictions.py",
    )
    ap.add_argument("--run_labels", nargs="*", default=None, help="Labels for each metrics file (same order)")
    ap.add_argument("--out_dir", type=str, default="eval_viz", help="Output directory for PNG files")
    ap.add_argument("--title", type=str, default=None, help="Title suffix for single-run macro chart")
    args = ap.parse_args()

    paths = [Path(p).resolve() for p in args.metrics_json]
    for p in paths:
        if not p.is_file():
            raise SystemExit(f"File not found: {p}")

    _setup_font()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = [_load_metrics(p) for p in paths]
    labels = args.run_labels
    if labels is None:
        labels = [p.parent.name or p.stem for p in paths]
    if len(labels) != len(paths):
        raise SystemExit("--run_labels must match --metrics_json count")

    if len(loaded) == 1:
        m = loaded[0]
        plot_macro_single(m, out_dir / "macro_metrics.png", title=args.title)
        plot_per_image_histograms(m, out_dir / "per_image_hist.png")
        plot_scatter_obj_vs_triplet(m, out_dir / "coverage_vs_recall.png")
        figures = [
            str(out_dir / "macro_metrics.png"),
            str(out_dir / "per_image_hist.png"),
            str(out_dir / "coverage_vs_recall.png"),
        ]
        if "macro_object_precision" in m:
            plot_extended_macro(m, out_dir / "extended_macro.png", title=args.title)
            figures.append(str(out_dir / "extended_macro.png"))
            plot_per_image_precision_hist(m, out_dir / "per_image_object_precision_hist.png")
            figures.append(str(out_dir / "per_image_object_precision_hist.png"))
        meta = {
            "inputs": [str(p) for p in paths],
            "figures": figures,
            "macro": {k: m[k] for k in ("macro_object_recall", "macro_triplet_recall", "macro_mean_iou_matched") if k in m},
        }
        if "macro_object_precision" in m:
            meta["macro_extended"] = {
                k: m[k]
                for k in (
                    "macro_object_precision",
                    "macro_f1_object",
                    "parse_success_rate",
                    "n_parse_errors",
                    "mean_gt_objects",
                    "mean_pred_objects",
                    "mean_gt_relationships",
                    "mean_pred_relationships",
                )
                if k in m
            }
    else:
        plot_macro_compare(loaded, labels, out_dir / "macro_metrics.png")
        # Histograms / scatter for first run only (most recent primary)
        plot_per_image_histograms(loaded[0], out_dir / "per_image_hist.png")
        plot_scatter_obj_vs_triplet(loaded[0], out_dir / "coverage_vs_recall.png")
        meta = {
            "inputs": [str(p) for p in paths],
            "run_labels": labels,
            "figures": [
                str(out_dir / "macro_metrics.png"),
                str(out_dir / "per_image_hist.png"),
                str(out_dir / "coverage_vs_recall.png"),
            ],
            "macro_by_run": [
                {
                    "label": lab,
                    "macro_object_recall": x["macro_object_recall"],
                    "macro_triplet_recall": x["macro_triplet_recall"],
                    "macro_mean_iou_matched": x["macro_mean_iou_matched"],
                }
                for lab, x in zip(labels, loaded)
            ],
        }

    (out_dir / "viz_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
