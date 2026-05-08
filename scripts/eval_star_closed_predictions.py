#!/usr/bin/env python3
"""
Evaluate STAR closed-vocab predictions (preds.json from star_closed_baseline_smallest_k.py)
against ground truth in a HF DatasetDict.

GT object bbox format (STAR): [x, y, w, h] in pixels -> converted to xyxy for IoU.
Pred bboxes: use objects_pixel_xyxy from preds.json (already in pixel xyxy).

Metrics (per-image then macro-averaged):
  - object_recall_iou: fraction of GT objects that have a matched pred with IoU >= threshold (Hungarian one-to-one).
  - triplet_recall_exact: fraction of GT triplets matched by (matched objects + exact predicate string on pred side).
  - mean_iou_matched_objects: mean IoU over matched GT-Pred object pairs (IoU >= threshold).

Example:
  python scripts/eval_star_closed_predictions.py \\
    --dataset_path /root/autodl-tmp/STAR/r1sgg_data/star_r1sgg_hf_closed \\
    --split val \\
    --preds_json /root/autodl-tmp/STAR/r1sgg_data/eval_visualizations/.../preds.json \\
    --iou_threshold 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


def iou_xyxy(a: list[float], b: list[float]) -> float:
    x_a = max(a[0], b[0])
    y_a = max(a[1], b[1])
    x_b = min(a[2], b[2])
    y_b = min(a[3], b[3])
    iw = max(0.0, x_b - x_a)
    ih = max(0.0, y_b - y_a)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def xywh_to_xyxy(b: list[float]) -> list[float]:
    x, y, w, h = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return [x, y, x + w, y + h]


def load_gt_objects_and_rels(sample: dict) -> tuple[list[dict], list[dict]]:
    objs = sample["objects"]
    rels = sample["relationships"]
    if isinstance(objs, str):
        objs = json.loads(objs)
    if isinstance(rels, str):
        rels = json.loads(rels)
    gt_objs = []
    for o in objs:
        bb = o["bbox"]
        xyxy = xywh_to_xyxy(bb)
        gt_objs.append({"id": str(o["id"]), "bbox_xyxy": xyxy})
    return gt_objs, rels


def match_objects_hungarian(
    gt_boxes_xyxy: list[list[float]],
    pred_boxes_xyxy: list[list[float]],
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    """Returns list of (gt_idx, pred_idx, iou) for pairs with IoU >= threshold."""
    n_gt = len(gt_boxes_xyxy)
    n_pred = len(pred_boxes_xyxy)
    if n_gt == 0 or n_pred == 0:
        return []

    if linear_sum_assignment is None:
        # Greedy fallback: for each GT pick best unused pred by IoU
        used_pred = set()
        pairs = []
        for i in range(n_gt):
            best_j, best_iou = -1, -1.0
            for j in range(n_pred):
                if j in used_pred:
                    continue
                iv = iou_xyxy(gt_boxes_xyxy[i], pred_boxes_xyxy[j])
                if iv > best_iou:
                    best_iou, best_j = iv, j
            if best_j >= 0 and best_iou >= iou_threshold:
                used_pred.add(best_j)
                pairs.append((i, best_j, best_iou))
        return pairs

    n = max(n_gt, n_pred)
    cost = np.ones((n, n), dtype=np.float64) * 2.0
    for i in range(n_gt):
        for j in range(n_pred):
            cost[i, j] = 1.0 - iou_xyxy(gt_boxes_xyxy[i], pred_boxes_xyxy[j])
    r, c = linear_sum_assignment(cost)
    pairs = []
    for ri, ci in zip(r, c):
        if ri < n_gt and ci < n_pred:
            iou_v = 1.0 - float(cost[ri, ci])
            if iou_v >= iou_threshold:
                pairs.append((int(ri), int(ci), iou_v))
    return pairs


def evaluate_one_image(
    gt_objs: list[dict],
    gt_rels: list[dict],
    pred_objs_px: list[dict],
    pred_rels_raw: list[dict],
    iou_threshold: float,
) -> dict:
    gt_boxes = [o["bbox_xyxy"] for o in gt_objs]
    pred_boxes = [o["bbox_xyxy"] for o in pred_objs_px]
    pred_ids = [str(o["id"]) for o in pred_objs_px]

    pairs = match_objects_hungarian(gt_boxes, pred_boxes, iou_threshold)
    gt_to_pred = {g: (p, iou) for g, p, iou in pairs}

    n_gt_obj = len(gt_objs)
    obj_recall = len(gt_to_pred) / n_gt_obj if n_gt_obj else 1.0
    mean_iou = float(np.mean([iou for _, _, iou in pairs])) if pairs else 0.0

    gt_id_to_idx = {o["id"]: i for i, o in enumerate(gt_objs)}

    hit_rel = 0
    n_gt_rel = len(gt_rels)
    for rel in gt_rels:
        sid = str(rel["subject"])
        oid = str(rel["object"])
        pred_cat = str(rel["predicate"]).strip().lower()
        if sid not in gt_id_to_idx or oid not in gt_id_to_idx:
            continue
        i_s = gt_id_to_idx[sid]
        i_o = gt_id_to_idx[oid]
        if i_s not in gt_to_pred or i_o not in gt_to_pred:
            continue
        ps, po = gt_to_pred[i_s][0], gt_to_pred[i_o][0]
        ps_id = pred_ids[ps] if ps < len(pred_ids) else ""
        po_id = pred_ids[po] if po < len(pred_ids) else ""
        found = False
        for pr in pred_rels_raw:
            if (
                str(pr.get("subject", "")).strip() == ps_id
                and str(pr.get("object", "")).strip() == po_id
                and str(pr.get("predicate", "")).strip().lower() == pred_cat
            ):
                found = True
                break
        if found:
            hit_rel += 1

    triplet_recall = hit_rel / n_gt_rel if n_gt_rel else 1.0

    return {
        "n_gt_objects": n_gt_obj,
        "n_pred_objects": len(pred_objs_px),
        "n_gt_relationships": n_gt_rel,
        "object_recall": obj_recall,
        "mean_iou_matched": mean_iou,
        "triplet_recall": triplet_recall,
        "matched_object_pairs": len(pairs),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", type=str, required=True)
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--preds_json", type=str, required=True)
    ap.add_argument("--iou_threshold", type=float, default=0.5)
    ap.add_argument("--out_metrics", type=str, default=None, help="Default: preds_json dir / metrics.json")
    args = ap.parse_args()

    data = json.loads(Path(args.preds_json).read_text(encoding="utf-8"))
    preds_map = data["predictions"]
    meta = data.get("meta", {})

    ds = load_from_disk(args.dataset_path)[args.split]
    id_to_row = {int(ds[i]["image_id"]): i for i in range(len(ds))}

    per_image = []
    for stem, pentry in preds_map.items():
        raw_id = pentry.get("image_id")
        im_id = int(raw_id) if raw_id is not None else int(stem)
        if im_id not in id_to_row:
            continue
        sample = ds[id_to_row[im_id]]
        gt_objs, gt_rels = load_gt_objects_and_rels(sample)
        pred_px = pentry.get("objects_pixel_xyxy") or []
        pred_rels = pentry.get("relationships_raw") or []
        if pentry.get("parse_error"):
            per_image.append(
                {
                    "image_id": im_id,
                    "error": "parse_error",
                    "object_recall": 0.0,
                    "triplet_recall": 0.0,
                    "mean_iou_matched": 0.0,
                }
            )
            continue
        stats = evaluate_one_image(gt_objs, gt_rels, pred_px, pred_rels, args.iou_threshold)
        stats["image_id"] = im_id
        per_image.append(stats)

    def mean_safe(key: str) -> float:
        vals = [x[key] for x in per_image if key in x and "error" not in x]
        return float(np.mean(vals)) if vals else 0.0

    summary = {
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "split": args.split,
        "preds_json": str(Path(args.preds_json).resolve()),
        "iou_threshold": args.iou_threshold,
        "n_images_evaluated": len(per_image),
        "preds_meta": meta,
        "macro_object_recall": mean_safe("object_recall"),
        "macro_triplet_recall": mean_safe("triplet_recall"),
        "macro_mean_iou_matched": mean_safe("mean_iou_matched"),
        "per_image": per_image,
    }

    if linear_sum_assignment is None:
        summary["note"] = "scipy not installed; using greedy object matching fallback."

    out_path = Path(args.out_metrics) if args.out_metrics else Path(args.preds_json).parent / "metrics.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "per_image"}, ensure_ascii=False, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
