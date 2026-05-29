#!/usr/bin/env python3
"""
Re-draw images/ from an existing preds.json (no model inference).

Use when visualization code was updated (e.g. relationship edges) or figures were lost.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from datasets import load_from_disk
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_vis_path = PROJECT_ROOT / "scripts" / "star_closed_infer_vis.py"
_spec = importlib.util.spec_from_file_location("star_closed_infer_vis", _vis_path)
_vis = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_vis)
draw_predictions = _vis.draw_predictions

_bl_path = PROJECT_ROOT / "scripts" / "star_closed_baseline_smallest_k.py"
_spec2 = importlib.util.spec_from_file_location("star_closed_baseline_smallest_k", _bl_path)
_bl = importlib.util.module_from_spec(_spec2)
assert _spec2.loader is not None
_spec2.loader.exec_module(_bl)
maybe_downscale_for_inference = _bl.maybe_downscale_for_inference


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_json", type=str, required=True)
    ap.add_argument("--dataset_path", type=str, required=True)
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory for images/ (default: preds_json parent / images_relvis)",
    )
    args = ap.parse_args()

    preds_path = Path(args.preds_json).resolve()
    data = json.loads(preds_path.read_text(encoding="utf-8"))
    meta = data.get("meta") or {}
    max_infer = int(meta.get("max_infer_pixels", 4_500_000))

    out_root = Path(args.out_dir).resolve() if args.out_dir else preds_path.parent / "images_relvis"
    vis_dir = out_root / "images"
    vis_dir.mkdir(parents=True, exist_ok=True)

    ds = load_from_disk(args.dataset_path)[args.split]
    id_to_idx = {int(ds[i]["image_id"]): i for i in range(len(ds))}

    n_drawn = 0
    n_rels = 0
    for stem, p in data["predictions"].items():
        raw_id = p.get("image_id")
        im_id = int(raw_id) if raw_id is not None else int(stem)
        if im_id not in id_to_idx:
            continue
        sample = ds[id_to_idx[im_id]]
        image_obj = sample["image"]
        if hasattr(image_obj, "convert"):
            image_orig = image_obj.convert("RGB")
        else:
            image_orig = Image.open(image_obj).convert("RGB")
        image, *_rest = maybe_downscale_for_inference(image_orig, max_infer)
        objs = p.get("objects_raw") or []
        rels = p.get("relationships_raw") or []
        if p.get("parse_error"):
            continue
        draw_predictions(image, objs, vis_dir / f"{stem}.jpg", relationships=rels if rels else None)
        n_drawn += 1
        n_rels += len(rels)

    print(f"Wrote {n_drawn} images under {vis_dir} (total relationship rows drawn from preds: {n_rels})")


if __name__ == "__main__":
    main()
