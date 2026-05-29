#!/usr/bin/env python3
"""
Export relationship triplets from preds.json (star_closed_baseline_smallest_k output) to CSV.

Columns: image_id, pred_key, subject, predicate, object, row_index_in_image
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_json", type=str, required=True)
    ap.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Output path (default: same dir as preds_json / triplets.csv)",
    )
    args = ap.parse_args()

    preds_path = Path(args.preds_json).resolve()
    data = json.loads(preds_path.read_text(encoding="utf-8"))

    out_path = Path(args.out_csv).resolve() if args.out_csv else preds_path.parent / "triplets.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "pred_key", "subject", "predicate", "object", "index_in_image"])
        for pred_key, p in sorted(data.get("predictions", {}).items(), key=lambda kv: int(kv[1].get("image_id") or kv[0])):
            im_id = p.get("image_id")
            if im_id is None:
                try:
                    im_id = int(pred_key)
                except ValueError:
                    im_id = pred_key
            rels = p.get("relationships_raw") or []
            for idx, rel in enumerate(rels):
                subj = str(rel.get("subject", "")).strip()
                pred = str(rel.get("predicate", "")).strip()
                obj = str(rel.get("object", "")).strip()
                w.writerow([im_id, pred_key, subj, pred, obj, idx])
                rows_written += 1

    print(f"Wrote {rows_written} triplets -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
