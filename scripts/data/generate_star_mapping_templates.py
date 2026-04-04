import argparse
import json
from pathlib import Path

import h5py


def _load_from_dict(star_dict_path: str):
    d = json.load(open(star_dict_path, "r", encoding="utf-8"))
    star_objs = sorted(list(d["object_count"].keys()))
    star_preds = sorted(list(d["predicate_count"].keys()))
    return star_objs, star_preds


def _load_from_h5(star_h5_path: str, star_dict_path: str | None = None):
    with h5py.File(star_h5_path, "r") as f:
        obj_ids = sorted({int(x[0]) for x in f["labels"][:] if int(x[0]) > 0})
        pred_ids = sorted({int(x[0]) for x in f["predicates"][:] if int(x[0]) > 0})

    if not star_dict_path:
        # Fallback: keep id-based labels if no dictionary is provided.
        star_objs = [f"obj_id_{i}" for i in obj_ids]
        star_preds = [f"pred_id_{i}" for i in pred_ids]
        return star_objs, star_preds

    d = json.load(open(star_dict_path, "r", encoding="utf-8"))
    idx_to_label = d.get("idx_to_label", {})
    idx_to_pred = d.get("idx_to_predicate", {})
    star_objs = [idx_to_label[str(i)] for i in obj_ids if str(i) in idx_to_label]
    star_preds = [idx_to_pred[str(i)] for i in pred_ids if str(i) in idx_to_pred]
    return sorted(set(star_objs)), sorted(set(star_preds))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--star_dict",
        default="/mnt/dataY/ly25/datasets/STAR/STAR_SGG_Annotation/STAR-SGG-dicts-with-attri.json",
    )
    parser.add_argument(
        "--output_dir",
        default="/mnt/dataY/ly25/R1-SGG/datasets/star_closed_vocab_maps",
    )
    parser.add_argument(
        "--star_h5",
        default=None,
        help="Optional STAR h5 annotation file. If provided, labels/predicates are collected from h5 ids.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.star_h5:
        star_objs, star_preds = _load_from_h5(args.star_h5, args.star_dict)
    else:
        star_objs, star_preds = _load_from_dict(args.star_dict)

    obj_map = {k: k for k in star_objs}
    pred_map = {k: k for k in star_preds}

    obj_meta = {
        "_note": "Map STAR object label -> target closed-vocab label. Use __UNK__ for unknown.",
        "_total_star_labels": len(star_objs),
        "mapping": obj_map,
    }
    pred_meta = {
        "_note": "Map STAR predicate label -> target closed-vocab predicate. Use __UNK__ for unknown.",
        "_total_star_labels": len(star_preds),
        "mapping": pred_map,
    }

    obj_file = out_dir / "obj_star2r1.json"
    pred_file = out_dir / "pred_star2r1.json"
    json.dump(obj_meta, open(obj_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(pred_meta, open(pred_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"Wrote: {obj_file}")
    print(f"Wrote: {pred_file}")
    print("Current files are identity mappings. Edit mapping values before retraining.")


if __name__ == "__main__":
    main()
