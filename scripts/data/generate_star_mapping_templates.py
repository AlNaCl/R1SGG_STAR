import argparse
import json
from pathlib import Path


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
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = json.load(open(args.star_dict, "r", encoding="utf-8"))
    star_objs = sorted(list(d["object_count"].keys()))
    star_preds = sorted(list(d["predicate_count"].keys()))

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
