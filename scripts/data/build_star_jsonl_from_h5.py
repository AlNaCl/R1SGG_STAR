import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

import h5py


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_split_ids_from_h5(
    h5_path: Path,
    image_ids: List[int],
    strategy: str,
    val_ratio: float,
) -> Dict[str, Set[int]]:
    with h5py.File(h5_path, "r") as f:
        split_arr = f["split"][:]

    if len(split_arr) != len(image_ids):
        raise ValueError(
            f"h5 split length ({len(split_arr)}) != image_data length ({len(image_ids)})"
        )

    split_ids = {"train": set(), "val": set(), "test": set()}
    for idx, split_code in enumerate(split_arr.tolist()):
        image_id = image_ids[idx]
        if split_code == 0:
            split_ids["train"].add(image_id)
        elif split_code == 1:
            split_ids["val"].add(image_id)
        elif split_code == 2:
            split_ids["test"].add(image_id)

    if strategy == "h5_train_val_test" and not split_ids["val"]:
        train_sorted = sorted(split_ids["train"])
        if train_sorted:
            val_n = max(1, int(len(train_sorted) * val_ratio))
            val_ids = set(train_sorted[-val_n:])
            split_ids["val"] = val_ids
            split_ids["train"] = set(train_sorted[:-val_n])

    return split_ids


def _load_split_ids_from_json(path: Path) -> Dict[str, Set[int]]:
    data = _load_json(path)
    split_ids = {}
    for key in ("train", "val", "test"):
        vals = data.get(key, [])
        split_ids[key] = set(int(x) for x in vals)
    return split_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--star_h5", required=True, help="Path to STAR-SGG-with-attri.h5")
    parser.add_argument(
        "--star_dict",
        required=True,
        help="Path to STAR-SGG-dicts-with-attri.json",
    )
    parser.add_argument(
        "--image_data_json",
        required=True,
        help="Path to STAR_image_data_v1.json",
    )
    parser.add_argument(
        "--objects_json",
        required=True,
        help="Path to STAR_objects_v1.json",
    )
    parser.add_argument(
        "--relationships_json",
        required=True,
        help="Path to STAR_relationships_v1.json",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory containing train/val/test jsonl",
    )
    parser.add_argument(
        "--split_strategy",
        choices=["h5_default", "h5_train_val_test", "from_json"],
        default="h5_default",
        help="How to create train/val/test split",
    )
    parser.add_argument(
        "--split_json",
        default=None,
        help="Optional JSON with keys train/val/test image_id list (used when split_strategy=from_json)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Only used by h5_train_val_test when h5 has no val split",
    )
    args = parser.parse_args()

    star_dict = _load_json(Path(args.star_dict))
    idx_to_label = {int(k): v for k, v in star_dict["idx_to_label"].items()}
    idx_to_predicate = {int(k): v for k, v in star_dict["idx_to_predicate"].items()}

    image_data = _load_json(Path(args.image_data_json))
    objects_data = _load_json(Path(args.objects_json))
    rels_data = _load_json(Path(args.relationships_json))

    if not (len(image_data) == len(objects_data) == len(rels_data)):
        raise ValueError("image_data / objects / relationships length mismatch")

    image_ids = [int(x["image_id"]) for x in image_data]

    if args.split_strategy == "from_json":
        if not args.split_json:
            raise ValueError("split_strategy=from_json requires --split_json")
        split_ids = _load_split_ids_from_json(Path(args.split_json))
    else:
        split_ids = _build_split_ids_from_h5(
            Path(args.star_h5), image_ids, args.split_strategy, args.val_ratio
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        "train": open(out_dir / "train.jsonl", "w", encoding="utf-8"),
        "val": open(out_dir / "val.jsonl", "w", encoding="utf-8"),
        "test": open(out_dir / "test.jsonl", "w", encoding="utf-8"),
    }
    split_stats = {
        "train": {"images": 0, "obj": 0, "rel": 0},
        "val": {"images": 0, "obj": 0, "rel": 0},
        "test": {"images": 0, "obj": 0, "rel": 0},
        "dropped_no_split": 0,
    }

    try:
        for i in range(len(image_data)):
            meta = image_data[i]
            obj_pack = objects_data[i]
            rel_pack = rels_data[i]
            image_id = int(meta["image_id"])

            split_name = None
            for name in ("train", "val", "test"):
                if image_id in split_ids[name]:
                    split_name = name
                    break
            if split_name is None:
                split_stats["dropped_no_split"] += 1
                continue

            objects = obj_pack.get("objects", [])
            rels = rel_pack.get("relationships", [])

            # Build object list and local id mapping.
            mapped_objects = []
            obj_id_map = {}
            for o in objects:
                oid = int(o.get("object_id", -1))
                if oid < 0:
                    continue
                # Prefer names from v1 json; fallback to dict via h5 id if needed.
                obj_name = str(o.get("names", "")).strip()
                if not obj_name:
                    label_idx = int(o.get("label", 0))
                    obj_name = idx_to_label.get(label_idx, "unknown")
                obj_uid = f"{obj_name}.{oid}"
                obj_id_map[oid] = obj_uid
                mapped_objects.append(
                    {
                        "id": obj_uid,
                        "object_id": oid,
                        "name": obj_name,
                        "bbox": [
                            float(o.get("x", 0.0)),
                            float(o.get("y", 0.0)),
                            float(o.get("w", 0.0)),
                            float(o.get("h", 0.0)),
                        ],
                    }
                )

            mapped_rels = []
            for r in rels:
                subj = r.get("subject", {})
                obj = r.get("object", {})
                s_oid = int(subj.get("object_id", -1))
                o_oid = int(obj.get("object_id", -1))
                if s_oid not in obj_id_map or o_oid not in obj_id_map:
                    continue

                pred = str(r.get("predicate", "")).strip()
                if not pred:
                    pidx = int(r.get("predicate_idx", 0))
                    pred = idx_to_predicate.get(pidx, "unknown")

                mapped_rels.append(
                    {
                        "subject": obj_id_map[s_oid],
                        "predicate": pred,
                        "object": obj_id_map[o_oid],
                    }
                )

            record = {
                "image_id": image_id,
                "width": int(meta.get("width", 0)),
                "height": int(meta.get("height", 0)),
                "objects": json.dumps(mapped_objects, ensure_ascii=False),
                "relationships": json.dumps(mapped_rels, ensure_ascii=False),
            }
            writers[split_name].write(json.dumps(record, ensure_ascii=False) + "\n")

            split_stats[split_name]["images"] += 1
            split_stats[split_name]["obj"] += len(mapped_objects)
            split_stats[split_name]["rel"] += len(mapped_rels)
    finally:
        for fp in writers.values():
            fp.close()

    with open(out_dir / "build_stats.json", "w", encoding="utf-8") as f:
        json.dump(split_stats, f, ensure_ascii=False, indent=2)

    print("Wrote:", out_dir / "train.jsonl")
    print("Wrote:", out_dir / "val.jsonl")
    print("Wrote:", out_dir / "test.jsonl")
    print("Wrote:", out_dir / "build_stats.json")
    print("Stats:", split_stats)


if __name__ == "__main__":
    main()
