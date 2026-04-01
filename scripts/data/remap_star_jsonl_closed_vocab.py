import argparse
import json
from pathlib import Path


def load_mapping(path: str):
    data = json.load(open(path, "r", encoding="utf-8"))
    return data["mapping"] if "mapping" in data else data


def normalize(name: str):
    if not isinstance(name, str):
        return ""
    return name.strip()


def remap_split(src_file: Path, dst_file: Path, obj_map, pred_map):
    stats = {
        "images": 0,
        "obj_total": 0,
        "obj_kept": 0,
        "obj_dropped": 0,
        "rel_total": 0,
        "rel_kept": 0,
        "rel_dropped": 0,
    }

    with open(src_file, "r", encoding="utf-8") as fin, open(dst_file, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            ex = json.loads(line)
            objects = ex["objects"]
            rels = ex["relationships"]
            if isinstance(objects, str):
                objects = json.loads(objects)
            if isinstance(rels, str):
                rels = json.loads(rels)

            id_name = {}
            new_objects = []
            for o in objects:
                stats["obj_total"] += 1
                src_name = normalize(o.get("name", ""))
                dst_name = obj_map.get(src_name, "__UNK__")
                if dst_name == "__UNK__":
                    stats["obj_dropped"] += 1
                    continue
                new_o = dict(o)
                new_o["name"] = dst_name
                # keep stable object id suffix while updating class prefix
                old_id = str(o.get("id", ""))
                suffix = old_id.split(".")[-1] if "." in old_id else old_id
                new_id = f"{dst_name}.{suffix}"
                new_o["id"] = new_id
                id_name[old_id] = new_id
                id_name[str(new_o.get("object_id", ""))] = new_id
                new_objects.append(new_o)
                stats["obj_kept"] += 1

            new_rels = []
            for r in rels:
                stats["rel_total"] += 1
                src_pred = normalize(r.get("predicate", ""))
                dst_pred = pred_map.get(src_pred, "__UNK__")
                if dst_pred == "__UNK__":
                    stats["rel_dropped"] += 1
                    continue
                s_old = str(r.get("subject", ""))
                o_old = str(r.get("object", ""))
                s_new = id_name.get(s_old, s_old)
                o_new = id_name.get(o_old, o_old)
                # keep only relations whose endpoints exist after object remap
                valid_ids = {obj["id"] for obj in new_objects}
                if s_new not in valid_ids or o_new not in valid_ids:
                    stats["rel_dropped"] += 1
                    continue
                new_rels.append({"subject": s_new, "predicate": dst_pred, "object": o_new})
                stats["rel_kept"] += 1

            ex["objects"] = json.dumps(new_objects, ensure_ascii=False)
            ex["relationships"] = json.dumps(new_rels, ensure_ascii=False)
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
            stats["images"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="source jsonl dir (train/val/test)")
    parser.add_argument("--output_dir", required=True, help="dest jsonl dir")
    parser.add_argument("--obj_map", required=True)
    parser.add_argument("--pred_map", required=True)
    args = parser.parse_args()

    obj_map = load_mapping(args.obj_map)
    pred_map = load_mapping(args.pred_map)
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for split in ["train", "val", "test"]:
        src = in_dir / f"{split}.jsonl"
        if not src.exists():
            continue
        dst = out_dir / f"{split}.jsonl"
        all_stats[split] = remap_split(src, dst, obj_map, pred_map)
        print(split, all_stats[split])

    stats_file = out_dir / "remap_stats.json"
    json.dump(all_stats, open(stats_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Wrote stats: {stats_file}")


if __name__ == "__main__":
    main()
