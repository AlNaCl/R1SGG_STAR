import argparse
import json
from collections import Counter
from pathlib import Path


def load_mapping(path: str):
    data = json.load(open(path, "r", encoding="utf-8"))
    return data["mapping"] if "mapping" in data else data


def norm_name(name: str):
    if not isinstance(name, str):
        return ""
    return name.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_dir", required=True, help="directory containing train/val/test jsonl")
    parser.add_argument("--obj_map", required=True, help="obj_star2r1.json path")
    parser.add_argument("--pred_map", required=True, help="pred_star2r1.json path")
    args = parser.parse_args()

    obj_map = load_mapping(args.obj_map)
    pred_map = load_mapping(args.pred_map)
    jsonl_dir = Path(args.jsonl_dir)

    files = [p for p in [jsonl_dir / "train.jsonl", jsonl_dir / "val.jsonl", jsonl_dir / "test.jsonl"] if p.exists()]
    if not files:
        raise FileNotFoundError(f"No jsonl found in {jsonl_dir}")

    obj_total = 0
    rel_total = 0
    obj_unk = 0
    pred_unk = 0
    obj_missing_key = Counter()
    pred_missing_key = Counter()

    for f in files:
        with open(f, "r", encoding="utf-8") as fin:
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

                for o in objects:
                    src = norm_name(o.get("name", ""))
                    obj_total += 1
                    if src not in obj_map:
                        obj_missing_key[src] += 1
                        continue
                    if obj_map[src] == "__UNK__":
                        obj_unk += 1

                for r in rels:
                    src = norm_name(r.get("predicate", ""))
                    rel_total += 1
                    if src not in pred_map:
                        pred_missing_key[src] += 1
                        continue
                    if pred_map[src] == "__UNK__":
                        pred_unk += 1

    print("=== Closed-Vocab Validation ===")
    print(f"obj_total={obj_total}, obj_unk={obj_unk}, obj_unk_ratio={obj_unk/max(obj_total,1):.4f}")
    print(f"rel_total={rel_total}, pred_unk={pred_unk}, pred_unk_ratio={pred_unk/max(rel_total,1):.4f}")
    print(f"obj_missing_in_map={len(obj_missing_key)}, pred_missing_in_map={len(pred_missing_key)}")
    if obj_missing_key:
        print("top_obj_missing:", obj_missing_key.most_common(20))
    if pred_missing_key:
        print("top_pred_missing:", pred_missing_key.most_common(20))


if __name__ == "__main__":
    main()
