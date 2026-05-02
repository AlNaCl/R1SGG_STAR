#!/usr/bin/env python3
"""
Attach STAR RGB images to jsonl rows and save a HuggingFace DatasetDict for GRPO/SFT.

Default layout (official STAR release):
  STAR-object/train/trainimg正确/{image_id:04d}.png
  STAR-object/val/valimg正确/{image_id:04d}.png
  STAR-object/test/testimg正确/{image_id:04d}.png
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from datasets import Dataset, DatasetDict, Value


def _default_image_roots(star_object_root: Path) -> dict[str, Path]:
    return {
        "train": star_object_root / "train" / "trainimg正确",
        "val": star_object_root / "val" / "valimg正确",
        "test": star_object_root / "test" / "testimg正确",
    }


def _build_prompt_close(star_dict_path: Path) -> str:
    # Load prompt_gallery directly (avoids importing open_r1.trainer which pulls trl).
    pg_path = _REPO_ROOT / "open_r1/trainer/utils/prompt_gallery.py"
    spec = importlib.util.spec_from_file_location("prompt_gallery_star", pg_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {pg_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    format_prompt_close_sg = mod.format_prompt_close_sg

    d = json.loads(star_dict_path.read_text(encoding="utf-8"))
    obj_cls = sorted(d["object_count"].keys())
    pred_cls = sorted(d["predicate_count"].keys())
    return format_prompt_close_sg(obj_cls, pred_cls)


def _load_split_jsonl(
    jsonl_path: Path,
    split: str,
    image_roots: dict[str, Path],
    prompt_close: str | None,
) -> Dataset:
    rows = []
    missing = []
    moved_cross_split = 0
    image_root = image_roots[split]
    fallback_roots = [image_roots[s] for s in ("train", "val", "test") if s != split]
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            iid = int(ex["image_id"])
            img_path = image_root / f"{iid:04d}.png"
            if not img_path.is_file():
                found = None
                for root in fallback_roots:
                    candidate = root / f"{iid:04d}.png"
                    if candidate.is_file():
                        found = candidate
                        break
                if found is None:
                    missing.append(str(img_path))
                    continue
                img_path = found
                moved_cross_split += 1
            ex["image"] = str(img_path.resolve())
            if prompt_close is not None:
                ex["prompt_close"] = prompt_close
            rows.append(ex)

    if missing:
        raise FileNotFoundError(
            f"[{split}] Missing {len(missing)} image files. First few:\n"
            + "\n".join(missing[:10])
        )
    if moved_cross_split:
        print(f"[{split}] Fallback-resolved {moved_cross_split} images from other split folders.")

    ds = Dataset.from_list(rows)
    # Store absolute paths as plain strings (small Arrow; grpo Collator opens PIL on the fly).
    ds = ds.cast_column("image", Value("string"))
    if prompt_close is not None:
        ds = ds.cast_column("prompt_close", Value("string"))
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl_dir",
        required=True,
        help="Directory with train.jsonl / val.jsonl / test.jsonl",
    )
    parser.add_argument(
        "--star_object_root",
        default="/root/shared-nvme/datasets/STAR/STAR/STAR-object",
        help="Root folder containing train/val/test image subfolders",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Where to save DatasetDict (load_from_disk)",
    )
    parser.add_argument(
        "--add_prompt_close",
        action="store_true",
        help="Add prompt_close using STAR closed vocab from --star_dict",
    )
    parser.add_argument(
        "--star_dict",
        default="/root/shared-nvme/datasets/STAR/STAR_SGG_Annotation/STAR-SGG-dicts-with-attri.json",
        help="Used when --add_prompt_close",
    )
    args = parser.parse_args()

    jsonl_dir = Path(args.jsonl_dir)
    roots = _default_image_roots(Path(args.star_object_root))
    prompt_close = None
    if args.add_prompt_close:
        prompt_close = _build_prompt_close(Path(args.star_dict))

    splits = {}
    for split in ("train", "val", "test"):
        p = jsonl_dir / f"{split}.jsonl"
        if not p.exists():
            continue
        splits[split] = _load_split_jsonl(p, split, roots, prompt_close)

    if not splits:
        raise FileNotFoundError(f"No train/val/test jsonl under {jsonl_dir}")

    dsd = DatasetDict(splits)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dsd.save_to_disk(str(out))

    meta = {
        "jsonl_dir": str(jsonl_dir.resolve()),
        "star_object_root": str(Path(args.star_object_root).resolve()),
        "splits": {k: len(v) for k, v in splits.items()},
        "add_prompt_close": bool(prompt_close),
        "star_dict": str(Path(args.star_dict).resolve()) if args.add_prompt_close else None,
    }
    (out / "build_star_hf_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Saved DatasetDict to:", out)
    print("Meta:", json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
