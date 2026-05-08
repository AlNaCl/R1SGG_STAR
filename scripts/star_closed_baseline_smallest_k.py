#!/usr/bin/env python3
"""
Batch inference on the K smallest-area images of a STAR HF split (reduces OOM risk without resizing).

Writes:
  preds.json       — structured predictions + meta (for eval_star_closed_predictions.py)
  raw_generations/ — model text per image_id
  images/          — visualization (same as star_closed_infer_vis)

Example:
  CUDA_VISIBLE_DEVICES=0 python scripts/star_closed_baseline_smallest_k.py \\
    --model_path /path/to/sft_checkpoint \\
    --dataset_path /root/autodl-tmp/STAR/r1sgg_data/star_r1sgg_hf_closed \\
    --split val \\
    --top_k_smallest 30
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qwen_vl_utils import process_vision_info

# Load star_closed_infer_vis by path (script dir is not a Python package).
_vis_path = PROJECT_ROOT / "scripts" / "star_closed_infer_vis.py"
_spec = importlib.util.spec_from_file_location("star_closed_infer_vis", _vis_path)
_vis = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_vis)

denorm_box_to_pixels = _vis.denorm_box_to_pixels
draw_predictions = _vis.draw_predictions
extract_answer_content = _vis.extract_answer_content
parse_scene_graph_json = _vis.parse_scene_graph_json
replace_answer_format = _vis.replace_answer_format
resolve_processor_path = _vis.resolve_processor_path

DEFAULT_OUT_PARENT = "/root/autodl-tmp/STAR/r1sgg_data/eval_visualizations"


def maybe_downscale_for_inference(
    image: Image.Image, max_pixels: int
) -> tuple[Image.Image, int, int, float, float]:
    """
    If max_pixels > 0 and image area exceeds it, resize (Qwen-friendly 28-multiple).
    Returns (image_for_model, orig_w, orig_h, scale_x, scale_y) where scale maps
    infer-pixel coords -> original-pixel coords: x_orig = x_infer * scale_x.
    """
    ow, oh = image.size
    if max_pixels is None or max_pixels <= 0:
        return image, ow, oh, 1.0, 1.0
    area = ow * oh
    if area <= max_pixels:
        return image, ow, oh, 1.0, 1.0
    scale = math.sqrt(float(max_pixels) / float(area))
    nw = max(28, int(ow * scale))
    nh = max(28, int(oh * scale))
    nw = max(28, (nw // 28) * 28)
    nh = max(28, (nh // 28) * 28)
    resample = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
    out = image.resize((nw, nh), resample=resample)
    sx = ow / float(out.width)
    sy = oh / float(out.height)
    return out, ow, oh, sx, sy


def scale_boxes_to_original(objs_px: list[dict], sx: float, sy: float) -> list[dict]:
    if sx == 1.0 and sy == 1.0:
        return objs_px
    out = []
    for o in objs_px:
        bb = o.get("bbox_xyxy")
        if not bb or len(bb) < 4:
            continue
        x1, y1, x2, y2 = bb
        out.append(
            {
                "id": o["id"],
                "bbox_xyxy": [
                    int(round(x1 * sx)),
                    int(round(y1 * sy)),
                    int(round(x2 * sx)),
                    int(round(y2 * sy)),
                ],
            }
        )
    return out


def objects_to_pixel_xyxy(objects: list[dict], iw: int, ih: int) -> list[dict]:
    """Convert model objects (bbox possibly 0–1000) to pixel xyxy + keep id."""
    out = []
    for o in objects:
        bbox = o.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = denorm_box_to_pixels(bbox, iw, ih)
        out.append({"id": str(o.get("id", "")), "bbox_xyxy": [x1, y1, x2, y2]})
    return out


def select_smallest_k_indices(ds, k: int) -> tuple[list[int], list[int], list[int]]:
    rows = []
    for i in range(len(ds)):
        w, h = int(ds[i]["width"]), int(ds[i]["height"])
        rows.append((w * h, i))
    rows.sort(key=lambda x: x[0])
    rows = rows[: min(k, len(rows))]
    areas = [a for a, _ in rows]
    indices = [i for _, i in rows]
    image_ids = [int(ds[i]["image_id"]) for i in indices]
    return indices, areas, image_ids


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--processor_path", type=str, default=None)
    ap.add_argument("--dataset_path", type=str, required=True)
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--top_k_smallest", type=int, required=True, help="Run inference only on K smallest-area images (by width*height).")
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument(
        "--output_parent",
        type=str,
        default=os.environ.get("STAR_VIS_OUTPUT_PARENT", DEFAULT_OUT_PARENT),
    )
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=4096,
        help="Hard cap on new tokens. Very large values (e.g. 8192) with big images can take hours if EOS is late; increase only if JSON truncates.",
    )
    ap.add_argument(
        "--max_infer_pixels",
        type=int,
        default=4_500_000,
        help="If >0, downscale each image before vision encode (baseline only). Pred boxes are mapped back to original pixels for eval/vis. Default ~4.5MP to reduce OOM on portrait STAR images. Use 0 to disable.",
    )
    ap.add_argument("--attn_implementation", type=str, default="sdpa")
    args = ap.parse_args()

    if args.output_dir:
        out_root = Path(args.output_dir)
    else:
        Path(args.output_parent).mkdir(parents=True, exist_ok=True)
        out_root = Path(args.output_parent) / f"baseline_smallest_{args.top_k_smallest}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    full = load_from_disk(args.dataset_path)
    ds = full[args.split]
    indices, areas, image_ids_sel = select_smallest_k_indices(ds, args.top_k_smallest)

    processor_src = resolve_processor_path(args.model_path, args.processor_path)
    print(f"[baseline] processor_path={processor_src}")
    print(f"[baseline] output_dir={out_root.resolve()}")
    print(f"[baseline] split={args.split} n={len(ds)} using_smallest_k={len(indices)} areas[:5]={areas[:5]}")

    processor = AutoProcessor.from_pretrained(processor_src, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    model.eval()

    raw_dir = out_root / "raw_generations"
    vis_dir = out_root / "images"
    raw_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    predictions: dict = {}
    meta = {
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "split": args.split,
        "selection": "smallest_k_by_area",
        "k": len(indices),
        "model_path": str(Path(args.model_path).resolve()),
        "areas": areas,
        "dataset_indices": indices,
        "image_ids": image_ids_sel,
        "max_infer_pixels": args.max_infer_pixels,
    }

    for j in tqdm(range(len(indices)), desc="infer"):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        idx = indices[j]
        sample = ds[idx]
        image_obj = sample["image"]
        if hasattr(image_obj, "convert"):
            image_orig = image_obj.convert("RGB")
        else:
            image_orig = Image.open(image_obj).convert("RGB")
        orig_w, orig_h = image_orig.size
        image, ow, oh, sx, sy = maybe_downscale_for_inference(image_orig, args.max_infer_pixels)
        iw, ih = image.size
        im_id_dbg = sample.get("image_id")
        area_dbg = ow * oh
        if (iw, ih) != (ow, oh):
            print(
                f"[baseline] ({j + 1}/{len(indices)}) image_id={im_id_dbg} infer={iw}x{ih} "
                f"(orig {ow}x{oh} area={area_dbg}) — encode+generate…",
                flush=True,
            )
        else:
            print(
                f"[baseline] ({j + 1}/{len(indices)}) image_id={im_id_dbg} {iw}x{ih} area={area_dbg} "
                f"— encode+generate starting…",
                flush=True,
            )
        t0 = time.perf_counter()
        prompt = replace_answer_format(sample["prompt_close"])
        prompt = prompt.replace(f"of size ({iw} x {ih}) ", "")

        messages = [
            {"role": "system", "content": "You are a helpful and multimodal AI assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(messages)
        inputs = processor(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt")
        inputs = inputs.to(model.device, dtype=torch.bfloat16)

        tok = getattr(processor, "tokenizer", None)
        gen_kwargs: dict = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 50,
        }
        if tok is not None:
            if getattr(tok, "eos_token_id", None) is not None:
                gen_kwargs["eos_token_id"] = tok.eos_token_id
            if getattr(tok, "pad_token_id", None) is not None:
                gen_kwargs["pad_token_id"] = tok.pad_token_id
        # Slight penalty reduces runaway repetition of object lines before EOS.
        gen_kwargs["repetition_penalty"] = 1.05

        im_id = sample.get("image_id")
        stem = str(im_id)
        decoded = ""
        oom_msg = None
        try:
            out_ids = model.generate(**inputs, **gen_kwargs)
            gen_only = out_ids[:, inputs["input_ids"].shape[-1] :]
            decoded = processor.batch_decode(gen_only, skip_special_tokens=True)[0]
        except torch.cuda.OutOfMemoryError as e:
            oom_msg = repr(e)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[baseline] image_id={im_id_dbg} CUDA OOM, skipping: {oom_msg[:200]}", flush=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        print(f"[baseline] image_id={im_id_dbg} done in {elapsed:.1f}s", flush=True)

        (raw_dir / f"{stem}.txt").write_text(decoded if decoded else f"[CUDA OOM]\n{oom_msg or ''}")

        parse_error = oom_msg
        objs_raw: list = []
        rels_raw: list = []
        objs_px: list = []
        if oom_msg is None:
            try:
                payload = extract_answer_content(decoded)
                sg = parse_scene_graph_json(payload)
                objs_raw = sg.get("objects") or []
                rels_raw = sg.get("relationships") or []
                objs_px = objects_to_pixel_xyxy(objs_raw, iw, ih)
                objs_px = scale_boxes_to_original(objs_px, sx, sy)
                parse_error = None
            except Exception as e:
                parse_error = repr(e)

        predictions[stem] = {
            "image_id": im_id,
            "width": orig_w,
            "height": orig_h,
            "area": orig_w * orig_h,
            "infer_width": iw,
            "infer_height": ih,
            "infer_scale_xy": [sx, sy],
            "objects_raw": objs_raw,
            "relationships_raw": rels_raw,
            "objects_pixel_xyxy": objs_px,
            "raw_text": decoded,
            "parse_error": parse_error,
        }

        if parse_error is None:
            # objs_raw bbox is relative to the image passed to the model (`image`, infer size).
            draw_predictions(image, objs_raw, vis_dir / f"{stem}.jpg")
        else:
            err_body = decoded if decoded else ""
            (raw_dir / f"{stem}.error").write_text(f"{err_body}\n\n{parse_error}", encoding="utf-8")

    payload_out = {"meta": meta, "predictions": predictions}
    (out_root / "preds.json").write_text(json.dumps(payload_out, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Done. preds.json -> {out_root / 'preds.json'}")
    print(f"Next: python scripts/eval_star_closed_predictions.py --dataset_path {args.dataset_path} --split {args.split} --preds_json {out_root / 'preds.json'}")


if __name__ == "__main__":
    main()
