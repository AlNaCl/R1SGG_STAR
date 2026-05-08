#!/usr/bin/env python3
"""
Run Qwen2-VL (e.g. STAR closed SFT checkpoint) on local HF dataset rows and save images with predicted boxes + labels.

Inference uses only image + prompt_close (no GT-driven tiling/subgraph from training collator).
Images are not resized here to preserve full spatial fidelity; extremely large inputs require enough VRAM (or run elsewhere).

Example (auto-created run folder under default parent):
  CUDA_VISIBLE_DEVICES=0 python scripts/star_closed_infer_vis.py \\
    --model_path /path/to/checkpoint \\
    --dataset_path /root/autodl-tmp/STAR/r1sgg_data/star_r1sgg_hf_closed \\
    --split val \\
    --max_samples 20

Override parent only (creates parent/run_YYYYMMDD_HHMMSS):
  --output_parent /path/to/eval_visualizations

Pin exact output directory:
  --output_dir /path/to/my_vis_run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_VIS_PARENT = "/root/autodl-tmp/STAR/r1sgg_data/eval_visualizations"

import torch
from datasets import load_from_disk
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from tqdm import tqdm

# Repo imports when run from R1SGG_STAR root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qwen_vl_utils import process_vision_info


def _first_snapshot_with_preprocessor(hub_model_dir: Path) -> str | None:
    """Pick newest snapshot under HF hub cache that contains preprocessor_config.json."""
    snap_root = hub_model_dir / "snapshots"
    if not snap_root.is_dir():
        return None
    candidates = []
    for p in snap_root.iterdir():
        if p.is_dir() and (p / "preprocessor_config.json").is_file():
            candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(candidates[0])


def resolve_processor_path(model_path: str, explicit: str | None) -> str:
    """
    SFT checkpoints saved with --save_only_model often omit preprocessor_config.json.
    In that case load Processor from base model dir / HF cache snapshot; weights still load from model_path.
    """
    if explicit:
        return explicit
    mp = Path(model_path)
    if (mp / "preprocessor_config.json").is_file():
        return str(mp.resolve())

    env_p = os.environ.get("STAR_INFER_PROCESSOR_PATH")
    if env_p and Path(env_p).is_dir() and (Path(env_p) / "preprocessor_config.json").is_file():
        return str(Path(env_p).resolve())

    hub_cache = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct"
    resolved = _first_snapshot_with_preprocessor(hub_cache)
    if resolved:
        return resolved

    # Last resort: repo id (may hit network)
    return "Qwen/Qwen2-VL-2B-Instruct"


def replace_answer_format(item: str) -> str:
    return item.replace("<answer>", "```json").replace("</answer>", "```")


def extract_answer_content(text: str) -> str:
    text = text.replace("```", " ").replace("json", " ").strip()
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"<answer>(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # bare JSON object
    m = re.search(r"\{[\s\S]*\}", text)
    return m.group(0).strip() if m else text


# When generation hits max_new_tokens, JSON is often truncated mid-string. Extract complete entries.
_SG_OBJECT_RE = re.compile(
    r'\{\s*"id"\s*:\s*"([^"]*)"\s*,\s*"bbox"\s*:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]\s*\}',
    re.MULTILINE,
)
_SG_REL_RE = re.compile(
    r'\{\s*"subject"\s*:\s*"([^"]*)"\s*,\s*"predicate"\s*:\s*"([^"]*)"\s*,\s*"object"\s*:\s*"([^"]*)"\s*\}',
    re.MULTILINE,
)


def _parse_scene_graph_lenient_regex(text: str) -> dict:
    """Recover objects/relationships from truncated or noisy model output.

    One image may contain many instances of the same category (e.g. car.1, car.2, …); we keep
    all distinct (id, bbox) pairs. We only skip exact duplicate rows (same id and same bbox),
    e.g. when the model repeats the same JSON object line.
    """
    objects: list[dict] = []
    seen_object_keys: set[tuple[str, tuple[float, float, float, float]]] = set()
    for m in _SG_OBJECT_RE.finditer(text):
        oid = m.group(1)
        bbox = tuple(float(m.group(i)) for i in range(2, 6))
        key = (oid, bbox)
        if key in seen_object_keys:
            continue
        seen_object_keys.add(key)
        objects.append({"id": oid, "bbox": list(bbox)})

    relationships: list[dict] = []
    seen_rel_keys: set[tuple[str, str, str]] = set()
    for m in _SG_REL_RE.finditer(text):
        subj, pred, obj = m.group(1), m.group(2), m.group(3)
        rk = (subj, pred, obj)
        if rk in seen_rel_keys:
            continue
        seen_rel_keys.add(rk)
        relationships.append({"subject": subj, "predicate": pred, "object": obj})
    return {"objects": objects, "relationships": relationships}


def parse_scene_graph_json(raw: str) -> dict:
    s = raw.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        s2 = re.sub(r"^```\w*\s*", "", s)
        s2 = re.sub(r"\s*```$", "", s2)
        try:
            return json.loads(s2)
        except json.JSONDecodeError:
            recovered = _parse_scene_graph_lenient_regex(s2 if s2 else s)
            if recovered["objects"] or recovered["relationships"]:
                return recovered
            raise


def denorm_box_to_pixels(box, iw: int, ih: int) -> list[int]:
    """Training targets use 0..1000; prompt text may say pixels — heuristics below."""
    x1, y1, x2, y2 = [float(x) for x in box[:4]]
    if max(x1, y1, x2, y2) <= 1000.0:
        x1, y1, x2, y2 = (
            x1 / 1000.0 * iw,
            y1 / 1000.0 * ih,
            x2 / 1000.0 * iw,
            y2 / 1000.0 * ih,
        )
    else:
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    # Model may output x2<x1 or confuse xyxy with xywh; PIL requires x0<=x1, y0<=y1.
    xa, xb = sorted((x1, x2))
    ya, yb = sorted((y1, y2))
    return [int(round(xa)), int(round(ya)), int(round(xb)), int(round(yb))]


def draw_predictions(image: Image.Image, objects: list[dict], out_path: Path) -> None:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for obj in objects:
        oid = str(obj.get("id", ""))
        bbox = obj.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = denorm_box_to_pixels(bbox, img.width, img.height)
        x1 = max(0, min(x1, img.width - 1))
        y1 = max(0, min(y1, img.height - 1))
        x2 = max(0, min(x2, img.width - 1))
        y2 = max(0, min(y2, img.height - 1))
        if x2 <= x1:
            x2 = min(x1 + 1, img.width - 1)
        if y2 <= y1:
            y2 = min(y1 + 1, img.height - 1)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        label = oid[:80]
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        bg = [x1, max(0, y1 - th - 2), x1 + tw + 4, y1]
        draw.rectangle(bg, fill="red")
        draw.text((x1 + 2, bg[1] + 1), label, fill="white", font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True, help="Fine-tuned checkpoint dir or HF id.")
    ap.add_argument(
        "--processor_path",
        type=str,
        default=None,
        help="Directory with preprocessor_config.json + tokenizer (default: same as model_path if complete, else ~/.cache/.../Qwen2-VL-2B-Instruct snapshot or STAR_INFER_PROCESSOR_PATH).",
    )
    ap.add_argument("--dataset_path", type=str, required=True, help="HF DatasetDict root (save_to_disk).")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument(
        "--output_parent",
        type=str,
        default=os.environ.get("STAR_VIS_OUTPUT_PARENT", DEFAULT_VIS_PARENT),
        help=f"Base directory for runs when --output_dir is omitted (default: {DEFAULT_VIS_PARENT}; env STAR_VIS_OUTPUT_PARENT overrides).",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Exact output root (images/, raw_generations/). If omitted, uses OUTPUT_PARENT/run_YYYYMMDD_HHMMSS.",
    )
    ap.add_argument("--max_samples", type=int, default=50)
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=4096,
        help="Max new tokens for generate(); huge values + big images can run for hours if EOS is late (default 4096).",
    )
    ap.add_argument("--attn_implementation", type=str, default="sdpa")
    args = ap.parse_args()

    if args.output_dir:
        out_root = Path(args.output_dir)
    else:
        Path(args.output_parent).mkdir(parents=True, exist_ok=True)
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out_root = Path(args.output_parent) / run_name

    ds = load_from_disk(args.dataset_path)[args.split]
    n = min(len(ds), args.max_samples)

    processor_src = resolve_processor_path(args.model_path, args.processor_path)
    print(f"[star_closed_infer_vis] processor_path={processor_src}")
    processor = AutoProcessor.from_pretrained(processor_src, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    model.eval()

    print(f"[star_closed_infer_vis] output_dir={out_root.resolve()}")

    raw_dir = out_root / "raw_generations"
    vis_dir = out_root / "images"
    raw_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(n), desc="infer"):
        sample = ds[i]
        image_obj = sample["image"]
        if hasattr(image_obj, "convert"):
            image = image_obj.convert("RGB")
        else:
            image = Image.open(image_obj).convert("RGB")
        iw, ih = image.size
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
            "repetition_penalty": 1.05,
        }
        if tok is not None:
            if getattr(tok, "eos_token_id", None) is not None:
                gen_kwargs["eos_token_id"] = tok.eos_token_id
            if getattr(tok, "pad_token_id", None) is not None:
                gen_kwargs["pad_token_id"] = tok.pad_token_id
        out_ids = model.generate(**inputs, **gen_kwargs)
        gen_only = out_ids[:, inputs["input_ids"].shape[-1] :]
        decoded = processor.batch_decode(gen_only, skip_special_tokens=True)[0]

        stem = f"{sample.get('image_id', i)}"
        (raw_dir / f"{stem}.txt").write_text(decoded, encoding="utf-8")

        try:
            payload = extract_answer_content(decoded)
            sg = parse_scene_graph_json(payload)
            objs = sg.get("objects") or []
        except Exception as e:
            (raw_dir / f"{stem}.error").write_text(f"{decoded}\n\n{e!r}", encoding="utf-8")
            objs = []

        draw_predictions(image, objs, vis_dir / f"{stem}.jpg")

    print(f"Done. Visualizations: {vis_dir}, raw text: {raw_dir}")


if __name__ == "__main__":
    main()
