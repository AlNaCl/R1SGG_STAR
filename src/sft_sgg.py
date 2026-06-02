
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Supervised fine-tuning script for decoder language models.

"""
import os
import json
import random
import hashlib
from tqdm import tqdm
import torch
import math
from dataclasses import dataclass, field
from pathlib import Path
import re
import glob
from typing import Any, Optional

from accelerate import Accelerator
from datasets import DatasetDict, load_dataset, load_from_disk

from transformers import (
    AutoProcessor, 
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor
)

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from qwen_vl_utils import process_vision_info
from src.rl.mask_utils import mask_labels_to_response_spans

# Allow dataset "image" column to be either a PIL.Image or a local image path (string).
# This avoids relying on HuggingFace `Image` feature embedding during `save_to_disk`.
from PIL import Image

# 避免超大卫星图触发 PIL DecompressionBombError
Image.MAX_IMAGE_PIXELS = None

#---------------------- prompt templates ----------------------------
from open_r1.trainer.utils.prompt_gallery import PROMPT_SG, PROMPT_CLOSE_TEMPLATE, PROMPT_CLOSE_PSG, PROMPT_CLOSE_VG150 

from src.mega_1m_category import megasg_object_categories, megasg_relation_categories
#---------------------------------------------------------------------------

def _resize_image_to_max_pixels(image, max_pixels):
    if max_pixels is None or max_pixels <= 0:
        return image
    iw, ih = image.size
    area = iw * ih
    if area <= max_pixels:
        return image

    scale = math.sqrt(float(max_pixels) / float(area))
    new_w = max(28, int(iw * scale))
    new_h = max(28, int(ih * scale))
    # Keep Qwen-VL friendly granularity.
    new_w = max(28, (new_w // 28) * 28)
    new_h = max(28, (new_h // 28) * 28)
    resample = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
    return image.resize((new_w, new_h), resample=resample)


def _bbox_center(box):
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def _clip_bbox_to_tile(box, x0, y0, x1, y1):
    nx1 = max(float(box[0]), float(x0))
    ny1 = max(float(box[1]), float(y0))
    nx2 = min(float(box[2]), float(x1))
    ny2 = min(float(box[3]), float(y1))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return [int(nx1 - x0), int(ny1 - y0), int(nx2 - x0), int(ny2 - y0)]


def _is_risky_sample(area, object_count, relationship_count, risk_pixels_threshold, risk_complexity_threshold):
    complexity = int(object_count) + 2 * int(relationship_count)
    very_large_image = area >= int(risk_pixels_threshold) * 3
    large_and_dense = area >= int(risk_pixels_threshold) and complexity >= int(risk_complexity_threshold)
    return very_large_image or large_and_dense


def maybe_tile_risky_sample(
    image,
    objects,
    relationships,
    rng=None,
    enabled=True,
    force_tile_all_samples=False,
    risk_pixels_threshold=12_000_000,
    risk_complexity_threshold=500,
    tile_max_pixels=8_028_16,
    tile_overlap_ratio=0.5,
    tile_pick_index=None,
    min_objects_after_tile=24,
    max_tile_trials=8,
):
    """
    Crop one deterministic tile for risky samples only.
    Keeps bbox/relationships consistent inside the selected tile.
    """
    if not enabled:
        return image, objects, relationships
    if rng is None:
        rng = random

    iw, ih = image.size
    area = iw * ih
    if (not force_tile_all_samples) and (not _is_risky_sample(
        area,
        object_count=len(objects),
        relationship_count=len(relationships),
        risk_pixels_threshold=risk_pixels_threshold,
        risk_complexity_threshold=risk_complexity_threshold,
    )):
        return image, objects, relationships

    if tile_max_pixels is None or tile_max_pixels <= 0:
        return image, objects, relationships

    tile_side = int(math.sqrt(float(tile_max_pixels)))
    tile_w = min(iw, max(56, tile_side))
    tile_h = min(ih, max(56, tile_side))
    tile_w = max(28, (tile_w // 28) * 28)
    tile_h = max(28, (tile_h // 28) * 28)
    if tile_w >= iw and tile_h >= ih:
        return image, objects, relationships

    # Build a coarse grid of candidate tiles.
    overlap = float(tile_overlap_ratio)
    overlap = min(max(overlap, 0.0), 0.9)
    step_x = max(28, int(tile_w * (1.0 - overlap)))
    step_y = max(28, int(tile_h * (1.0 - overlap)))
    xs = [0]
    ys = [0]
    if iw > tile_w:
        xs = list(range(0, max(1, iw - tile_w + 1), step_x))
        if xs[-1] != iw - tile_w:
            xs.append(iw - tile_w)
    if ih > tile_h:
        ys = list(range(0, max(1, ih - tile_h + 1), step_y))
        if ys[-1] != ih - tile_h:
            ys.append(ih - tile_h)

    candidates = [(x0, y0) for x0 in xs for y0 in ys]
    if len(candidates) == 0:
        return image, objects, relationships

    # Deterministic tile cycling for "tile every sample" mode.
    if tile_pick_index is not None:
        chosen_idx = int(tile_pick_index) % len(candidates)
        candidates = [candidates[chosen_idx]]
    else:
        rng.shuffle(candidates)
        if max_tile_trials is not None and max_tile_trials > 0:
            candidates = candidates[: max_tile_trials]

    best_pack = None
    best_score = -1
    for x0, y0 in candidates:
        x1, y1 = x0 + tile_w, y0 + tile_h
        kept_objects = []
        kept_ids = set()
        for obj in objects:
            box = obj["bbox"]
            cx, cy = _bbox_center(box)
            if not (x0 <= cx < x1 and y0 <= cy < y1):
                continue
            clipped = _clip_bbox_to_tile(box, x0, y0, x1, y1)
            if clipped is None:
                continue
            kept_ids.add(obj["id"])
            kept_objects.append({"id": obj["id"], "bbox": clipped})

        kept_rels = [r for r in relationships if r["subject"] in kept_ids and r["object"] in kept_ids]
        score = len(kept_objects) + 2 * len(kept_rels)
        if score > best_score:
            best_score = score
            best_pack = (x0, y0, kept_objects, kept_rels)

        if len(kept_objects) >= int(min_objects_after_tile):
            cropped = image.crop((x0, y0, x1, y1))
            return cropped, kept_objects, kept_rels

    if best_pack is None:
        return image, objects, relationships
    x0, y0, kept_objects, kept_rels = best_pack
    if len(kept_objects) == 0:
        return image, objects, relationships
    x1, y1 = x0 + tile_w, y0 + tile_h
    cropped = image.crop((x0, y0, x1, y1))
    return cropped, kept_objects, kept_rels


def maybe_resize_risky_sample(
    image,
    object_count,
    relationship_count,
    enabled=True,
    risk_max_pixels=802816,
    risk_pixels_threshold=25_000_000,
    risk_complexity_threshold=1200,
):
    """
    Adaptive image downscale only for OOM-risk samples.
    A sample is considered risky when:
      1) image is very large AND graph is dense, or
      2) image is extremely large.
    """
    if not enabled:
        return image
    if risk_max_pixels is None or risk_max_pixels <= 0:
        return image

    iw, ih = image.size
    area = iw * ih
    if not _is_risky_sample(
        area,
        object_count=object_count,
        relationship_count=relationship_count,
        risk_pixels_threshold=risk_pixels_threshold,
        risk_complexity_threshold=risk_complexity_threshold,
    ):
        return image

    return _resize_image_to_max_pixels(image, risk_max_pixels)


def _stable_sample_uid(sample):
    """Build a stable per-sample uid used for deterministic RNG seeding."""
    for key in ("image_id", "img_id", "id", "uid", "file_name", "filename"):
        if key in sample and sample[key] is not None:
            return str(sample[key])

    image_field = sample.get("image")
    if isinstance(image_field, str):
        return image_field

    # Fallback: deterministic signature from annotation sizes.
    obj_count = len(sample.get("objects", [])) if sample.get("objects") is not None else 0
    rel_count = len(sample.get("relationships", [])) if sample.get("relationships") is not None else 0
    return f"fallback_{obj_count}_{rel_count}"


def _build_deterministic_rng(base_seed, sample_uid, sample_visit):
    payload = f"{int(base_seed)}::{sample_uid}::{int(sample_visit)}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    seed = int.from_bytes(digest, byteorder="big", signed=False) % (2**32)
    return random.Random(seed)


def sample_random_subgraph(
    objects,
    relationships,
    max_objects=None,
    max_relationships=None,
    rng=None,
):
    """Sample a reproducible random subgraph under object/relationship budgets."""
    if rng is None:
        rng = random

    if (max_objects is None or max_objects <= 0) and (max_relationships is None or max_relationships <= 0):
        return objects, relationships

    id_to_obj = {obj["id"]: obj for obj in objects}
    valid_obj_ids = set(id_to_obj.keys())
    rels = [r for r in relationships if r["subject"] in valid_obj_ids and r["object"] in valid_obj_ids]

    # Step 1: object budget via random walk expansion on relation graph.
    if max_objects is not None and max_objects > 0 and len(objects) > max_objects:
        adjacency = {obj_id: set() for obj_id in valid_obj_ids}
        for rel in rels:
            s = rel["subject"]
            o = rel["object"]
            adjacency[s].add(o)
            adjacency[o].add(s)

        start = rng.choice(list(valid_obj_ids))
        selected = {start}
        frontier = [start]

        while len(selected) < max_objects and frontier:
            current = frontier.pop(0)
            neighbors = list(adjacency.get(current, []))
            rng.shuffle(neighbors)
            progressed = False
            for nb in neighbors:
                if nb not in selected:
                    selected.add(nb)
                    frontier.append(nb)
                    progressed = True
                    if len(selected) >= max_objects:
                        break

            # If no new node can be expanded, break to fallback random fill.
            if not progressed and not frontier and len(selected) < max_objects:
                break

            # If frontier is exhausted but graph still expandable, restart from a selected node.
            if progressed and not frontier and len(selected) < max_objects:
                frontier.append(rng.choice(list(selected)))

        if len(selected) < max_objects:
            remaining = [obj_id for obj_id in valid_obj_ids if obj_id not in selected]
            rng.shuffle(remaining)
            selected.update(remaining[: max_objects - len(selected)])

        kept_ids = selected
    else:
        kept_ids = set(valid_obj_ids)

    sampled_objects = [obj for obj in objects if obj["id"] in kept_ids]
    sampled_rels = [r for r in rels if r["subject"] in kept_ids and r["object"] in kept_ids]

    # Step 2: relationship budget by reproducible random sampling.
    if max_relationships is not None and max_relationships > 0 and len(sampled_rels) > max_relationships:
        indices = list(range(len(sampled_rels)))
        chosen = set(rng.sample(indices, max_relationships))
        sampled_rels = [rel for idx, rel in enumerate(sampled_rels) if idx in chosen]

    return sampled_objects, sampled_rels


def format_answer(objects:str, relationships:str, shuffle=False):
    if isinstance(objects, str):
        objects = json.loads(objects) # a list of {"id": xxx, "bbox": xxx}
    if isinstance(relationships, str):
        relationships = json.loads(relationships)

    if shuffle:
        random.shuffle(objects)

        obj_map = {}
        new_objects = []
        for new_idx, obj in enumerate(objects):
            name, old_idx = obj["id"].split('.')
            bbox = obj["bbox"]

            new_obj = '%s.%s'%(name, new_idx+1)
            obj_map[obj["id"]]  = new_obj

            new_objects.append({"id": new_obj, "bbox": bbox})

        new_rels = []
        for r in relationships:
            sub = obj_map[r["subject"]]
            obj = obj_map[r["object"]]
            rel = r["predicate"]
            tmp = {"subject": sub, 
                   "predicate": rel,
                   "object": obj 
                   }

            new_rels.append(tmp)
        objects, relationships = new_objects, new_rels


    objects = [json.dumps(e) for e in objects]
    relationships = [json.dumps(e) for e in relationships]
    

    # Format structured answer
    structured_answer = (
        "```json\n"
        "{\n"
        "  \"objects\": [\n" + ",\n".join(objects) + "\n  ],\n"
        "  \"relationships\": [\n" + ",\n".join(relationships) + "\n  ]\n"
        "}\n"
        "```\n"
    )
    return structured_answer


def replace_answer_format(item: str) -> str:
    return item.replace("<answer>", "```json").replace("</answer>", "```")


def _load_train_dataset(dataset_name: str):
    dataset_path = Path(dataset_name)
    if dataset_path.exists() and dataset_path.is_dir() and (dataset_path / "dataset_dict.json").exists():
        ds_local = load_from_disk(dataset_name)
        return ds_local["train"] if isinstance(ds_local, DatasetDict) else ds_local
    if dataset_path.exists() and dataset_path.is_file() and dataset_path.suffix.lower() in {".json", ".jsonl"}:
        return load_dataset("json", data_files={"train": str(dataset_path)})["train"]
    return load_dataset(dataset_name)["train"]


def _messages_from_dataset(example):
    messages = example.get("messages")
    if not messages:
        raise ValueError("use_dataset_messages=True requires each sample to contain a non-empty messages field")
    if isinstance(messages, str):
        messages = json.loads(messages)
    return messages


def _messages_before_final_assistant(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the prompt turns before the final assistant response."""

    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "assistant":
            return messages[:idx]
    raise ValueError("assistant-only SFT masking requires at least one assistant message")


def _active_token_count(batch: dict[str, Any]) -> int:
    """Count non-padding tokens in a processor output batch with one row."""

    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        return int(attention_mask[0].sum().item())
    return int(batch["input_ids"].shape[-1])


def _token_count_for_messages(processor: Any, messages: list[dict[str, Any]], max_length: int, *, add_generation_prompt: bool) -> int:
    """Tokenize a message prefix and return its active token count."""

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    image_input = process_vision_info(messages)[0]
    batch = processor(
        text=[text],
        images=image_input,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return _active_token_count(batch)


def _assistant_response_token_spans(
    processor: Any,
    messages: list[dict[str, Any]],
    max_length: int,
) -> list[tuple[int, int]]:
    """Return active-token spans for every assistant response in a chat."""

    spans: list[tuple[int, int]] = []
    for idx, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        start = _token_count_for_messages(
            processor,
            messages[:idx],
            max_length,
            add_generation_prompt=True,
        )
        end = _token_count_for_messages(
            processor,
            messages[: idx + 1],
            max_length,
            add_generation_prompt=False,
        )
        if end > start:
            spans.append((start, end))
    if not spans:
        raise ValueError("assistant-only SFT masking requires at least one assistant message")
    return spans


def format_data(
    dataset_name,
    sample,
    use_predefined_cats=False,
    remove_image_size_in_prompt=True,
    shuffle=False,
    max_objects=None,
    max_relationships=None,
    random_subgraph_sampling=True,
    rng=None,
    adaptive_image_resize=True,
    adaptive_risk_max_pixels=802816,
    adaptive_risk_pixels_threshold=25_000_000,
    adaptive_risk_complexity_threshold=1200,
    adaptive_tile_risky_sample=True,
    force_tile_all_samples=False,
    adaptive_tile_max_pixels=802816,
    adaptive_tile_overlap_ratio=0.5,
    sample_visit=0,
    adaptive_tile_min_objects=24,
    adaptive_tile_max_trials=8,
):
    """Prepare dataset example for training."""

    image_obj = sample["image"]
    if hasattr(image_obj, "convert"):
        image = image_obj.convert("RGB")
    else:
        # When dataset is loaded from json/jsonl, `image` is typically a string path.
        image = Image.open(image_obj).convert("RGB")
    iw, ih = image.size
    if use_predefined_cats:
        if 'prompt_close' in sample:
            prompt = sample['prompt_close']
        else:
            if 'psg' in dataset_name:
                prompt = PROMPT_CLOSE_PSG
            elif 'vg' in dataset_name:
                prompt = PROMPT_CLOSE_VG150
            elif 'mega' in dataset_name:
                obj_sets = megasg_object_categories[sample['data_source']]
                rel_sets = megasg_relation_categories[sample['data_source']]
                prompt = PROMPT_CLOSE_TEMPLATE.replace("{OBJ_CLS}", json.dumps(obj_sets)).replace(
                   "{REL_CLS}", json.dumps(rel_sets))
            else:
                raise Exception("Unsupported dataset:{}".format(dataset_name))
    else:
        prompt = PROMPT_SG

    use_think = 'think' in sample

    if remove_image_size_in_prompt:
        prompt = prompt.replace(f"of size ({iw} x {ih}) ", "")

    prompt = replace_answer_format(prompt)

    # normalize box to [0, 1000], and keep only keys used by the target format
    objs = []
    sample_objects = sample["objects"]
    if isinstance(sample_objects, str):
        sample_objects = json.loads(sample_objects)
    rels = sample["relationships"]
    if isinstance(rels, str):
        rels = json.loads(rels)

    image, sample_objects, rels = maybe_tile_risky_sample(
        image,
        sample_objects,
        rels,
        rng=rng,
        enabled=adaptive_tile_risky_sample,
        force_tile_all_samples=force_tile_all_samples,
        risk_pixels_threshold=adaptive_risk_pixels_threshold,
        risk_complexity_threshold=adaptive_risk_complexity_threshold,
        tile_max_pixels=adaptive_tile_max_pixels,
        tile_overlap_ratio=adaptive_tile_overlap_ratio,
        tile_pick_index=sample_visit if force_tile_all_samples else None,
        min_objects_after_tile=adaptive_tile_min_objects,
        max_tile_trials=adaptive_tile_max_trials,
    )
    iw, ih = image.size
    image = maybe_resize_risky_sample(
        image,
        object_count=len(sample_objects),
        relationship_count=len(rels),
        enabled=adaptive_image_resize,
        risk_max_pixels=adaptive_risk_max_pixels,
        risk_pixels_threshold=adaptive_risk_pixels_threshold,
        risk_complexity_threshold=adaptive_risk_complexity_threshold,
    )
    iw, ih = image.size

    for obj in sample_objects:
        box = obj['bbox']
        norm_bbox = [
            int(box[0] / iw * 1000),
            int(box[1] / ih * 1000),
            int(box[2] / iw * 1000),
            int(box[3] / ih * 1000),
        ]
        # keep the output schema compact: only id + bbox
        objs.append({"id": obj["id"], "bbox": norm_bbox})

    if random_subgraph_sampling:
        objs, rels = sample_random_subgraph(
            objs,
            rels,
            max_objects=max_objects,
            max_relationships=max_relationships,
            rng=rng,
        )
    else:
        # Backward-compatible fallback (deterministic prefix truncation).
        if max_objects is not None and max_objects > 0 and len(objs) > max_objects:
            objs = objs[:max_objects]
        kept_obj_ids = {o["id"] for o in objs}
        rels = [r for r in rels if r["subject"] in kept_obj_ids and r["object"] in kept_obj_ids]
        if max_relationships is not None and max_relationships > 0 and len(rels) > max_relationships:
            rels = rels[:max_relationships]

    answer = format_answer(objs, rels, shuffle=shuffle)
    if use_think:
        answer = '{}<answer>\n{}\n</answer>'.format(sample['think'], answer)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful and multimodal AI assistant."
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        },
    ]
    return {"messages": messages}



@dataclass
class CustomScriptArguments(ScriptArguments):
    use_predefined_cats: bool = field(
        default=False, 
        metadata={"help": "Whether to use predefined object categories"}
    )
    max_pixels: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of pixels for the image. Set <=0 or unset to disable explicit cap."},
    )
    use_dataset_messages: bool = field(
        default=False,
        metadata={"help": "Use prebuilt Qwen-style messages from the dataset instead of rebuilding legacy SGG targets."},
    )
    train_on_assistant_only: bool = field(
        default=True,
        metadata={"help": "Mask system/user prompt tokens so only assistant response tokens contribute to SFT loss."},
    )
    min_pixels: Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of pixels for the image. Set <=0 or unset to disable explicit floor."},
    )
    max_objects: Optional[int] = field(
        default=160,
        metadata={"help": "Optional cap on objects per sample for SFT target JSON."},
    )
    max_relationships: Optional[int] = field(
        default=600,
        metadata={"help": "Optional cap on relationships per sample for SFT target JSON."},
    )
    max_token_length: Optional[int] = field(
        default=4096,
        metadata={"help": "Max token length used by collator tokenization."},
    )
    random_subgraph_sampling: bool = field(
        default=True,
        metadata={"help": "Enable reproducible random subgraph sampling instead of fixed truncation."},
    )
    subgraph_seed: Optional[int] = field(
        default=None,
        metadata={"help": "Seed for random subgraph sampling. If unset, falls back to training --seed."},
    )
    adaptive_image_resize: bool = field(
        default=True,
        metadata={"help": "Enable adaptive downscale only for OOM-risk samples."},
    )
    adaptive_risk_max_pixels: Optional[int] = field(
        default=802816,
        metadata={"help": "Target max pixels for risky samples only."},
    )
    adaptive_risk_pixels_threshold: Optional[int] = field(
        default=12000000,
        metadata={"help": "Image area threshold to consider a sample large."},
    )
    adaptive_risk_complexity_threshold: Optional[int] = field(
        default=500,
        metadata={"help": "Annotation complexity threshold (objects + 2*relationships)."},
    )
    adaptive_tile_risky_sample: bool = field(
        default=True,
        metadata={"help": "Enable deterministic tiling for risky samples before subgraph sampling."},
    )
    force_tile_all_samples: bool = field(
        default=False,
        metadata={"help": "Force tiling for every sample (not only risky samples)."},
    )
    adaptive_tile_max_pixels: Optional[int] = field(
        default=802816,
        metadata={"help": "Tile area budget used when tiling risky samples."},
    )
    adaptive_tile_overlap_ratio: float = field(
        default=0.5,
        metadata={"help": "Tile overlap ratio in [0, 0.9]. Used when building tile grid."},
    )
    adaptive_tile_min_objects: Optional[int] = field(
        default=24,
        metadata={"help": "Minimum desired object count in selected tile."},
    )
    adaptive_tile_max_trials: Optional[int] = field(
        default=8,
        metadata={"help": "Max candidate tile trials per risky sample."},
    )



def main():
    accelerator = Accelerator()
    # args
    parser = TrlParser((CustomScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # load dataset
    train_dataset = _load_train_dataset(script_args.dataset_name)

    print(f"Training set size: {len(train_dataset)}")
    # print(f"Validation set size: {len(val_dataset)}")
    # Avoid dumping gigantic JSON in logs; print a compact sample summary instead.
    if accelerator.is_main_process:
        sample0 = train_dataset[0]
        raw_img = sample0.get("image")
        if hasattr(raw_img, "size"):
            iw, ih = raw_img.size
        else:
            with Image.open(raw_img).convert("RGB") as _im:
                iw, ih = _im.size
        raw_objs = sample0.get("objects", [])
        if isinstance(raw_objs, str):
            raw_objs = json.loads(raw_objs)
        raw_rels = sample0.get("relationships", [])
        if isinstance(raw_rels, str):
            raw_rels = json.loads(raw_rels)
        print(
            f"[Sample0] image_size=({iw}x{ih}), "
            f"objects={len(raw_objs)}, relationships={len(raw_rels)}, "
            f"use_predefined_cats={script_args.use_predefined_cats}"
        )

    
    # model config.
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False, #if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs


    model_type = None
    base_name = None
    model_name = model_args.model_name_or_path.lower()
    config_model_type = ""
    config_path = Path(model_args.model_name_or_path) / "config.json"
    if config_path.is_file():
        try:
            config_model_type = str(json.loads(config_path.read_text(encoding="utf-8")).get("model_type", "")).lower()
        except json.JSONDecodeError:
            config_model_type = ""
    model_type_hint = f"{model_name} {config_model_type}"

    if any(key in model_type_hint for key in ['qwen2_5_vl', 'qwen25vl', 'qwen2.5vl', 'qwen2.5-vl', 'qwen2-5-vl', 'qwen-2.5-vl']):
        model_type = "qwen2.5vl"
        if '3b' in model_type_hint:
            base_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        else:
            base_name = "Qwen/Qwen2.5-VL-7B-Instruct"

    elif any(key in model_type_hint for key in ['qwen2_vl', 'qwen2vl', 'qwen2-vl', 'qwen-2-vl']):
        model_type = "qwen2vl"
        if '7b' in model_type_hint:
            base_name = "Qwen/Qwen2-VL-7B-Instruct"
        elif '2b' in model_type_hint:
            base_name = "Qwen/Qwen2-VL-2B-Instruct"
        else:
            raise Exception(f"Unknown model size in: {model_name}")

    else:
        raise Exception(f"Unknown model type: {model_args.model_name_or_path}")

    model_path_is_local = os.path.isdir(model_args.model_name_or_path)
    processor_source = model_args.model_name_or_path if model_path_is_local else base_name
    processor_kwargs = {
        "local_files_only": model_path_is_local,
    }
    if script_args.min_pixels is not None and script_args.min_pixels > 0:
        processor_kwargs["min_pixels"] = script_args.min_pixels
    if script_args.max_pixels is not None and script_args.max_pixels > 0:
        processor_kwargs["max_pixels"] = script_args.max_pixels

    processor = AutoProcessor.from_pretrained(
        processor_source,
        **processor_kwargs,
    )
    model_cls = None
    if model_type == "qwen2vl":
        model_cls = Qwen2VLForConditionalGeneration
    elif model_type == "qwen2.5vl":
        model_cls = Qwen2_5_VLForConditionalGeneration

    assert model_cls is not None, " Unsupported model:{}".format(model_args.model_name_or_path)

    model = model_cls.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )

    class Collator(object):
        def __init__(
            self,
            dataset_name,
            processor,
            use_predefined_cats,
            use_dataset_messages,
            train_on_assistant_only,
            max_length,
            max_objects,
            max_relationships,
            random_subgraph_sampling,
            subgraph_seed,
            adaptive_image_resize,
            adaptive_risk_max_pixels,
            adaptive_risk_pixels_threshold,
            adaptive_risk_complexity_threshold,
            adaptive_tile_risky_sample,
            force_tile_all_samples,
            adaptive_tile_max_pixels,
            adaptive_tile_overlap_ratio,
            adaptive_tile_min_objects,
            adaptive_tile_max_trials,
        ):
            self.dataset_name = dataset_name
            self.processor = processor
            self.use_predefined_cats = use_predefined_cats
            self.use_dataset_messages = use_dataset_messages
            self.train_on_assistant_only = train_on_assistant_only
            self._db = {}
            self.max_length = max_length
            self.max_objects = max_objects
            self.max_relationships = max_relationships
            self.random_subgraph_sampling = random_subgraph_sampling
            self.subgraph_seed = subgraph_seed
            self.adaptive_image_resize = adaptive_image_resize
            self.adaptive_risk_max_pixels = adaptive_risk_max_pixels
            self.adaptive_risk_pixels_threshold = adaptive_risk_pixels_threshold
            self.adaptive_risk_complexity_threshold = adaptive_risk_complexity_threshold
            self.adaptive_tile_risky_sample = adaptive_tile_risky_sample
            self.force_tile_all_samples = force_tile_all_samples
            self.adaptive_tile_max_pixels = adaptive_tile_max_pixels
            self.adaptive_tile_overlap_ratio = adaptive_tile_overlap_ratio
            self.adaptive_tile_min_objects = adaptive_tile_min_objects
            self.adaptive_tile_max_trials = adaptive_tile_max_trials

        def __call__(self, examples):
            # Get the texts and images, and apply the chat template
            texts, image_inputs = [], []
            response_spans = []
            for example in examples:
                if str(example) not in self._db:
                    self._db[str(example)] = 0

                visit_idx = self._db[str(example)]
                sample_uid = _stable_sample_uid(example)
                rng = _build_deterministic_rng(self.subgraph_seed, sample_uid, visit_idx)
                shuffle = (visit_idx > 0) and (rng.random() > 0.5)
                if self.use_dataset_messages:
                    format_example = _messages_from_dataset(example)
                else:
                    format_example = format_data(
                        self.dataset_name,
                        example,
                        use_predefined_cats=self.use_predefined_cats,
                        shuffle=shuffle,
                        max_objects=self.max_objects,
                        max_relationships=self.max_relationships,
                        random_subgraph_sampling=self.random_subgraph_sampling,
                        rng=rng,
                        adaptive_image_resize=self.adaptive_image_resize,
                        adaptive_risk_max_pixels=self.adaptive_risk_max_pixels,
                        adaptive_risk_pixels_threshold=self.adaptive_risk_pixels_threshold,
                        adaptive_risk_complexity_threshold=self.adaptive_risk_complexity_threshold,
                        adaptive_tile_risky_sample=self.adaptive_tile_risky_sample,
                        force_tile_all_samples=self.force_tile_all_samples,
                        adaptive_tile_max_pixels=self.adaptive_tile_max_pixels,
                        adaptive_tile_overlap_ratio=self.adaptive_tile_overlap_ratio,
                        sample_visit=visit_idx,
                        adaptive_tile_min_objects=self.adaptive_tile_min_objects,
                        adaptive_tile_max_trials=self.adaptive_tile_max_trials,
                    )["messages"]
                self._db[str(example)] += 1

                text = self.processor.apply_chat_template(format_example, tokenize=False, add_generation_prompt=False)
                image_input = process_vision_info(format_example)[0]
                if self.train_on_assistant_only:
                    response_spans.append(
                        _assistant_response_token_spans(
                            self.processor,
                            format_example,
                            self.max_length,
                        )
                    )
                texts.append(text)
                image_inputs.append(image_input)
    
            # Tokenize the texts and process the images
            batch = self.processor(
                text=texts,
                images=image_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
    
            # The labels are the input_ids, and we mask the padding tokens in the loss computation
            labels = batch["input_ids"].clone()
            if self.train_on_assistant_only:
                mask_labels_to_response_spans(labels, response_spans, batch.get("attention_mask"))
            labels[labels == self.processor.tokenizer.pad_token_id] = -100  #
            # Ignore the image token index in the loss computation (model specific)
            if isinstance(self.processor, Qwen2VLProcessor) or isinstance(self.processor, Qwen2_5_VLProcessor):
                image_tokens = [151652,151653,151655]
            else:
                image_tokens = [self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)]
            for image_token_id in image_tokens:
                labels[labels == image_token_id] = -100
            batch["labels"] = labels
    
            return batch

    ################
    # Training
    ################
    try:
        rank = torch.distributed.get_rank()  # GPU ID or node rank
        world_size = torch.distributed.get_world_size()  # Total number of GPUs/nodes

        global_batch_size = (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * world_size
        )
        total_steps = len(train_dataset) // global_batch_size * training_args.num_train_epochs
        print("*"*100, "\nglobal_batch_size:", global_batch_size, " total steps:", total_steps, "\n", "*"*100)
    except:
        pass

    training_args.gradient_checkpointing_kwargs={"use_reentrant": False}
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    training_args.dataset_text_field=""

    tokenizer_max_len = getattr(processor.tokenizer, "model_max_length", 32768)
    if tokenizer_max_len is None or tokenizer_max_len > 100_000:
        tokenizer_max_len = 32768
    # Cap effective sequence length for memory safety on 24GB GPUs.
    if script_args.max_token_length is not None and script_args.max_token_length > 0:
        tokenizer_max_len = min(tokenizer_max_len, script_args.max_token_length)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, 
        eval_dataset=None, #val_dataset,
        processing_class=processor.tokenizer,
        data_collator=Collator(
            script_args.dataset_name,
            processor,
            script_args.use_predefined_cats,
            script_args.use_dataset_messages,
            script_args.train_on_assistant_only,
            max_length=tokenizer_max_len,
            max_objects=script_args.max_objects,
            max_relationships=script_args.max_relationships,
            random_subgraph_sampling=script_args.random_subgraph_sampling,
            subgraph_seed=script_args.subgraph_seed if script_args.subgraph_seed is not None else training_args.seed,
            adaptive_image_resize=script_args.adaptive_image_resize,
            adaptive_risk_max_pixels=script_args.adaptive_risk_max_pixels,
            adaptive_risk_pixels_threshold=script_args.adaptive_risk_pixels_threshold,
            adaptive_risk_complexity_threshold=script_args.adaptive_risk_complexity_threshold,
            adaptive_tile_risky_sample=script_args.adaptive_tile_risky_sample,
            force_tile_all_samples=script_args.force_tile_all_samples,
            adaptive_tile_max_pixels=script_args.adaptive_tile_max_pixels,
            adaptive_tile_overlap_ratio=script_args.adaptive_tile_overlap_ratio,
            adaptive_tile_min_objects=script_args.adaptive_tile_min_objects,
            adaptive_tile_max_trials=script_args.adaptive_tile_max_trials,
        ),
        peft_config=get_peft_config(model_args),
    )
    # Check for existing checkpoint
    def find_valid_checkpoint(output_dir):
        ckpt_re = re.compile(r"checkpoint-(\d+)$")      # ↳ ends right after the digits
        
        checkpoints = sorted(
            [
                p for p in glob.glob(os.path.join(output_dir, "checkpoint-*"))
                if ckpt_re.search(os.path.basename(p))   # keep only pure-numeric checkpoints
            ],
            key=lambda p: int(ckpt_re.search(os.path.basename(p)).group(1))
        )
        for ckpt in reversed(checkpoints):  # Check latest first
            if glob.glob(os.path.join(ckpt, "global_step*")):
                return ckpt
        return None
    
    ckpt_to_resume = find_valid_checkpoint(training_args.output_dir)
    if ckpt_to_resume:
        print(f"[INFO] Resuming from checkpoint: {ckpt_to_resume}")
        trainer.train(resume_from_checkpoint=ckpt_to_resume)
    else:
        print("[INFO] Starting training from scratch")
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    main()
