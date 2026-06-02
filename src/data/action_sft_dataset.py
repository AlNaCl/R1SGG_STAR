"""Build action-format SFT data for Agentic GRPO warmup."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.data.rlvr_dataset import RLVRDatasetConfig, load_rlvr_dataset
from src.rl.grpo_trainer import _load_yaml_config
from src.rl.model_load_smoke import _bool_value, _config_section
from src.rl.paths import ensure_output_dirs, resolve_agentic_paths

SYSTEM_PROMPT = "You are a precise remote-sensing visual grounding and scene graph assistant."
PROMPT_MODES = {"dataset", "action_only", "action_content"}
TARGET_MODES = {"scene_graph", "format_only", "zoom_then_scene_graph", "mixed_scene_graph"}


@dataclass(frozen=True)
class ActionSFTBuildConfig:
    """Configuration for action-format SFT data export."""

    split: str = "train"
    max_samples: int | None = 256
    prompt_mode: str = "dataset"
    target_mode: str = "scene_graph"
    include_conversations: bool = True
    output_dir: str | None = None
    save_hf_dataset: bool = True
    overwrite: bool = False
    max_objects: int | None = 32
    max_relationships: int | None = 64


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def _non_overwriting_dir(path: Path) -> Path:
    if not path.exists():
        return path
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%fZ")
    for idx in range(1000):
        suffix = f"_{stamp}" if idx == 0 else f"_{stamp}_{idx}"
        candidate = path.with_name(f"{path.name}{suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"could not find non-overwriting directory for {path}")


def _default_output_dir(output_root: Path, split: str) -> Path:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%fZ")
    return output_root / "tmp" / f"action_sft_{split}_{stamp}"


def _coerce_bbox_xyxy(
    bbox: Any,
    *,
    width: int | None = None,
    height: int | None = None,
) -> list[int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2_or_w, y2_or_h = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if x2_or_w <= x1 or y2_or_h <= y1:
        x2 = x1 + max(0.0, x2_or_w)
        y2 = y1 + max(0.0, y2_or_h)
    else:
        x2 = x2_or_w
        y2 = y2_or_h
    if x2 <= x1 or y2 <= y1:
        return None
    if width is not None and width > 0:
        x1 = max(0.0, min(x1, float(width)))
        x2 = max(0.0, min(x2, float(width)))
    if height is not None and height > 0:
        y1 = max(0.0, min(y1, float(height)))
        y2 = max(0.0, min(y2, float(height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def _limited_graph(sample: dict[str, Any], max_objects: int | None, max_relationships: int | None) -> dict[str, Any]:
    source_objects = list(sample.get("objects") or [])
    relationships = list(sample.get("relationships") or [])
    width = int(sample["width"]) if sample.get("width") is not None else None
    height = int(sample["height"]) if sample.get("height") is not None else None
    objects = []
    for obj in source_objects:
        if not isinstance(obj, dict) or obj.get("id") is None:
            continue
        bbox = _coerce_bbox_xyxy(obj.get("bbox"), width=width, height=height)
        if bbox is None:
            continue
        objects.append({"id": str(obj["id"]), "bbox": bbox})
    if max_objects is not None and max_objects > 0 and len(objects) > max_objects:
        objects = objects[:max_objects]
    kept_ids = {obj.get("id") for obj in objects if isinstance(obj, dict)}
    relationships = [
        {"subject": str(rel.get("subject")), "predicate": str(rel.get("predicate")), "object": str(rel.get("object"))}
        for rel in relationships
        if isinstance(rel, dict) and rel.get("subject") in kept_ids and rel.get("object") in kept_ids
    ]
    if max_relationships is not None and max_relationships > 0 and len(relationships) > max_relationships:
        relationships = relationships[:max_relationships]
    return {"objects": objects, "relationships": relationships}



def _valid_xyxy(bbox: Any) -> list[float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _clip_and_pad_bbox(
    bbox: list[float],
    *,
    width: int | None,
    height: int | None,
    pad_ratio: float = 0.75,
) -> list[int]:
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = max(8.0, box_w * pad_ratio)
    pad_y = max(8.0, box_h * pad_ratio)
    x1 -= pad_x
    y1 -= pad_y
    x2 += pad_x
    y2 += pad_y
    if width is not None and width > 0:
        x1 = max(0.0, min(x1, float(width)))
        x2 = max(0.0, min(x2, float(width)))
    if height is not None and height > 0:
        y1 = max(0.0, min(y1, float(height)))
        y2 = max(0.0, min(y2, float(height)))
    if x2 <= x1:
        x2 = x1 + 1.0
    if y2 <= y1:
        y2 = y1 + 1.0
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def _select_zoom_bbox(sample: dict[str, Any], graph: dict[str, Any]) -> list[int] | None:
    """Choose a deterministic local evidence bbox from the limited target graph."""

    width = int(sample["width"]) if sample.get("width") is not None else None
    height = int(sample["height"]) if sample.get("height") is not None else None
    for obj in graph.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        bbox = _valid_xyxy(obj.get("bbox"))
        if bbox is not None:
            return _clip_and_pad_bbox(bbox, width=width, height=height)
    return None


def _stable_sample_bucket(sample: dict[str, Any]) -> int:
    key = str(sample.get("id") or sample.get("image_id") or sample.get("image") or "")
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _resolve_target_mode(sample: dict[str, Any], target_mode: str) -> str:
    if target_mode == "mixed_scene_graph":
        return "zoom_then_scene_graph" if _stable_sample_bucket(sample) % 2 else "scene_graph"
    return target_mode


def _compact_action_answer(
    sample: dict[str, Any],
    max_objects: int | None = None,
    max_relationships: int | None = None,
    *,
    target_mode: str = "scene_graph",
) -> str:
    if target_mode == "scene_graph":
        thought = "produce the final scene graph as action JSON"
        answer = _limited_graph(sample, max_objects, max_relationships)
    elif target_mode == "format_only":
        thought = "format check"
        answer = {"objects": [], "relationships": []}
    else:
        raise ValueError(f"unknown target_mode: {target_mode}")
    payload = {
        "thought": thought,
        "action": "final_answer",
        "answer": answer,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _compact_zoom_action(bbox: list[int]) -> str:
    payload = {
        "thought": "inspect local evidence",
        "action": "zoom_in",
        "bbox": bbox,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _action_only_prompt(sample: dict[str, Any]) -> str:
    return (
        "You are running an Agentic GRPO format smoke test on one remote-sensing image. "
        "Return exactly one short raw JSON action object, with no markdown fences and no extra text. "
        "Do not enumerate the scene graph. "
        "Valid option 1: {\"thought\":\"need local evidence\",\"action\":\"zoom_in\",\"bbox\":[x1,y1,x2,y2]}. "
        "Valid option 2: {\"thought\":\"format check\",\"action\":\"final_answer\",\"answer\":{\"objects\":[],\"relationships\":[]}}."
    )


def _action_content_prompt(sample: dict[str, Any]) -> str:
    size_text = f"{sample.get('width')}x{sample.get('height')}" if sample.get("width") and sample.get("height") else "unknown size"
    return (
        "You are analyzing one ultra-high-resolution remote-sensing image. "
        f"Image size: {size_text} pixels. "
        "Return strict raw JSON action objects only, with no markdown fences and no extra text. "
        "Use {\"thought\":\"...\",\"action\":\"zoom_in\",\"bbox\":[x1,y1,x2,y2]} when local evidence is needed. "
        "Use {\"thought\":\"...\",\"action\":\"final_answer\",\"answer\":{\"objects\":[],\"relationships\":[]}} when answering. "
        "The final_answer objects must use {\"id\":string,\"bbox\":[x1,y1,x2,y2]} and relationships must use "
        "{\"subject\":string,\"predicate\":string,\"object\":string}. Preserve class.index object ids."
    )


def _tool_observation_message_text(sample: dict[str, Any], bbox: list[int]) -> str:
    metadata = {
        "bbox_xyxy": bbox,
        "original_size": [sample.get("width"), sample.get("height")],
        "valid": True,
        "source": sample.get("image"),
    }
    return (
        "zoom_in observation metadata: "
        + json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
        + "\nUse the original image and this observation to return the final scene graph as strict JSON."
    )


def sample_to_action_sft_record(
    sample: dict[str, Any],
    *,
    prompt_mode: str = "dataset",
    target_mode: str = "scene_graph",
    include_conversations: bool = True,
    max_objects: int | None = 32,
    max_relationships: int | None = 64,
) -> dict[str, Any]:
    """Convert one RLVR sample to a Qwen-style action-format SFT record."""

    if target_mode not in TARGET_MODES:
        raise ValueError(f"unknown target_mode: {target_mode}")
    if prompt_mode == "dataset":
        prompt = str(sample.get("prompt") or sample.get("base_prompt") or "Generate the scene graph for this image.")
    elif prompt_mode == "action_only":
        if target_mode != "format_only":
            raise ValueError("prompt_mode=action_only is only compatible with target_mode=format_only")
        prompt = _action_only_prompt(sample)
    elif prompt_mode == "action_content":
        prompt = _action_content_prompt(sample)
    else:
        raise ValueError(f"unknown prompt_mode: {prompt_mode}")

    resolved_target_mode = _resolve_target_mode(sample, target_mode)
    if resolved_target_mode == "format_only":
        final_target_mode = "format_only"
    elif resolved_target_mode in {"scene_graph", "zoom_then_scene_graph"}:
        final_target_mode = "scene_graph"
    else:
        raise ValueError(f"unknown resolved target_mode: {resolved_target_mode}")
    target = _compact_action_answer(
        sample,
        max_objects=max_objects,
        max_relationships=max_relationships,
        target_mode=final_target_mode,
    )
    limited_graph = json.loads(target)["answer"]
    image = str(sample["image"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]},
    ]
    target_actions = [target]
    if resolved_target_mode == "zoom_then_scene_graph":
        zoom_bbox = _select_zoom_bbox(sample, limited_graph)
        if zoom_bbox is None:
            resolved_target_mode = "scene_graph"
        else:
            zoom_target = _compact_zoom_action(zoom_bbox)
            target_actions = [zoom_target, target]
            messages.extend(
                [
                    {"role": "assistant", "content": [{"type": "text", "text": zoom_target}]},
                    {"role": "user", "content": [{"type": "text", "text": _tool_observation_message_text(sample, zoom_bbox)}]},
                ]
            )
    messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    record: dict[str, Any] = {
        "id": sample.get("id"),
        "image_id": sample.get("image_id"),
        "image": image,
        "width": sample.get("width"),
        "height": sample.get("height"),
        "objects": limited_graph["objects"],
        "relationships": limited_graph["relationships"],
        "messages": messages,
        "target_action": target,
        "target_actions": target_actions,
        "prompt_mode": prompt_mode,
        "target_mode": target_mode,
        "target_mode_resolved": resolved_target_mode,
        "num_assistant_actions": len(target_actions),
        "used_zoom_target": resolved_target_mode == "zoom_then_scene_graph",
        "task_type": sample.get("task_type", "scene_graph"),
    }
    if include_conversations:
        record["conversations"] = [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": target},
        ]
        record["data"] = [{"type": "image", "image_list": [image]}]
    return record


def _build_config(raw_config: dict[str, Any], **overrides: Any) -> ActionSFTBuildConfig:
    raw = _config_section(raw_config, "action_sft")
    return ActionSFTBuildConfig(
        split=str(overrides.get("split") or raw.get("split", "train")),
        max_samples=_optional_int(overrides.get("max_samples") if overrides.get("max_samples") is not None else raw.get("max_samples", 256)),
        prompt_mode=str(overrides.get("prompt_mode") or raw.get("prompt_mode", "dataset")),
        target_mode=str(overrides.get("target_mode") or raw.get("target_mode", "scene_graph")),
        include_conversations=_bool_value(raw.get("include_conversations"), True),
        output_dir=overrides.get("output_dir") or raw.get("output_dir"),
        save_hf_dataset=_bool_value(raw.get("save_hf_dataset"), True),
        overwrite=_bool_value(overrides.get("overwrite") if overrides.get("overwrite") is not None else raw.get("overwrite"), False),
        max_objects=_optional_int(overrides.get("max_objects") if overrides.get("max_objects") is not None else raw.get("max_objects", 32)),
        max_relationships=_optional_int(overrides.get("max_relationships") if overrides.get("max_relationships") is not None else raw.get("max_relationships", 64)),
    )


def _arrow_safe_record(record: dict[str, Any]) -> dict[str, Any]:
    safe = dict(record)
    for key in ("messages", "conversations", "data", "objects", "relationships", "target_actions"):
        if key in safe and not isinstance(safe[key], str):
            safe[key] = json.dumps(safe[key], ensure_ascii=False, separators=(",", ":"))
    return safe


def _load_source_dataset(raw_config: dict[str, Any], split: str):
    paths = resolve_agentic_paths(create_output=True)
    dataset_raw = _config_section(raw_config, "rlvr_dataset")
    cfg = RLVRDatasetConfig(
        source=str(dataset_raw.get("source", "hf_closed")),
        dataset_path=dataset_raw.get("dataset_path") or str(paths.dataset_path),
        jsonl_dir=dataset_raw.get("jsonl_dir") or str(paths.jsonl_closed_dir),
        split=split,
        task_type=str(dataset_raw.get("task_type", "scene_graph")),
        prompt_field=str(dataset_raw.get("prompt_field", "prompt_close")),
        input_style=str(dataset_raw.get("input_style", "eagle_grounding")),
        require_image_exists=bool(dataset_raw.get("require_image_exists", True)),
    )
    return load_rlvr_dataset(cfg, paths=paths)


def build_action_sft_dataset(
    *,
    config_path: str | None = None,
    output_dir: str | None = None,
    split: str | None = None,
    max_samples: int | None = None,
    prompt_mode: str | None = None,
    target_mode: str | None = None,
    max_objects: int | None = None,
    max_relationships: int | None = None,
    overwrite: bool | None = None,
) -> dict[str, Any]:
    """Build action-format SFT JSONL and optional HF DatasetDict under OUTPUT_ROOT/tmp."""

    raw_config = _load_yaml_config(config_path)
    cfg = _build_config(
        raw_config,
        output_dir=output_dir,
        split=split,
        max_samples=max_samples,
        prompt_mode=prompt_mode,
        target_mode=target_mode,
        max_objects=max_objects,
        max_relationships=max_relationships,
        overwrite=overwrite,
    )
    paths = resolve_agentic_paths(create_output=True)
    ensure_output_dirs(paths)
    out_dir = Path(cfg.output_dir).expanduser() if cfg.output_dir else _default_output_dir(paths.output_root, cfg.split)
    if out_dir.exists() and not cfg.overwrite:
        out_dir = _non_overwriting_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=cfg.overwrite)
    jsonl_dir = out_dir / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = jsonl_dir / f"{cfg.split}.jsonl"
    if jsonl_path.exists() and not cfg.overwrite:
        raise FileExistsError(f"output JSONL already exists: {jsonl_path}")

    dataset = _load_source_dataset(raw_config, cfg.split)
    limit = len(dataset) if cfg.max_samples is None else min(len(dataset), cfg.max_samples)
    records = [
        sample_to_action_sft_record(
            dataset[idx],
            prompt_mode=cfg.prompt_mode,
            target_mode=cfg.target_mode,
            include_conversations=cfg.include_conversations,
            max_objects=cfg.max_objects,
            max_relationships=cfg.max_relationships,
        )
        for idx in range(limit)
    ]
    with jsonl_path.open("x" if not cfg.overwrite else "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    hf_path = None
    if cfg.save_hf_dataset:
        from datasets import Dataset, DatasetDict

        hf_path = out_dir / "hf_dataset"
        if hf_path.exists() and not cfg.overwrite:
            hf_path = _non_overwriting_dir(hf_path)
        DatasetDict({cfg.split: Dataset.from_list([_arrow_safe_record(record) for record in records])}).save_to_disk(str(hf_path))

    summary = {
        "action_sft_dataset": True,
        "split": cfg.split,
        "prompt_mode": cfg.prompt_mode,
        "target_mode": cfg.target_mode,
        "num_records": len(records),
        "max_objects": cfg.max_objects,
        "max_relationships": cfg.max_relationships,
        "source_length": len(dataset),
        "output_dir": str(out_dir),
        "jsonl_path": str(jsonl_path),
        "hf_dataset_path": str(hf_path) if hf_path is not None else None,
        "first_record": {
            "id": records[0].get("id") if records else None,
            "image_id": records[0].get("image_id") if records else None,
            "target_chars": len(records[0].get("target_action", "")) if records else 0,
            "num_objects": len(records[0].get("objects") or []) if records else 0,
            "num_relationships": len(records[0].get("relationships") or []) if records else 0,
            "target_mode_resolved": records[0].get("target_mode_resolved") if records else None,
            "num_assistant_actions": records[0].get("num_assistant_actions") if records else 0,
        },
        "num_zoom_targets": sum(1 for record in records if record.get("used_zoom_target")),
    }
    summary_path = out_dir / "build_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build action-format SFT data for Agentic GRPO warmup")
    parser.add_argument("--config", default="configs/agentic_grpo.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--prompt-mode", choices=sorted(PROMPT_MODES), default=None)
    parser.add_argument("--target-mode", choices=sorted(TARGET_MODES), default=None)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--max-relationships", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    summary = build_action_sft_dataset(
        config_path=args.config,
        output_dir=args.output_dir,
        split=args.split,
        max_samples=args.max_samples,
        prompt_mode=args.prompt_mode,
        target_mode=args.target_mode,
        max_objects=args.max_objects,
        max_relationships=args.max_relationships,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
