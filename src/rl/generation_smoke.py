"""One-sample no-training generation smoke for Agentic GRPO."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from src.rl.model_load_smoke import (
    ModelLoadSmokeConfig,
    _build_smoke_config,
    _config_section,
    _load_model,
    _load_processor,
    _load_yaml_config,
    _shape_summary,
    build_qwen_processor_inputs,
    resolve_processor_path,
    sample_to_qwen_messages,
)
from src.data.rlvr_dataset import RLVRDatasetConfig, load_rlvr_dataset
from src.rl.paths import ensure_output_dirs, resolve_agentic_paths
from src.rl.rewards import compute_format_reward, compute_total_reward, parse_json_action
from src.tools.zoom_tool import zoom_in


@dataclass(frozen=True)
class GenerationSmokeConfig:
    """Configuration for a one-sample generation smoke run."""

    model: ModelLoadSmokeConfig
    max_new_tokens: int = 128
    repetition_penalty: float = 1.05
    do_sample: bool = False
    temperature: float | None = None
    max_zoom_steps: int = 1
    coord_type: str = "pixel"
    zoom_output_size: int | None = 448
    zoom_min_area_ratio: float = 1e-6
    zoom_max_area_ratio: float = 0.8
    strict_format: bool = False
    action_reminder: bool = True
    prompt_mode: str = "action_only"


def _bool_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _build_generation_config(raw_config: dict[str, Any], **overrides: Any) -> GenerationSmokeConfig:
    base = _build_smoke_config(
        raw_config,
        model_path=overrides.get("model_path"),
        processor_path=overrides.get("processor_path"),
        load_model=True,
        sample_index=overrides.get("sample_index"),
        split=overrides.get("split"),
    )
    raw = _config_section(raw_config, "generation_smoke")
    return GenerationSmokeConfig(
        model=base,
        max_new_tokens=int(overrides.get("max_new_tokens") or raw.get("max_new_tokens", 128)),
        repetition_penalty=float(raw.get("repetition_penalty", 1.05)),
        do_sample=_bool_value(raw.get("do_sample"), False),
        temperature=_optional_float(raw.get("temperature")),
        max_zoom_steps=int(raw.get("max_zoom_steps", 1)),
        coord_type=str(raw.get("coord_type", "pixel")),
        zoom_output_size=int(raw.get("zoom_output_size", 448)) if raw.get("zoom_output_size", 448) is not None else None,
        zoom_min_area_ratio=float(raw.get("zoom_min_area_ratio", 1e-6)),
        zoom_max_area_ratio=float(raw.get("zoom_max_area_ratio", 0.8)),
        strict_format=_bool_value(raw.get("strict_format"), False),
        action_reminder=_bool_value(raw.get("action_reminder"), True),
        prompt_mode=str(raw.get("prompt_mode", "action_only")),
    )


def _load_sample(raw_config: dict[str, Any], cfg: ModelLoadSmokeConfig) -> tuple[Any, dict[str, Any]]:
    paths = resolve_agentic_paths(create_output=True)
    ensure_output_dirs(paths)
    dataset_raw = _config_section(raw_config, "rlvr_dataset")
    dataset_cfg = RLVRDatasetConfig(
        source=str(dataset_raw.get("source", "hf_closed")),
        dataset_path=dataset_raw.get("dataset_path") or str(paths.dataset_path),
        jsonl_dir=dataset_raw.get("jsonl_dir") or str(paths.jsonl_closed_dir),
        split=cfg.split,
        task_type=str(dataset_raw.get("task_type", "scene_graph")),
        prompt_field=str(dataset_raw.get("prompt_field", "prompt_close")),
        input_style=str(dataset_raw.get("input_style", "eagle_grounding")),
        require_image_exists=bool(dataset_raw.get("require_image_exists", True)),
    )
    dataset = load_rlvr_dataset(dataset_cfg, paths=paths)
    if len(dataset) < 1:
        raise ValueError("RLVR dataset is empty; cannot run generation smoke")
    if cfg.sample_index < 0 or cfg.sample_index >= len(dataset):
        raise IndexError(f"sample_index {cfg.sample_index} out of range for dataset length {len(dataset)}")
    return dataset, dataset[cfg.sample_index]


def _move_inputs_to_model(inputs: Any, model: Any, dtype: torch.dtype | None = torch.bfloat16) -> Any:
    device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if value.is_floating_point() and dtype is not None:
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def _generation_kwargs(processor: Any, cfg: GenerationSmokeConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": cfg.max_new_tokens,
        "do_sample": cfg.do_sample,
        "num_beams": 1,
        "use_cache": True,
        "repetition_penalty": cfg.repetition_penalty,
    }
    if cfg.temperature is not None:
        kwargs["temperature"] = cfg.temperature
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        if getattr(tok, "eos_token_id", None) is not None:
            kwargs["eos_token_id"] = tok.eos_token_id
        if getattr(tok, "pad_token_id", None) is not None:
            kwargs["pad_token_id"] = tok.pad_token_id
    return kwargs


def _decode_generated(processor: Any, out_ids: torch.Tensor, prompt_len: int) -> str:
    gen_only = out_ids[:, prompt_len:]
    return processor.batch_decode(gen_only, skip_special_tokens=True)[0]


def _generate_once(
    *,
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    cfg: GenerationSmokeConfig,
    vision_info_fn: Callable[[list[dict[str, Any]]], tuple[Any, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
    prepared = build_qwen_processor_inputs(processor, messages, vision_info_fn=vision_info_fn)
    inputs = _move_inputs_to_model(prepared["batch"], model)
    prompt_len = int(inputs["input_ids"].shape[-1])
    with torch.inference_mode():
        out_ids = model.generate(**inputs, **_generation_kwargs(processor, cfg))
    decoded = _decode_generated(processor, out_ids, prompt_len)
    info = {
        "prompt_token_count": prompt_len,
        "chat_text_chars": len(prepared["text"]),
        "num_images": len(prepared["images"] or []),
        "batch_shapes": _shape_summary(prepared["batch"]),
    }
    return decoded, info


def _assistant_message(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def _apply_action_reminder(messages: list[dict[str, Any]]) -> None:
    reminder = (
        "\n\nGeneration smoke override: return exactly one short raw JSON action object and no markdown fences. "
        "This is a format smoke test, not full scene-graph evaluation. Do not enumerate all objects. "
        "Use either {\"thought\":\"...\",\"action\":\"zoom_in\",\"bbox\":[x1,y1,x2,y2]} "
        "or, if answering now, exactly {\"thought\":\"...\",\"action\":\"final_answer\",\"answer\":{\"objects\":[],\"relationships\":[]}}."
    )
    for message in reversed(messages):
        content = message.get("content")
        if isinstance(content, list):
            for item in reversed(content):
                if item.get("type") == "text":
                    item["text"] = str(item.get("text", "")) + reminder
                    return


def _action_only_messages(sample: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = (
        "You are running an Agentic GRPO format smoke test on one remote-sensing image. "
        "Return exactly one short raw JSON action object, with no markdown fences and no extra text. "
        "Do not enumerate the scene graph. "
        "Valid option 1: {\"thought\":\"need local evidence\",\"action\":\"zoom_in\",\"bbox\":[x1,y1,x2,y2]}. "
        "Valid option 2: {\"thought\":\"format check\",\"action\":\"final_answer\",\"answer\":{\"objects\":[],\"relationships\":[]}}."
    )
    return [
        {
            "role": "system",
            "content": "You are a precise remote-sensing visual grounding and scene graph assistant.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def _initial_messages(sample: dict[str, Any], cfg: GenerationSmokeConfig) -> list[dict[str, Any]]:
    if cfg.prompt_mode == "action_only":
        return _action_only_messages(sample)
    if cfg.prompt_mode == "dataset":
        messages = sample_to_qwen_messages(sample)
        if cfg.action_reminder:
            _apply_action_reminder(messages)
        return messages
    raise ValueError(f"unknown generation smoke prompt_mode: {cfg.prompt_mode}")


def _tool_observation_message(observation: dict[str, Any], crop_image: Any | None) -> dict[str, Any]:
    text = (
        "zoom_in observation metadata: "
        + json.dumps(observation, ensure_ascii=False, separators=(",", ":"))
        + "\nReturn the next action as strict JSON. If enough evidence is available, use final_answer."
    )
    content = []
    if crop_image is not None:
        content.append({"type": "image", "image": crop_image})
    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def _trajectory_summary(steps: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "steps": steps,
        "tool_calls": [s["action_json"] for s in steps if s.get("action") == "zoom_in" and s.get("action_json")],
        "used_zoom": any(s.get("action") == "zoom_in" for s in steps),
        "invalid_bbox": any(s.get("observation", {}).get("valid") is False for s in steps),
        "num_steps": len(steps),
    }


def run_generation_smoke(
    *,
    config_path: str | None = None,
    output_root: str | None = None,
    model_path: str | None = None,
    processor_path: str | None = None,
    split: str | None = None,
    sample_index: int | None = None,
    max_new_tokens: int | None = None,
    processor_loader: Callable[[str, ModelLoadSmokeConfig], Any] | None = None,
    model_loader: Callable[[str, ModelLoadSmokeConfig], Any] | None = None,
    vision_info_fn: Callable[[list[dict[str, Any]]], tuple[Any, Any]] | None = None,
) -> dict[str, Any]:
    """Run one no-training generation chain with at most one zoom-in call."""

    if output_root:
        os.environ["OUTPUT_ROOT"] = output_root
    raw_config = _load_yaml_config(config_path)
    cfg = _build_generation_config(
        raw_config,
        model_path=model_path,
        processor_path=processor_path,
        split=split,
        sample_index=sample_index,
        max_new_tokens=max_new_tokens,
    )
    paths = resolve_agentic_paths(create_output=True)
    ensure_output_dirs(paths)
    dataset, sample = _load_sample(raw_config, cfg.model)

    processor_src = resolve_processor_path(cfg.model.model_path, cfg.model.processor_path)
    processor = (processor_loader or _load_processor)(processor_src, cfg.model)
    model = (model_loader or _load_model)(cfg.model.model_path, cfg.model)
    if hasattr(model, "eval"):
        model.eval()
    if not cfg.do_sample and hasattr(model, "generation_config"):
        model.generation_config.temperature = None

    messages = _initial_messages(sample, cfg)
    steps: list[dict[str, Any]] = []
    zoom_count = 0
    final_prediction = ""
    input_infos: list[dict[str, Any]] = []

    for step_idx in range(cfg.max_zoom_steps + 1):
        decoded, input_info = _generate_once(
            model=model,
            processor=processor,
            messages=messages,
            cfg=cfg,
            vision_info_fn=vision_info_fn,
        )
        final_prediction = decoded
        input_infos.append(input_info)
        parsed = parse_json_action(decoded, strict=cfg.strict_format)
        action_json = parsed.data if parsed.data else None
        action = action_json.get("action") if action_json else None
        step: dict[str, Any] = {
            "step": step_idx,
            "model_output": decoded,
            "is_valid_json": parsed.is_valid_json,
            "has_extra_text": parsed.has_extra_text,
            "format_reward": compute_format_reward(decoded, strict=cfg.strict_format),
            "action": action,
            "action_json": action_json,
            "input": input_info,
        }
        steps.append(step)

        if action == "zoom_in" and zoom_count < cfg.max_zoom_steps and action_json is not None:
            zoom_count += 1
            try:
                obs = zoom_in(
                    sample["image"],
                    action_json.get("bbox"),
                    coord_type=cfg.coord_type,  # type: ignore[arg-type]
                    output_size=cfg.zoom_output_size,
                    min_bbox_area_ratio=cfg.zoom_min_area_ratio,
                    max_bbox_area_ratio=cfg.zoom_max_area_ratio,
                )
                observation = obs.metadata()
                observation["has_image"] = obs.image is not None
                step["observation"] = observation
                messages.append(_assistant_message(decoded))
                messages.append(_tool_observation_message(observation, obs.image))
                if not obs.valid:
                    break
                continue
            except Exception as exc:
                step["observation"] = {"valid": False, "error": str(exc), "has_image": False}
                break
        break

    trajectory = _trajectory_summary(steps)
    reward_cfg = _config_section(raw_config, "reward")
    reward = compute_total_reward(final_prediction, sample, trajectory, reward_cfg | {"max_tool_steps": cfg.max_zoom_steps})
    summary = {
        "generation_smoke": True,
        "no_training": True,
        "checkpoint_written": False,
        "model_path": cfg.model.model_path,
        "processor_path": processor_src,
        "split": cfg.model.split,
        "sample_index": cfg.model.sample_index,
        "dataset_length": len(dataset),
        "sample": {
            "id": sample.get("id"),
            "image_id": sample.get("image_id"),
            "image": sample.get("image"),
            "width": sample.get("width"),
            "height": sample.get("height"),
            "input_style": sample.get("input_style"),
            "num_objects": len(sample.get("objects") or []),
            "num_relationships": len(sample.get("relationships") or []),
        },
        "config": {
            "max_new_tokens": cfg.max_new_tokens,
            "max_zoom_steps": cfg.max_zoom_steps,
            "coord_type": cfg.coord_type,
            "zoom_output_size": cfg.zoom_output_size,
            "strict_format": cfg.strict_format,
            "action_reminder": cfg.action_reminder,
            "prompt_mode": cfg.prompt_mode,
        },
        "trajectory": trajectory,
        "reward": reward,
        "input_infos": input_infos,
    }
    log_path = paths.output_root / "logs" / "generation_smoke.json"
    from src.rl.model_load_smoke import _write_json_no_overwrite

    actual_log = _write_json_no_overwrite(paths.output_root / "logs", log_path.name, summary)
    summary["log_path"] = str(actual_log)
    if torch.cuda.is_available():
        summary["cuda_memory_allocated_bytes"] = int(torch.cuda.memory_allocated())
        summary["cuda_memory_reserved_bytes"] = int(torch.cuda.memory_reserved())
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run one no-training Agentic GRPO generation smoke")
    parser.add_argument("--config", default="configs/agentic_grpo.yaml")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--processor-path", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args(argv)
    summary = run_generation_smoke(
        config_path=args.config,
        output_root=args.output_root,
        model_path=args.model_path,
        processor_path=args.processor_path,
        split=args.split,
        sample_index=args.sample_index,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
