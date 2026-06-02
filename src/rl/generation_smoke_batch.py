"""Batch no-training generation smoke for base-vs-adapter comparisons."""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import replace
from typing import Any, Callable

import torch

from src.data.rlvr_dataset import RLVRDatasetConfig, load_rlvr_dataset
from src.rl.generation_smoke import (
    GenerationSmokeConfig,
    _assistant_message,
    _build_generation_config,
    _generate_once,
    _initial_messages,
    _tool_observation_message,
    _trajectory_summary,
)
from src.rl.model_load_smoke import (
    ModelLoadSmokeConfig,
    _config_section,
    _load_model,
    _load_processor,
    _load_yaml_config,
    _write_json_no_overwrite,
    resolve_processor_path,
)
from src.rl.paths import ensure_output_dirs, resolve_agentic_paths
from src.rl.rewards import compute_format_reward, compute_total_reward, parse_json_action
from src.tools.zoom_tool import zoom_in


NUMERIC_METRICS = (
    "valid_json_rate",
    "extra_text_rate",
    "zoom_in_usage_rate",
    "final_answer_rate",
    "final_answer_valid_rate",
    "invalid_bbox_rate",
    "mean_reward",
    "mean_format_reward",
    "mean_tool_reward",
    "mean_correctness_reward",
    "avg_num_steps",
)


def _load_dataset(raw_config: dict[str, Any], cfg: ModelLoadSmokeConfig) -> Any:
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
        raise ValueError("RLVR dataset is empty; cannot run batch generation smoke")
    return dataset


def _select_indices(
    *,
    dataset_length: int,
    start_index: int,
    num_samples: int,
    sample_indices: list[int] | None,
) -> list[int]:
    if sample_indices is not None:
        indices = sample_indices
    else:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        indices = list(range(start_index, min(start_index + num_samples, dataset_length)))
    if not indices:
        raise ValueError("no sample indices selected")
    bad = [idx for idx in indices if idx < 0 or idx >= dataset_length]
    if bad:
        raise IndexError(f"sample indices out of range for dataset length {dataset_length}: {bad}")
    return indices


def _parse_sample_indices(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _run_one_loaded(
    *,
    raw_config: dict[str, Any],
    cfg: GenerationSmokeConfig,
    dataset: Any,
    sample_index: int,
    processor: Any,
    model: Any,
    processor_src: str,
    vision_info_fn: Callable[[list[dict[str, Any]]], tuple[Any, Any]] | None = None,
) -> dict[str, Any]:
    sample = dataset[sample_index]
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
    return {
        "sample_index": sample_index,
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
        "processor_path": processor_src,
        "trajectory": trajectory,
        "reward": reward,
        "input_infos": input_infos,
    }


def _compact_result(summary: dict[str, Any]) -> dict[str, Any]:
    steps = summary.get("trajectory", {}).get("steps") or []
    final_step = steps[-1] if steps else {}
    reward = summary.get("reward", {})
    return {
        "sample_index": summary.get("sample_index"),
        "sample": summary.get("sample", {}),
        "final_action": final_step.get("action"),
        "final_is_valid_json": bool(reward.get("is_valid_json")),
        "final_has_extra_text": bool(final_step.get("has_extra_text")),
        "used_zoom": bool(summary.get("trajectory", {}).get("used_zoom")),
        "invalid_bbox": bool(summary.get("trajectory", {}).get("invalid_bbox")),
        "num_steps": int(summary.get("trajectory", {}).get("num_steps", 0)),
        "reward": reward,
        "steps": steps,
    }


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _aggregate_metrics(results: list[dict[str, Any]], errors: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    rewards = [r.get("reward", {}) for r in results]
    return {
        "num_requested": n + len(errors),
        "num_completed": n,
        "num_errors": len(errors),
        "valid_json_rate": _mean([float(r.get("final_is_valid_json", False)) for r in results]),
        "extra_text_rate": _mean([float(r.get("final_has_extra_text", False)) for r in results]),
        "zoom_in_usage_rate": _mean([float(r.get("used_zoom", False)) for r in results]),
        "final_answer_rate": _mean([float(r.get("final_action") == "final_answer") for r in results]),
        "final_answer_valid_rate": _mean(
            [float(r.get("final_action") == "final_answer" and r.get("final_is_valid_json")) for r in results]
        ),
        "invalid_bbox_rate": _mean([float(r.get("invalid_bbox", False)) for r in results]),
        "mean_reward": _mean([float(reward.get("reward", 0.0)) for reward in rewards]),
        "mean_format_reward": _mean([float(reward.get("format_reward", 0.0)) for reward in rewards]),
        "mean_tool_reward": _mean([float(reward.get("tool_reward", 0.0)) for reward in rewards]),
        "mean_correctness_reward": _mean([float(reward.get("correctness_reward", 0.0)) for reward in rewards]),
        "avg_num_steps": _mean([float(r.get("num_steps", 0)) for r in results]),
    }


def _run_variant(
    *,
    name: str,
    peft_adapter_path: str | None,
    raw_config: dict[str, Any],
    cfg: GenerationSmokeConfig,
    dataset: Any,
    sample_indices: list[int],
    processor_loader: Callable[[str, ModelLoadSmokeConfig], Any] | None = None,
    model_loader: Callable[[str, ModelLoadSmokeConfig], Any] | None = None,
    vision_info_fn: Callable[[list[dict[str, Any]]], tuple[Any, Any]] | None = None,
) -> dict[str, Any]:
    model_cfg = replace(cfg.model, peft_adapter_path=peft_adapter_path)
    variant_cfg = replace(cfg, model=model_cfg)
    processor_src = resolve_processor_path(model_cfg.model_path, model_cfg.processor_path)
    processor = (processor_loader or _load_processor)(processor_src, model_cfg)
    model = (model_loader or _load_model)(model_cfg.model_path, model_cfg)
    if hasattr(model, "eval"):
        model.eval()
    if not variant_cfg.do_sample and hasattr(model, "generation_config"):
        model.generation_config.temperature = None

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    try:
        for sample_index in sample_indices:
            try:
                full = _run_one_loaded(
                    raw_config=raw_config,
                    cfg=variant_cfg,
                    dataset=dataset,
                    sample_index=sample_index,
                    processor=processor,
                    model=model,
                    processor_src=processor_src,
                    vision_info_fn=vision_info_fn,
                )
                results.append(_compact_result(full))
            except Exception as exc:  # keep the rest of the batch useful
                errors.append({"sample_index": sample_index, "error": type(exc).__name__, "message": str(exc)})
    finally:
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "name": name,
        "model_path": model_cfg.model_path,
        "peft_adapter_path": peft_adapter_path,
        "processor_path": processor_src,
        "metrics": _aggregate_metrics(results, errors),
        "errors": errors,
        "samples": results,
    }


def _comparison(variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    base = variants.get("base")
    adapter_name = next((name for name in variants if name != "base"), None)
    adapter = variants.get(adapter_name) if adapter_name else None
    if not base or not adapter or adapter_name is None:
        return {}
    base_metrics = base.get("metrics", {})
    adapter_metrics = adapter.get("metrics", {})
    delta = {}
    for key in NUMERIC_METRICS:
        if key in base_metrics and key in adapter_metrics:
            delta[key] = float(adapter_metrics[key]) - float(base_metrics[key])
    return {"adapter_variant": adapter_name, "adapter_minus_base": delta, f"{adapter_name}_minus_base": delta}


def _allow_large_images() -> None:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None


def run_generation_batch_smoke(
    *,
    config_path: str | None = None,
    output_root: str | None = None,
    model_path: str | None = None,
    peft_adapter_path: str | None = None,
    processor_path: str | None = None,
    split: str | None = None,
    start_index: int = 0,
    num_samples: int = 20,
    sample_indices: list[int] | None = None,
    max_new_tokens: int | None = None,
    skip_base: bool = False,
    adapter_name: str = "adapter",
    allow_large_images: bool = False,
    processor_loader: Callable[[str, ModelLoadSmokeConfig], Any] | None = None,
    model_loader: Callable[[str, ModelLoadSmokeConfig], Any] | None = None,
    vision_info_fn: Callable[[list[dict[str, Any]]], tuple[Any, Any]] | None = None,
) -> dict[str, Any]:
    """Run base and optional PEFT-adapter generation smoke over a shared sample set."""

    if output_root:
        os.environ["OUTPUT_ROOT"] = output_root
    if allow_large_images:
        _allow_large_images()
    raw_config = _load_yaml_config(config_path)
    cfg = _build_generation_config(
        raw_config,
        model_path=model_path,
        peft_adapter_path=None,
        processor_path=processor_path,
        split=split,
        sample_index=start_index,
        max_new_tokens=max_new_tokens,
    )
    paths = resolve_agentic_paths(create_output=True)
    ensure_output_dirs(paths)
    dataset = _load_dataset(raw_config, cfg.model)
    indices = _select_indices(
        dataset_length=len(dataset),
        start_index=start_index,
        num_samples=num_samples,
        sample_indices=sample_indices,
    )

    variant_specs: list[tuple[str, str | None]] = []
    if not skip_base:
        variant_specs.append(("base", None))
    if peft_adapter_path:
        variant_specs.append((adapter_name, peft_adapter_path))
    if not variant_specs:
        raise ValueError("no variants selected; provide --peft-adapter-path or omit --skip-base")

    variant_results = {}
    for name, adapter_path in variant_specs:
        variant_results[name] = _run_variant(
            name=name,
            peft_adapter_path=adapter_path,
            raw_config=raw_config,
            cfg=cfg,
            dataset=dataset,
            sample_indices=indices,
            processor_loader=processor_loader,
            model_loader=model_loader,
            vision_info_fn=vision_info_fn,
        )

    summary = {
        "generation_batch_smoke": True,
        "no_training": True,
        "checkpoint_written": False,
        "model_path": cfg.model.model_path,
        "peft_adapter_path": peft_adapter_path,
        "split": cfg.model.split,
        "dataset_length": len(dataset),
        "sample_indices": indices,
        "num_samples": len(indices),
        "config": {
            "max_new_tokens": cfg.max_new_tokens,
            "max_zoom_steps": cfg.max_zoom_steps,
            "coord_type": cfg.coord_type,
            "zoom_output_size": cfg.zoom_output_size,
            "strict_format": cfg.strict_format,
            "action_reminder": cfg.action_reminder,
            "prompt_mode": cfg.prompt_mode,
            "allow_large_images": allow_large_images,
        },
        "variants": variant_results,
        "comparison": _comparison(variant_results),
    }
    log_path = _write_json_no_overwrite(paths.output_root / "logs", "generation_batch_smoke.json", summary)
    summary["log_path"] = str(log_path)
    return summary


def _summary_view(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "generation_batch_smoke": summary.get("generation_batch_smoke"),
        "log_path": summary.get("log_path"),
        "model_path": summary.get("model_path"),
        "peft_adapter_path": summary.get("peft_adapter_path"),
        "split": summary.get("split"),
        "sample_indices": summary.get("sample_indices"),
        "num_samples": summary.get("num_samples"),
        "config": summary.get("config"),
        "variants": {
            name: {
                "model_path": variant.get("model_path"),
                "peft_adapter_path": variant.get("peft_adapter_path"),
                "metrics": variant.get("metrics"),
                "errors": variant.get("errors"),
            }
            for name, variant in (summary.get("variants") or {}).items()
        },
        "comparison": summary.get("comparison"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run batch Agentic GRPO generation smoke over base and PEFT adapter")
    parser.add_argument("--config", default="configs/agentic_grpo.yaml")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--peft-adapter-path", default=None)
    parser.add_argument("--adapter-name", default="adapter")
    parser.add_argument("--processor-path", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--sample-indices", default=None, help="Comma-separated sample indices; overrides start/num")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--allow-large-images", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args(argv)
    summary = run_generation_batch_smoke(
        config_path=args.config,
        output_root=args.output_root,
        model_path=args.model_path,
        peft_adapter_path=args.peft_adapter_path,
        adapter_name=args.adapter_name,
        processor_path=args.processor_path,
        split=args.split,
        start_index=args.start_index,
        num_samples=args.num_samples,
        sample_indices=_parse_sample_indices(args.sample_indices),
        max_new_tokens=args.max_new_tokens,
        skip_base=args.skip_base,
        allow_large_images=args.allow_large_images,
    )
    payload = _summary_view(summary) if args.summary_only else summary
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
