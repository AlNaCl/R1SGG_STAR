"""Model/processor loading smoke test for Agentic GRPO RLVR data."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch

from src.data.rlvr_dataset import RLVRDatasetConfig, load_rlvr_dataset
from src.rl.grpo_trainer import _load_yaml_config
from src.rl.paths import ensure_output_dirs, resolve_agentic_paths

DEFAULT_MODEL_PATH = "/root/autodl-tmp/STAR/r1sgg_data/checkpoints/qwen25vl-7b-sft-star-close-20260507_182608Z"


@dataclass(frozen=True)
class ModelLoadSmokeConfig:
    """Configuration for a no-training model/processor smoke test."""

    model_path: str = DEFAULT_MODEL_PATH
    processor_path: str | None = None
    split: str = "val"
    sample_index: int = 0
    load_model: bool = True
    build_inputs: bool = True
    attn_implementation: str = "sdpa"
    torch_dtype: str = "bfloat16"
    device_map: str = "cuda"
    max_pixels: int | None = None
    min_pixels: int | None = None


def _config_section(raw_config: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw_config.get(key, {}) if isinstance(raw_config, dict) else {}
    return value if isinstance(value, dict) else {}


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


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


def _torch_dtype(name: str | None) -> torch.dtype | None:
    if not name or name == "auto":
        return None
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name not in aliases:
        raise ValueError(f"unsupported torch dtype: {name}")
    return aliases[name]


def _first_snapshot_with_preprocessor(hub_model_dir: Path) -> str | None:
    snap_root = hub_model_dir / "snapshots"
    if not snap_root.is_dir():
        return None
    candidates = [p for p in snap_root.iterdir() if p.is_dir() and (p / "preprocessor_config.json").is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(candidates[0])


def _infer_base_repo_id(model_path: str) -> str:
    lower = model_path.lower()
    cfg_path = Path(model_path) / "config.json"
    model_type = ""
    if cfg_path.is_file():
        try:
            model_type = str(json.loads(cfg_path.read_text(encoding="utf-8")).get("model_type", ""))
        except json.JSONDecodeError:
            model_type = ""
    joined = f"{lower} {model_type}".lower()
    if "qwen2_5_vl" in joined or "qwen2.5" in joined or "qwen25" in joined or "qwen2-5" in joined:
        if "3b" in joined:
            return "Qwen/Qwen2.5-VL-3B-Instruct"
        return "Qwen/Qwen2.5-VL-7B-Instruct"
    if "7b" in joined:
        return "Qwen/Qwen2-VL-7B-Instruct"
    return "Qwen/Qwen2-VL-2B-Instruct"


def _hub_cache_dirs(repo_id: str) -> list[Path]:
    hub_name = "models--" + repo_id.replace("/", "--")
    dirs = []
    for root in (os.environ.get("HF_HOME"), os.environ.get("TRANSFORMERS_CACHE")):
        if root:
            dirs.append(Path(root) / "hub" / hub_name)
            dirs.append(Path(root) / hub_name)
    dirs.append(Path.home() / ".cache" / "huggingface" / "hub" / hub_name)
    return dirs


def resolve_processor_path(model_path: str, explicit: str | None = None) -> str:
    """Resolve a Qwen-VL processor path without requiring network access first."""

    if explicit:
        return explicit
    model_dir = Path(model_path)
    if (model_dir / "preprocessor_config.json").is_file():
        return str(model_dir.resolve())
    env_path = os.environ.get("STAR_INFER_PROCESSOR_PATH") or os.environ.get("PROCESSOR_PATH")
    if env_path and (Path(env_path) / "preprocessor_config.json").is_file():
        return str(Path(env_path).resolve())
    repo_id = _infer_base_repo_id(model_path)
    for hub_dir in _hub_cache_dirs(repo_id):
        resolved = _first_snapshot_with_preprocessor(hub_dir)
        if resolved:
            return resolved
    return repo_id


def sample_to_qwen_messages(sample: dict[str, Any], *, add_system: bool = True) -> list[dict[str, Any]]:
    """Convert one RLVR sample into Qwen-VL chat messages for inference/RL rollout."""

    messages: list[dict[str, Any]] = []
    if add_system:
        messages.append(
            {
                "role": "system",
                "content": "You are a precise remote-sensing visual grounding and scene graph assistant.",
            }
        )
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": sample["prompt"]},
            ],
        }
    )
    return messages


def build_qwen_processor_inputs(
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    vision_info_fn: Callable[[list[dict[str, Any]]], tuple[Any, Any]] | None = None,
) -> dict[str, Any]:
    """Apply Qwen chat template and processor to one multimodal message."""

    if vision_info_fn is None:
        from qwen_vl_utils import process_vision_info

        vision_info_fn = process_vision_info
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = vision_info_fn(messages)
    batch = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt")
    return {"text": text, "images": images, "videos": videos, "batch": batch}


def _load_processor(processor_path: str, cfg: ModelLoadSmokeConfig) -> Any:
    from transformers import AutoProcessor

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if cfg.max_pixels is not None:
        kwargs["max_pixels"] = cfg.max_pixels
    if cfg.min_pixels is not None:
        kwargs["min_pixels"] = cfg.min_pixels
    path_obj = Path(processor_path)
    if path_obj.exists() or processor_path.startswith("/"):
        kwargs["local_files_only"] = True
    return AutoProcessor.from_pretrained(processor_path, **kwargs)


def _load_model(model_path: str, cfg: ModelLoadSmokeConfig) -> Any:
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "qwen_vl_model_load.py"
    spec = importlib.util.spec_from_file_location("qwen_vl_model_load", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.load_qwen_vl_for_inference(
        model_path,
        attn_implementation=cfg.attn_implementation,
        torch_dtype=_torch_dtype(cfg.torch_dtype),
        device_map=cfg.device_map,
        trust_remote_code=True,
    )
    if hasattr(model, "eval"):
        model.eval()
    return model


def _shape_summary(value: Any) -> Any:
    if hasattr(value, "shape"):
        return list(value.shape)
    if isinstance(value, Mapping):
        return {k: _shape_summary(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_shape_summary(v) for v in value]
    return type(value).__name__


def _non_overwriting_json_path(log_dir: Path, name: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem
    suffix = Path(name).suffix
    candidate = log_dir / name
    if not candidate.exists():
        return candidate
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%fZ")
    for idx in range(1000):
        extra = f"_{stamp}" if idx == 0 else f"_{stamp}_{idx}"
        candidate = log_dir / f"{stem}{extra}{suffix}"
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"could not find non-overwriting log path for {log_dir / name}")


def _write_json_no_overwrite(log_dir: Path, name: str, payload: dict[str, Any]) -> Path:
    path = _non_overwriting_json_path(log_dir, name)
    with path.open("x", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def _build_smoke_config(
    raw_config: dict[str, Any],
    *,
    model_path: str | None = None,
    processor_path: str | None = None,
    load_model: bool | None = None,
    sample_index: int | None = None,
    split: str | None = None,
) -> ModelLoadSmokeConfig:
    model_raw = _config_section(raw_config, "model")
    smoke_raw = _config_section(raw_config, "model_load_smoke")
    return ModelLoadSmokeConfig(
        model_path=model_path or str(model_raw.get("name_or_path") or DEFAULT_MODEL_PATH),
        processor_path=processor_path or _optional_str(model_raw.get("processor_path")),
        split=split or str(smoke_raw.get("split", "val")),
        sample_index=sample_index if sample_index is not None else int(smoke_raw.get("sample_index", 0)),
        load_model=load_model if load_model is not None else _bool_value(smoke_raw.get("load_model"), True),
        build_inputs=_bool_value(smoke_raw.get("build_inputs"), True),
        attn_implementation=str(model_raw.get("attn_implementation", "sdpa")),
        torch_dtype=str(model_raw.get("torch_dtype", "bfloat16")),
        device_map=str(model_raw.get("device_map", "cuda")),
        max_pixels=_optional_int(smoke_raw.get("max_pixels", model_raw.get("max_pixels"))),
        min_pixels=_optional_int(smoke_raw.get("min_pixels", model_raw.get("min_pixels"))),
    )


def run_model_load_smoke(
    *,
    config_path: str | None = None,
    output_root: str | None = None,
    model_path: str | None = None,
    processor_path: str | None = None,
    load_model: bool | None = None,
    sample_index: int | None = None,
    split: str | None = None,
    processor_loader: Callable[[str, ModelLoadSmokeConfig], Any] | None = None,
    model_loader: Callable[[str, ModelLoadSmokeConfig], Any] | None = None,
    vision_info_fn: Callable[[list[dict[str, Any]]], tuple[Any, Any]] | None = None,
) -> dict[str, Any]:
    """Load one RLVR sample, build Qwen-VL inputs, and optionally load the model."""

    if output_root:
        os.environ["OUTPUT_ROOT"] = output_root
    raw_config = _load_yaml_config(config_path)
    cfg = _build_smoke_config(
        raw_config,
        model_path=model_path,
        processor_path=processor_path,
        load_model=load_model,
        sample_index=sample_index,
        split=split,
    )
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
        raise ValueError("RLVR dataset is empty; cannot run model-load smoke")
    if cfg.sample_index < 0 or cfg.sample_index >= len(dataset):
        raise IndexError(f"sample_index {cfg.sample_index} out of range for dataset length {len(dataset)}")
    sample = dataset[cfg.sample_index]
    messages = sample_to_qwen_messages(sample)

    processor_src = resolve_processor_path(cfg.model_path, cfg.processor_path)
    processor_loader = processor_loader or _load_processor
    processor = processor_loader(processor_src, cfg)

    input_summary: dict[str, Any] = {}
    if cfg.build_inputs:
        prepared = build_qwen_processor_inputs(processor, messages, vision_info_fn=vision_info_fn)
        batch = prepared["batch"]
        input_summary = {
            "chat_text_chars": len(prepared["text"]),
            "num_images": len(prepared["images"] or []),
            "num_videos": len(prepared["videos"] or []),
            "batch_shapes": _shape_summary(batch),
        }

    model_loaded = False
    model_summary: dict[str, Any] = {}
    if cfg.load_model:
        model_loader = model_loader or _load_model
        model = model_loader(cfg.model_path, cfg)
        model_loaded = True
        model_config = getattr(model, "config", None)
        model_summary = {
            "model_class": type(model).__name__,
            "model_type": getattr(model_config, "model_type", None),
        }
        if torch.cuda.is_available():
            model_summary["cuda_memory_allocated_bytes"] = int(torch.cuda.memory_allocated())
            model_summary["cuda_memory_reserved_bytes"] = int(torch.cuda.memory_reserved())

    summary = {
        "model_load_smoke": True,
        "model_loaded": model_loaded,
        "built_processor_inputs": bool(input_summary),
        "model_path": cfg.model_path,
        "processor_path": processor_src,
        "split": cfg.split,
        "sample_index": cfg.sample_index,
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
        "messages": {
            "num_turns": len(messages),
            "roles": [m.get("role") for m in messages],
            "user_content_types": [c.get("type") for c in messages[-1].get("content", [])],
        },
        "inputs": input_summary,
        "model": model_summary,
        "no_training": True,
        "checkpoint_written": False,
    }
    log_path = _write_json_no_overwrite(paths.output_root / "logs", "model_load_smoke.json", summary)
    summary["log_path"] = str(log_path)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Load Qwen-VL processor/model against one Agentic RLVR sample without training")
    parser.add_argument("--config", default="configs/agentic_grpo.yaml")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--processor-path", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--skip-model-load", action="store_true", help="Only load processor and build inputs")
    args = parser.parse_args(argv)
    summary = run_model_load_smoke(
        config_path=args.config,
        output_root=args.output_root,
        model_path=args.model_path,
        processor_path=args.processor_path,
        load_model=False if args.skip_model_load else None,
        sample_index=args.sample_index,
        split=args.split,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
