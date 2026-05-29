"""Dataset adapters for STAR/R1-SGG Agentic RLVR training data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from src.rl.paths import AgenticPaths, resolve_agentic_paths


@dataclass(frozen=True)
class RLVRDatasetConfig:
    dataset_path: str | None = None
    jsonl_dir: str | None = None
    split: str = "train"
    source: str = "hf_closed"
    task_type: str = "scene_graph"
    prompt_field: str = "prompt_close"
    input_style: str = "legacy"
    require_image_exists: bool = True


def _loads_if_json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value


def _xywh_to_xyxy(bbox: Sequence[float | int]) -> list[float]:
    x, y, w, h = (float(v) for v in bbox)
    return [x, y, x + w, y + h]


def build_eagle_grounding_prompt(
    base_prompt: str,
    *,
    width: int | None,
    height: int | None,
    task_type: str = "scene_graph",
) -> str:
    """Build an information-first high-resolution grounding prompt."""

    size_text = f"{width}x{height}" if width and height else "unknown size"
    return "\n".join(
        [
            "<image>",
            "You are analyzing one ultra-high-resolution remote-sensing image.",
            f"Image size: {size_text} pixels.",
            f"Task: {task_type}.",
            "",
            "Information-first policy:",
            "- First use the global image to identify likely objects and spatial layout.",
            "- If local evidence is unclear, call the zoom_in tool with a tight pixel-space bbox [x1, y1, x2, y2].",
            "- Do not use whole-image or tiny invalid zoom boxes.",
            "- Preserve object ids as class.index, for example ship.1 or harbor.2.",
            "- Use pixel-space xyxy boxes in the final scene graph.",
            "",
            "Action format:",
            '- Tool call: {"thought": "...", "action": "zoom_in", "bbox": [x1, y1, x2, y2]}',
            '- Final answer: {"thought": "...", "action": "final_answer", "answer": {"objects": [], "relationships": []}}',
            "",
            "Final answer schema:",
            '- objects: list of {"id": string, "bbox": [x1, y1, x2, y2]}',
            '- relationships: list of {"subject": string, "predicate": string, "object": string}',
            "",
            "Dataset instruction:",
            base_prompt.strip(),
        ]
    )


def build_eagle_conversations(prompt: str, answer: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return Eagle-style conversation turns for multimodal SFT/RLVR data."""

    conversations: list[dict[str, Any]] = [
        {
            "from": "system",
            "value": "You are a precise remote-sensing visual grounding and scene graph assistant.",
        },
        {"from": "human", "value": prompt},
    ]
    if answer is not None:
        conversations.append(
            {
                "from": "gpt",
                "value": json.dumps(
                    {"action": "final_answer", "answer": answer},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )
    return conversations


def build_eagle_data_entry(image_path: str) -> list[dict[str, Any]]:
    """Return Eagle-style interleaved data metadata for one image."""

    return [{"type": "image", "image_list": [image_path]}]


def _default_image_path(paths: AgenticPaths, split: str, image_id: int) -> Path:
    object_root = paths.star_raw_root / "STAR-object"
    split_roots = {
        "train": object_root / "train" / "trainimg正确",
        "val": object_root / "val" / "valimg正确",
        "test": object_root / "test" / "testimg正确",
    }
    candidates = [split_roots.get(split, split_roots["train"]) / f"{image_id:04d}.png"]
    candidates.extend(root / f"{image_id:04d}.png" for name, root in split_roots.items() if name != split)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0]


def _normalize_star_row(
    row: dict[str, Any],
    *,
    split: str,
    paths: AgenticPaths,
    prompt_field: str = "prompt_close",
    task_type: str = "scene_graph",
    input_style: str = "legacy",
    require_image_exists: bool = True,
) -> dict[str, Any]:
    image_id = int(row["image_id"]) if "image_id" in row else None
    objects = _loads_if_json(row.get("objects"), [])
    relationships = _loads_if_json(row.get("relationships", row.get("relations")), [])
    if not isinstance(objects, list):
        raise ValueError("objects must be a JSON list or list")
    if not isinstance(relationships, list):
        raise ValueError("relationships must be a JSON list or list")

    image = row.get("image") or row.get("image_path")
    if image is None and image_id is not None:
        image = str(_default_image_path(paths, split, image_id))
    if image is None:
        raise ValueError("row must contain image/image_path or image_id")
    image_path = Path(str(image))
    if require_image_exists and not image_path.is_file():
        raise FileNotFoundError(f"image file not found: {image_path}")

    base_prompt = row.get(prompt_field) or row.get("prompt") or row.get("question") or "Generate the scene graph for this remote sensing image."
    sample_id = row.get("id") or (f"star_{split}_{image_id}" if image_id is not None else None)
    answer = {
        "objects": objects,
        "relationships": relationships,
    }
    width = int(row["width"]) if row.get("width") is not None else None
    height = int(row["height"]) if row.get("height") is not None else None
    if input_style == "legacy":
        prompt = base_prompt
        conversations = build_eagle_conversations(str(prompt), answer=None)
    elif input_style in {"eagle", "eagle_grounding", "information_first"}:
        prompt = build_eagle_grounding_prompt(
            str(base_prompt),
            width=width,
            height=height,
            task_type=task_type,
        )
        conversations = build_eagle_conversations(prompt, answer=answer)
    else:
        raise ValueError(f"unknown input_style: {input_style}")
    return {
        "id": sample_id,
        "image_id": image_id,
        "task_type": task_type,
        "image": str(image_path),
        "prompt": prompt,
        "base_prompt": base_prompt,
        "prompt_close": row.get("prompt_close"),
        "width": width,
        "height": height,
        "objects": objects,
        "relationships": relationships,
        "answer": answer,
        "conversations": conversations,
        "data": build_eagle_data_entry(str(image_path)),
        "input_style": input_style,
        "split": split,
    }


class JsonlRLVRDataset:
    """Simple JSONL-backed STAR/RLVR dataset."""

    def __init__(
        self,
        jsonl_path: str | Path,
        *,
        split: str,
        paths: AgenticPaths | None = None,
        prompt_field: str = "prompt_close",
        task_type: str = "scene_graph",
        input_style: str = "legacy",
        require_image_exists: bool = True,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.split = split
        self.paths = paths or resolve_agentic_paths()
        self.prompt_field = prompt_field
        self.task_type = task_type
        self.input_style = input_style
        self.require_image_exists = require_image_exists
        self._rows = self._load_rows()

    def _load_rows(self) -> list[dict[str, Any]]:
        if not self.jsonl_path.is_file():
            raise FileNotFoundError(f"jsonl file not found: {self.jsonl_path}")
        rows = []
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {self.jsonl_path}:{line_no}: {exc}") from exc
                rows.append(
                    _normalize_star_row(
                        row,
                        split=self.split,
                        paths=self.paths,
                        prompt_field=self.prompt_field,
                        task_type=self.task_type,
                        input_style=self.input_style,
                        require_image_exists=self.require_image_exists,
                    )
                )
        return rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._rows[index]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._rows)


class HFRLVRDataset:
    """HuggingFace load_from_disk-backed STAR/RLVR dataset."""

    def __init__(
        self,
        dataset_path: str | Path,
        *,
        split: str,
        paths: AgenticPaths | None = None,
        prompt_field: str = "prompt_close",
        task_type: str = "scene_graph",
        input_style: str = "legacy",
        require_image_exists: bool = True,
    ) -> None:
        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise ImportError("datasets is required to load HF RLVR datasets") from exc
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"dataset path not found: {self.dataset_path}")
        self.split = split
        self.paths = paths or resolve_agentic_paths()
        ds = load_from_disk(str(self.dataset_path))
        if split not in ds:
            raise KeyError(f"split {split!r} not found in {self.dataset_path}")
        self._dataset = ds[split]
        self.prompt_field = prompt_field
        self.task_type = task_type
        self.input_style = input_style
        self.require_image_exists = require_image_exists

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return _normalize_star_row(
            dict(self._dataset[index]),
            split=self.split,
            paths=self.paths,
            prompt_field=self.prompt_field,
            task_type=self.task_type,
            input_style=self.input_style,
            require_image_exists=self.require_image_exists,
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]


def load_rlvr_dataset(config: RLVRDatasetConfig | dict[str, Any] | None = None, *, paths: AgenticPaths | None = None):
    """Load the configured RLVR dataset adapter."""

    cfg = config if isinstance(config, RLVRDatasetConfig) else RLVRDatasetConfig(**(config or {}))
    resolved_paths = paths or resolve_agentic_paths()
    if cfg.source in {"hf", "hf_closed", "star_r1sgg_hf_closed"}:
        dataset_path = Path(cfg.dataset_path) if cfg.dataset_path else resolved_paths.dataset_path
        return HFRLVRDataset(
            dataset_path,
            split=cfg.split,
            paths=resolved_paths,
            prompt_field=cfg.prompt_field,
            task_type=cfg.task_type,
            input_style=cfg.input_style,
            require_image_exists=cfg.require_image_exists,
        )
    if cfg.source in {"jsonl", "jsonl_closed", "star_r1sgg_jsonl_closed"}:
        jsonl_dir = Path(cfg.jsonl_dir) if cfg.jsonl_dir else resolved_paths.jsonl_closed_dir
        return JsonlRLVRDataset(
            jsonl_dir / f"{cfg.split}.jsonl",
            split=cfg.split,
            paths=resolved_paths,
            prompt_field=cfg.prompt_field,
            task_type=cfg.task_type,
            input_style=cfg.input_style,
            require_image_exists=cfg.require_image_exists,
        )
    raise ValueError(f"unknown RLVR dataset source: {cfg.source}")


def summarize_rlvr_dataset(dataset: Any, max_items: int = 3) -> dict[str, Any]:
    """Return a small adapter summary without materializing large images."""

    examples = []
    for idx, sample in enumerate(dataset):
        if idx >= max_items:
            break
        examples.append(
            {
                "id": sample.get("id"),
                "image_id": sample.get("image_id"),
                "image": sample.get("image"),
                "width": sample.get("width"),
                "height": sample.get("height"),
                "num_objects": len(sample.get("objects") or []),
                "num_relationships": len(sample.get("relationships") or []),
                "task_type": sample.get("task_type"),
                "input_style": sample.get("input_style"),
            }
        )
    return {"num_rows": len(dataset), "examples": examples}


__all__ = [
    "HFRLVRDataset",
    "JsonlRLVRDataset",
    "RLVRDatasetConfig",
    "build_eagle_conversations",
    "build_eagle_data_entry",
    "build_eagle_grounding_prompt",
    "load_rlvr_dataset",
    "summarize_rlvr_dataset",
]
