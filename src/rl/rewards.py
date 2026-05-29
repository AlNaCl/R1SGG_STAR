"""Verifiable rewards for Agentic GRPO / RLVR rollouts."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any


VALID_ACTIONS = {"zoom_in", "final_answer"}


@dataclass(frozen=True)
class ParsedAction:
    data: dict[str, Any] | None
    is_valid_json: bool
    has_extra_text: bool


def normalize_text(value: Any) -> str:
    """Normalize short text answers for exact-match style rewards."""

    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = text.strip(" .,:;!?")
    return text


def parse_json_action(text: str, *, strict: bool = True) -> ParsedAction:
    """Parse a JSON action, optionally rejecting extra text around the JSON."""

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    decoder = json.JSONDecoder()
    try:
        data, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError:
        return ParsedAction(None, False, bool(stripped))
    has_extra = bool(stripped[end:].strip())
    if strict and has_extra:
        return ParsedAction(data if isinstance(data, dict) else None, False, True)
    return ParsedAction(data if isinstance(data, dict) else None, isinstance(data, dict), has_extra)


def extract_answer(prediction: str) -> Any:
    """Extract final answer text from JSON action or answer tags."""

    parsed = parse_json_action(prediction, strict=False)
    if parsed.data and parsed.data.get("action") == "final_answer":
        return parsed.data.get("answer", "")
    match = re.search(r"<answer>(.*?)</answer>", prediction, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return prediction.strip()


def _sample_answer(sample: dict[str, Any]) -> Any:
    for key in ("answer", "label", "target", "ground_truth", "solution"):
        if key in sample:
            return sample[key]
    return ""


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", str(value))
    return float(match.group(0)) if match else None


def _text_or_numeric_match(prediction: Any, target: Any, tolerance: float) -> bool:
    pred_num = _as_float(prediction)
    target_num = _as_float(target)
    if pred_num is not None and target_num is not None:
        return abs(pred_num - target_num) <= tolerance
    return normalize_text(prediction) == normalize_text(target)


def _scene_graph_triplets(graph: Any) -> set[tuple[str, str, str]]:
    if isinstance(graph, str):
        try:
            graph = json.loads(graph)
        except json.JSONDecodeError:
            return set()
    if not isinstance(graph, dict):
        return set()
    rels = graph.get("relationships") or []
    if isinstance(rels, str):
        try:
            rels = json.loads(rels)
        except json.JSONDecodeError:
            return set()
    triplets = set()
    for rel in rels:
        if not isinstance(rel, dict):
            continue
        triplets.add(
            (
                normalize_text(rel.get("subject", "")),
                normalize_text(rel.get("predicate", "")),
                normalize_text(rel.get("object", "")),
            )
        )
    return {t for t in triplets if all(t)}


def _extract_scene_graph_prediction(prediction: str) -> dict[str, Any] | None:
    parsed = parse_json_action(prediction, strict=False)
    if parsed.data and parsed.data.get("action") == "final_answer":
        answer = parsed.data.get("answer")
        if isinstance(answer, dict):
            return answer
        if isinstance(answer, str):
            try:
                loaded = json.loads(answer)
                return loaded if isinstance(loaded, dict) else None
            except json.JSONDecodeError:
                return None
    return parsed.data


def compute_correctness_reward(
    prediction: str,
    sample: dict[str, Any],
    *,
    numeric_tolerance: float = 1e-3,
    scene_graph_triplet_threshold: float = 1.0,
) -> float:
    """Return a binary task correctness reward."""

    task_type = str(sample.get("task_type", sample.get("type", "open_qa"))).lower()
    answer = extract_answer(prediction)
    target = _sample_answer(sample)

    if task_type in {"multiple_choice", "true_false"}:
        return float(normalize_text(answer) == normalize_text(target))
    if task_type in {"fill_blank", "numeric"}:
        return float(_text_or_numeric_match(answer, target, numeric_tolerance))
    if task_type == "open_qa":
        keywords = sample.get("keywords") or sample.get("acceptable_answers")
        if keywords:
            pred_norm = normalize_text(answer)
            return float(any(normalize_text(item) in pred_norm for item in keywords))
        return float(normalize_text(answer) == normalize_text(target))
    if task_type == "scene_graph":
        pred_graph = _extract_scene_graph_prediction(prediction)
        gt_graph = target if isinstance(target, dict) else sample
        pred_triplets = _scene_graph_triplets(pred_graph)
        gt_triplets = _scene_graph_triplets(gt_graph)
        if not gt_triplets:
            return 1.0 if not pred_triplets else 0.0
        recall = len(pred_triplets & gt_triplets) / len(gt_triplets)
        return float(recall >= scene_graph_triplet_threshold)
    return 0.0


def _valid_bbox(
    bbox: Any,
    image_size: tuple[int | float, int | float] | None = None,
) -> tuple[bool, bool, bool]:
    """Return (valid, is_whole_image, is_tiny)."""

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False, False, False
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return False, False, False
    if x2 <= x1 or y2 <= y1:
        return False, False, False
    width = height = None
    if image_size is not None:
        width, height = float(image_size[0]), float(image_size[1])
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            return False, False, False
    area = (x2 - x1) * (y2 - y1)
    whole = False
    tiny = area <= 4.0
    if width and height:
        image_area = width * height
        whole = area >= 0.9 * image_area
        tiny = area <= 0.0001 * image_area
    return True, whole, tiny


def compute_format_reward(prediction: str, *, strict: bool = True) -> float:
    """Score whether a model response is a valid action JSON."""

    parsed = parse_json_action(prediction, strict=strict)
    if not parsed.is_valid_json or parsed.data is None:
        return 0.0
    action = parsed.data.get("action")
    if action not in VALID_ACTIONS:
        return 0.25
    if action == "zoom_in":
        valid, _, _ = _valid_bbox(parsed.data.get("bbox"))
        return 1.0 if valid else 0.5
    answer = parsed.data.get("answer")
    if isinstance(answer, str):
        return 1.0 if answer.strip() else 0.5
    return 1.0 if answer is not None else 0.5


def _iter_tool_calls(trajectory: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not trajectory:
        return []
    for key in ("tool_calls", "actions", "steps"):
        calls = trajectory.get(key)
        if isinstance(calls, list):
            result = []
            for item in calls:
                if isinstance(item, dict) and "action" in item:
                    result.append(item)
                elif isinstance(item, dict) and isinstance(item.get("action_json"), dict):
                    result.append(item["action_json"])
                elif isinstance(item, dict) and isinstance(item.get("model_output"), str):
                    parsed = parse_json_action(item["model_output"], strict=False)
                    if parsed.data:
                        result.append(parsed.data)
            return result
    return []


def _bbox_iou(a: Any, b: Any) -> float:
    valid_a, _, _ = _valid_bbox(a)
    valid_b, _, _ = _valid_bbox(b)
    if not valid_a or not valid_b:
        return 0.0
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def compute_tool_reward(
    trajectory: dict[str, Any] | None,
    sample: dict[str, Any],
    *,
    max_tool_steps: int = 2,
) -> float:
    """Score whether zoom-in tool usage is valid and useful."""

    calls = [c for c in _iter_tool_calls(trajectory) if c.get("action") == "zoom_in"]
    if not calls:
        return 0.0

    image_size = None
    if sample.get("width") and sample.get("height"):
        image_size = (sample["width"], sample["height"])
    evidence_bbox = sample.get("evidence_bbox") or sample.get("target_bbox")

    valid_scores = []
    bboxes = []
    for call in calls:
        bbox = call.get("bbox")
        valid, whole, tiny = _valid_bbox(bbox, image_size)
        if not valid:
            valid_scores.append(0.0)
            continue
        bboxes.append(bbox)
        score = 1.0
        if whole:
            score -= 0.4
        if tiny:
            score -= 0.3
        if evidence_bbox is not None:
            score = max(0.0, score) * min(1.0, _bbox_iou(bbox, evidence_bbox) * 2.0)
        valid_scores.append(max(0.0, score))

    if not valid_scores:
        return 0.0
    reward = sum(valid_scores) / len(valid_scores)
    if len(calls) > max_tool_steps:
        reward *= max(0.0, 1.0 - 0.25 * (len(calls) - max_tool_steps))
    repeated = 0
    for i, bbox in enumerate(bboxes):
        for other in bboxes[:i]:
            if _bbox_iou(bbox, other) >= 0.95:
                repeated += 1
                break
    if repeated:
        reward *= max(0.0, 1.0 - 0.25 * repeated)
    return max(0.0, min(1.0, reward))


def compute_total_reward(
    prediction: str,
    sample: dict[str, Any],
    trajectory: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute correctness, format, tool, and weighted total reward."""

    cfg = config or {}
    correctness = compute_correctness_reward(
        prediction,
        sample,
        numeric_tolerance=float(cfg.get("numeric_tolerance", 1e-3)),
        scene_graph_triplet_threshold=float(cfg.get("scene_graph_triplet_threshold", 1.0)),
    )
    format_reward = compute_format_reward(
        prediction,
        strict=bool(cfg.get("strict_format", True)),
    )
    tool_reward = compute_tool_reward(
        trajectory,
        sample,
        max_tool_steps=int(cfg.get("max_tool_steps", 2)),
    )
    lambda_format = float(cfg.get("lambda_format", 0.1))
    lambda_tool = float(cfg.get("lambda_tool", 0.2))
    total = correctness + lambda_format * format_reward
    if correctness > 0:
        total += lambda_tool * tool_reward

    parsed = parse_json_action(prediction, strict=bool(cfg.get("strict_format", True)))
    calls = _iter_tool_calls(trajectory)
    image_size = (sample["width"], sample["height"]) if sample.get("width") and sample.get("height") else None
    invalid_bbox = any(
        call.get("action") == "zoom_in" and not _valid_bbox(call.get("bbox"), image_size)[0]
        for call in calls
    )
    return {
        "reward": float(total),
        "correctness_reward": float(correctness),
        "format_reward": float(format_reward),
        "tool_reward": float(tool_reward),
        "is_correct": bool(correctness > 0),
        "is_valid_json": bool(parsed.is_valid_json),
        "used_zoom": any(call.get("action") == "zoom_in" for call in calls),
        "invalid_bbox": bool(invalid_bbox),
    }
