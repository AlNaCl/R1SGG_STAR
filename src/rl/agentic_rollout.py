"""Agentic rollout loop for zoom-in RLVR experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from src.rl.mask_utils import TokenSpan
from src.rl.rewards import compute_total_reward, parse_json_action
from src.tools.zoom_tool import ZoomObservation, zoom_in


class ActionGenerator(Protocol):
    """Minimal model interface used by the rollout loop."""

    def __call__(self, history: list[dict[str, Any]], sample: dict[str, Any], step: int) -> str:
        ...


@dataclass(frozen=True)
class RolloutConfig:
    max_tool_steps: int = 2
    coord_type: str = "pixel"
    zoom_output_size: int | tuple[int, int] | None = None
    zoom_padding: int = 0
    min_bbox_area_ratio: float = 0.0
    max_bbox_area_ratio: float = 1.0
    stop_on_invalid_action: bool = True
    reward_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutStep:
    step: int
    model_output: str
    action: str | None
    action_json: dict[str, Any] | None
    observation: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class RolloutResult:
    trajectory: dict[str, Any]
    reward: dict[str, Any]
    token_spans: list[TokenSpan]
    final_prediction: str
    history: list[dict[str, Any]]


def _sample_image(sample: dict[str, Any]) -> Any:
    image = sample.get("image") or sample.get("image_path")
    if image is None:
        raise ValueError("sample must contain image or image_path for zoom-in rollout")
    return image


def _initial_history(sample: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = sample.get("prompt") or sample.get("prompt_close") or ""
    history = [{"role": "prompt", "content": prompt}]
    if sample.get("image") or sample.get("image_path"):
        history.append({"role": "image", "content": sample.get("image") or sample.get("image_path")})
    return history


def _append_span(spans: list[TokenSpan], role: str, token_count: int) -> None:
    start = spans[-1].end if spans else 0
    spans.append(TokenSpan(start, start + max(0, int(token_count)), role))


def _rough_token_count(text: str) -> int:
    return max(1, len(text.split())) if text else 0


def _observation_from_zoom(obs: ZoomObservation) -> dict[str, Any]:
    meta = obs.metadata()
    meta["has_image"] = obs.image is not None
    return meta


def run_agentic_rollout(
    sample: dict[str, Any],
    generator: ActionGenerator | Callable[[list[dict[str, Any]], dict[str, Any], int], str],
    config: RolloutConfig | dict[str, Any] | None = None,
) -> RolloutResult:
    """Run one agentic rollout with JSON actions and optional zoom-in calls."""

    cfg = config if isinstance(config, RolloutConfig) else RolloutConfig(**(config or {}))
    history = _initial_history(sample)
    token_spans: list[TokenSpan] = []
    _append_span(token_spans, "prompt", _rough_token_count(str(history[0].get("content", ""))))
    if len(history) > 1:
        _append_span(token_spans, "image", 1)

    steps: list[RolloutStep] = []
    final_prediction = ""
    invalid_bbox = False

    for step_idx in range(cfg.max_tool_steps + 1):
        model_output = generator(history, sample, step_idx)
        final_prediction = model_output
        _append_span(token_spans, "model_generation", _rough_token_count(model_output))
        parsed = parse_json_action(model_output, strict=True)
        if not parsed.is_valid_json or parsed.data is None:
            steps.append(
                RolloutStep(
                    step=step_idx,
                    model_output=model_output,
                    action=None,
                    action_json=None,
                    error="invalid_json",
                )
            )
            break

        action = parsed.data.get("action")
        if action == "final_answer":
            steps.append(RolloutStep(step_idx, model_output, action, parsed.data))
            break
        if action != "zoom_in":
            steps.append(
                RolloutStep(
                    step=step_idx,
                    model_output=model_output,
                    action=action,
                    action_json=parsed.data,
                    error="invalid_action",
                )
            )
            if cfg.stop_on_invalid_action:
                break
            continue

        if step_idx >= cfg.max_tool_steps:
            steps.append(
                RolloutStep(
                    step=step_idx,
                    model_output=model_output,
                    action=action,
                    action_json=parsed.data,
                    error="max_tool_steps_exceeded",
                )
            )
            break

        try:
            obs = zoom_in(
                _sample_image(sample),
                parsed.data.get("bbox"),
                coord_type=cfg.coord_type,  # type: ignore[arg-type]
                output_size=cfg.zoom_output_size,
                padding=cfg.zoom_padding,
                min_bbox_area_ratio=cfg.min_bbox_area_ratio,
                max_bbox_area_ratio=cfg.max_bbox_area_ratio,
            )
            observation = _observation_from_zoom(obs)
            if not obs.valid:
                invalid_bbox = True
            history.append({"role": "tool_observation", "content": observation, "image": obs.image})
            _append_span(token_spans, "tool_observation", 1)
            steps.append(RolloutStep(step_idx, model_output, action, parsed.data, observation=observation))
            if not obs.valid and cfg.stop_on_invalid_action:
                break
        except Exception as exc:
            invalid_bbox = True
            observation = {"valid": False, "error": str(exc)}
            history.append({"role": "tool_observation", "content": observation, "image": None})
            _append_span(token_spans, "tool_observation", 1)
            steps.append(
                RolloutStep(
                    step=step_idx,
                    model_output=model_output,
                    action=action,
                    action_json=parsed.data,
                    observation=observation,
                    error="tool_error",
                )
            )
            if cfg.stop_on_invalid_action:
                break

    trajectory = {
        "steps": [step.__dict__ for step in steps],
        "tool_calls": [step.action_json for step in steps if step.action_json and step.action == "zoom_in"],
        "used_zoom": any(step.action == "zoom_in" for step in steps),
        "invalid_bbox": invalid_bbox,
        "num_steps": len(steps),
    }
    reward = compute_total_reward(final_prediction, sample, trajectory, cfg.reward_config | {"max_tool_steps": cfg.max_tool_steps})
    if invalid_bbox:
        reward["invalid_bbox"] = True
    return RolloutResult(
        trajectory=trajectory,
        reward=reward,
        token_spans=token_spans,
        final_prediction=final_prediction,
        history=history,
    )
