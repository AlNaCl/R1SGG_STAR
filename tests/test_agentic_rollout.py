from PIL import Image

from src.rl.agentic_rollout import RolloutConfig, run_agentic_rollout
from src.rl.mask_utils import build_loss_mask
from src.tools.zoom_tool import zoom_in


class SequenceGenerator:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def __call__(self, history, sample, step):
        return self.outputs[step]


def test_rollout_stops_on_final_answer_and_computes_reward():
    sample = {"prompt": "answer yes or no", "task_type": "true_false", "answer": "yes"}
    generator = SequenceGenerator(['{"action": "final_answer", "answer": "yes"}'])

    result = run_agentic_rollout(sample, generator, RolloutConfig(max_tool_steps=2))

    assert result.final_prediction == '{"action": "final_answer", "answer": "yes"}'
    assert result.trajectory["num_steps"] == 1
    assert result.reward["is_correct"] is True
    assert result.reward["used_zoom"] is False


def test_rollout_calls_zoom_and_then_final_answer():
    image = Image.new("RGB", (100, 80), "white")
    sample = {
        "prompt": "inspect then answer",
        "image": image,
        "width": 100,
        "height": 80,
        "task_type": "true_false",
        "answer": "yes",
    }
    generator = SequenceGenerator(
        [
            '{"thought": "look", "action": "zoom_in", "bbox": [10, 20, 40, 50]}',
            '{"action": "final_answer", "answer": "yes"}',
        ]
    )

    result = run_agentic_rollout(sample, generator, RolloutConfig(max_tool_steps=2))

    assert result.trajectory["used_zoom"] is True
    assert result.trajectory["tool_calls"][0]["bbox"] == [10, 20, 40, 50]
    assert result.trajectory["steps"][0]["observation"]["bbox_xyxy"] == [10, 20, 40, 50]
    assert result.history[-1]["role"] == "tool_observation"
    assert result.history[-1]["image"].size == (30, 30)
    assert result.reward["tool_reward"] == 1.0


def test_rollout_invalid_json_does_not_crash():
    sample = {"prompt": "answer", "task_type": "true_false", "answer": "yes"}
    generator = SequenceGenerator(["not json"])

    result = run_agentic_rollout(sample, generator)

    assert result.trajectory["steps"][0]["error"] == "invalid_json"
    assert result.reward["is_valid_json"] is False
    assert result.reward["is_correct"] is False


def test_rollout_invalid_bbox_records_tool_error():
    image = Image.new("RGB", (100, 80), "white")
    sample = {"prompt": "zoom", "image": image, "width": 100, "height": 80}
    generator = SequenceGenerator(['{"action": "zoom_in", "bbox": [10, 10, 11, 11]}'])

    result = run_agentic_rollout(
        sample,
        generator,
        RolloutConfig(max_tool_steps=1, min_bbox_area_ratio=0.01),
    )

    assert result.trajectory["invalid_bbox"] is True
    assert result.reward["invalid_bbox"] is True
    assert result.history[-1]["role"] == "tool_observation"
    assert result.history[-1]["content"]["valid"] is False


def test_rollout_token_spans_mask_only_model_generation():
    image = Image.new("RGB", (100, 80), "white")
    sample = {"prompt": "inspect", "image": image, "task_type": "true_false", "answer": "yes"}
    generator = SequenceGenerator(
        [
            '{"action": "zoom_in", "bbox": [10, 20, 40, 50]}',
            '{"action": "final_answer", "answer": "yes"}',
        ]
    )

    result = run_agentic_rollout(sample, generator, RolloutConfig(max_tool_steps=1))
    seq_len = result.token_spans[-1].end
    mask = build_loss_mask(__import__("torch").arange(seq_len), result.token_spans)

    for span in result.token_spans:
        values = mask[span.start : span.end].tolist()
        if span.role == "model_generation":
            assert all(v == 1.0 for v in values)
        else:
            assert all(v == 0.0 for v in values)


def test_zoom_in_supports_normalized_coordinates():
    image = Image.new("RGB", (100, 80), "white")

    obs = zoom_in(image, [0.1, 0.25, 0.4, 0.625], coord_type="normalized")

    assert obs.valid is True
    assert obs.bbox_xyxy == (10, 20, 40, 50)
    assert obs.crop_size == (30, 30)


def test_zoom_in_area_ratio_can_mark_invalid_without_crashing():
    image = Image.new("RGB", (100, 80), "white")

    obs = zoom_in(image, [0, 0, 100, 80], max_bbox_area_ratio=0.8)

    assert obs.valid is False
    assert obs.image is None
    assert obs.area_ratio == 1.0
