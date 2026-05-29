import pytest

from src.rl.rewards import (
    compute_correctness_reward,
    compute_format_reward,
    compute_tool_reward,
    compute_total_reward,
)


def test_correctness_multiple_choice_normalized_exact_match():
    pred = '{"action": "final_answer", "answer": " A "}'

    reward = compute_correctness_reward(pred, {"task_type": "multiple_choice", "answer": "a"})

    assert reward == 1.0


def test_correctness_numeric_uses_tolerance():
    pred = '{"action": "final_answer", "answer": "about 3.142"}'

    reward = compute_correctness_reward(
        pred,
        {"task_type": "numeric", "answer": 3.14159},
        numeric_tolerance=0.01,
    )

    assert reward == 1.0


def test_correctness_open_qa_accepts_keywords():
    pred = '{"action": "final_answer", "answer": "The object is a cargo ship near the dock."}'

    reward = compute_correctness_reward(
        pred,
        {"task_type": "open_qa", "keywords": ["cargo ship", "vessel"]},
    )

    assert reward == 1.0


def test_correctness_scene_graph_triplet_match():
    pred = (
        '{"action": "final_answer", "answer": '
        '{"relationships": [{"subject": "airplane.1", "predicate": "on", "object": "runway.2"}]}}'
    )
    sample = {
        "task_type": "scene_graph",
        "relationships": [{"subject": "airplane.1", "predicate": "on", "object": "runway.2"}],
    }

    assert compute_correctness_reward(pred, sample) == 1.0


def test_format_reward_accepts_valid_zoom_action():
    pred = '{"thought": "inspect target", "action": "zoom_in", "bbox": [10, 20, 40, 50]}'

    assert compute_format_reward(pred) == 1.0


def test_format_reward_rejects_extra_text_in_strict_mode():
    pred = 'prefix {"action": "final_answer", "answer": "yes"}'

    assert compute_format_reward(pred, strict=True) == 0.0


def test_format_reward_partial_for_missing_final_answer():
    pred = '{"action": "final_answer"}'

    assert compute_format_reward(pred) == 0.5


def test_tool_reward_scores_valid_nonwhole_zoom():
    trajectory = {"tool_calls": [{"action": "zoom_in", "bbox": [10, 10, 40, 40]}]}
    sample = {"width": 100, "height": 100}

    assert compute_tool_reward(trajectory, sample, max_tool_steps=2) == 1.0


def test_tool_reward_penalizes_repeated_and_whole_image_zoom():
    trajectory = {
        "tool_calls": [
            {"action": "zoom_in", "bbox": [0, 0, 100, 100]},
            {"action": "zoom_in", "bbox": [0, 0, 100, 100]},
        ]
    }
    sample = {"width": 100, "height": 100}

    assert compute_tool_reward(trajectory, sample, max_tool_steps=2) == pytest.approx(0.45)


def test_tool_reward_uses_evidence_overlap():
    trajectory = {"tool_calls": [{"action": "zoom_in", "bbox": [10, 10, 40, 40]}]}
    sample = {"width": 100, "height": 100, "evidence_bbox": [20, 20, 50, 50]}

    assert compute_tool_reward(trajectory, sample) == pytest.approx(4 / 7)


def test_total_reward_gates_tool_reward_on_correctness():
    pred = '{"action": "final_answer", "answer": "yes"}'
    trajectory = {"tool_calls": [{"action": "zoom_in", "bbox": [10, 10, 40, 40]}]}
    sample = {"task_type": "true_false", "answer": "yes", "width": 100, "height": 100}

    result = compute_total_reward(
        pred,
        sample,
        trajectory,
        {"lambda_format": 0.1, "lambda_tool": 0.2},
    )

    assert result["reward"] == pytest.approx(1.3)
    assert result["correctness_reward"] == 1.0
    assert result["format_reward"] == 1.0
    assert result["tool_reward"] == 1.0
    assert result["is_correct"] is True
    assert result["is_valid_json"] is True
    assert result["used_zoom"] is True
    assert result["invalid_bbox"] is False


def test_total_reward_does_not_add_tool_reward_when_wrong():
    pred = '{"action": "final_answer", "answer": "no"}'
    trajectory = {"tool_calls": [{"action": "zoom_in", "bbox": [10, 10, 40, 40]}]}
    sample = {"task_type": "true_false", "answer": "yes", "width": 100, "height": 100}

    result = compute_total_reward(pred, sample, trajectory, {"lambda_format": 0.1, "lambda_tool": 0.2})

    assert result["reward"] == pytest.approx(0.1)
    assert result["is_correct"] is False
