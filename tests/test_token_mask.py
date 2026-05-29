import pytest
import torch

from src.rl.mask_utils import (
    TokenSpan,
    build_loss_mask,
    build_loss_mask_from_roles,
    validate_non_overlapping_spans,
)


def test_build_loss_mask_only_marks_model_generation_tokens():
    input_ids = torch.arange(10)
    spans = [
        TokenSpan(0, 3, "prompt"),
        TokenSpan(3, 5, "image"),
        TokenSpan(5, 8, "model_generation"),
        TokenSpan(8, 10, "tool_observation"),
    ]

    mask = build_loss_mask(input_ids, spans)

    assert mask.tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
    assert mask.dtype == torch.float32


def test_build_loss_mask_supports_batched_input_ids():
    input_ids = torch.arange(20).reshape(2, 10)
    spans = [TokenSpan(1, 4, "model_generation"), TokenSpan(6, 9, "model_generation")]

    mask = build_loss_mask(input_ids, spans)

    expected = [[0, 1, 1, 1, 0, 0, 1, 1, 1, 0]] * 2
    assert mask.tolist() == expected


def test_non_generation_span_overrides_previous_generation_span():
    input_ids = torch.arange(6)
    spans = [TokenSpan(0, 6, "model_generation"), TokenSpan(2, 4, "tool_observation")]

    mask = build_loss_mask(input_ids, spans)

    assert mask.tolist() == [1, 1, 0, 0, 1, 1]


def test_build_loss_mask_rejects_out_of_bounds_span():
    input_ids = torch.arange(5)

    with pytest.raises(ValueError, match="exceeds sequence length"):
        build_loss_mask(input_ids, [TokenSpan(0, 6, "model_generation")])


def test_build_loss_mask_rejects_unknown_role():
    input_ids = torch.arange(5)

    with pytest.raises(ValueError, match="unknown token span role"):
        build_loss_mask(input_ids, [TokenSpan(0, 2, "assistant")])


def test_build_loss_mask_from_roles():
    input_ids = torch.arange(5)
    roles = ["prompt", "model_generation", "image", "model_generation", "tool_observation"]

    mask = build_loss_mask_from_roles(input_ids, roles)

    assert mask.tolist() == [0, 1, 0, 1, 0]


def test_build_loss_mask_from_roles_supports_batched_input_ids():
    input_ids = torch.arange(10).reshape(2, 5)
    roles = ["prompt", "model_generation", "image", "model_generation", "tool_observation"]

    mask = build_loss_mask_from_roles(input_ids, roles)

    assert mask.tolist() == [[0, 1, 0, 1, 0], [0, 1, 0, 1, 0]]


def test_build_loss_mask_from_roles_rejects_length_mismatch():
    input_ids = torch.arange(5)

    with pytest.raises(ValueError, match="does not match sequence length"):
        build_loss_mask_from_roles(input_ids, ["prompt"])


def test_validate_non_overlapping_spans_rejects_overlap():
    spans = [TokenSpan(0, 3, "prompt"), TokenSpan(2, 4, "model_generation")]

    with pytest.raises(ValueError, match="overlapping token span"):
        validate_non_overlapping_spans(spans, sequence_length=5)


def test_validate_non_overlapping_spans_accepts_adjacent_spans():
    spans = [TokenSpan(0, 3, "prompt"), TokenSpan(3, 5, "model_generation")]

    validate_non_overlapping_spans(spans, sequence_length=5)
