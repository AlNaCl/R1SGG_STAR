"""Token-level loss masking utilities for agentic rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


MODEL_GENERATION_ROLE = "model_generation"
VALID_ROLES = {"prompt", "image", "tool_observation", MODEL_GENERATION_ROLE}


@dataclass(frozen=True)
class TokenSpan:
    """Half-open token span [start, end) with an agentic rollout role."""

    start: int
    end: int
    role: str

    def validate(self, sequence_length: int) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(f"unknown token span role: {self.role}")
        if self.start < 0 or self.end < 0:
            raise ValueError("token span indices must be non-negative")
        if self.end < self.start:
            raise ValueError("token span end must be >= start")
        if self.end > sequence_length:
            raise ValueError(
                f"token span end {self.end} exceeds sequence length {sequence_length}"
            )


def _sequence_length(input_ids: torch.Tensor) -> int:
    if input_ids.ndim == 0:
        raise ValueError("input_ids must have at least one dimension")
    return int(input_ids.shape[-1])


def build_loss_mask(input_ids: torch.Tensor, spans: list[TokenSpan]) -> torch.Tensor:
    """Build a mask with 1s only on model-generated token spans.

    The mask follows the last dimension of ``input_ids``. For batched tensors, the
    same spans are broadcast to every batch row.
    """

    seq_len = _sequence_length(input_ids)
    mask = torch.zeros_like(input_ids, dtype=torch.float32)
    for span in spans:
        span.validate(seq_len)
        value = 1.0 if span.role == MODEL_GENERATION_ROLE else 0.0
        if input_ids.ndim == 1:
            mask[span.start : span.end] = value
        else:
            mask[..., span.start : span.end] = value
    return mask


def build_loss_mask_from_roles(input_ids: torch.Tensor, roles: Iterable[str]) -> torch.Tensor:
    """Build a per-token mask from a role sequence."""

    role_list = list(roles)
    seq_len = _sequence_length(input_ids)
    if len(role_list) != seq_len:
        raise ValueError(f"roles length {len(role_list)} does not match sequence length {seq_len}")
    for role in role_list:
        if role not in VALID_ROLES:
            raise ValueError(f"unknown token role: {role}")
    values = [1.0 if role == MODEL_GENERATION_ROLE else 0.0 for role in role_list]
    base = torch.tensor(values, dtype=torch.float32, device=input_ids.device)
    if input_ids.ndim == 1:
        return base
    return base.expand(*input_ids.shape[:-1], seq_len).clone()


def mask_prompt_tokens_in_labels(
    labels: torch.Tensor,
    prompt_lengths: list[int],
    attention_mask: torch.Tensor | None = None,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Mask prompt tokens in a batch of causal-LM labels in place.

    ``prompt_lengths`` should count real tokens before the assistant response for
    each row. When left padding is present, ``attention_mask`` is used to offset
    the masked span to the first non-padding token.
    """

    if labels.ndim != 2:
        raise ValueError("labels must be a 2D batch tensor")
    if len(prompt_lengths) != int(labels.shape[0]):
        raise ValueError("prompt_lengths must contain one value per batch row")
    if attention_mask is not None and attention_mask.shape != labels.shape:
        raise ValueError("attention_mask must match labels shape")

    seq_len = int(labels.shape[-1])
    for row_idx, prompt_len in enumerate(prompt_lengths):
        if prompt_len < 0:
            raise ValueError("prompt lengths must be non-negative")
        start = 0
        if attention_mask is not None:
            active = torch.nonzero(attention_mask[row_idx] > 0, as_tuple=False)
            if active.numel() == 0:
                continue
            start = int(active[0].item())
        end = min(start + int(prompt_len), seq_len)
        labels[row_idx, start:end] = ignore_index
    return labels


def _active_layout(attention_row: torch.Tensor | None, seq_len: int) -> tuple[int, int]:
    if attention_row is None:
        return 0, seq_len
    active = torch.nonzero(attention_row > 0, as_tuple=False)
    if active.numel() == 0:
        return 0, 0
    start = int(active[0].item())
    return start, int(active.numel())


def mask_labels_to_response_spans(
    labels: torch.Tensor,
    response_spans: list[list[tuple[int, int]]],
    attention_mask: torch.Tensor | None = None,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Mask labels in place so only response spans contribute to loss.

    Spans are half-open ``(start, end)`` token offsets relative to the first
    non-padding token in each row. This supports multi-turn agentic SFT where
    user/tool-observation turns between assistant actions must remain masked.
    Spans that are partly truncated are clipped to the active token length.
    """

    if labels.ndim != 2:
        raise ValueError("labels must be a 2D batch tensor")
    if len(response_spans) != int(labels.shape[0]):
        raise ValueError("response_spans must contain one span list per batch row")
    if attention_mask is not None and attention_mask.shape != labels.shape:
        raise ValueError("attention_mask must match labels shape")

    original = labels.clone()
    labels[:] = ignore_index
    seq_len = int(labels.shape[-1])
    for row_idx, spans in enumerate(response_spans):
        active_start, active_len = _active_layout(
            attention_mask[row_idx] if attention_mask is not None else None,
            seq_len,
        )
        if active_len == 0:
            continue
        for start, end in spans:
            if start < 0 or end < 0:
                raise ValueError("response span indices must be non-negative")
            if end < start:
                raise ValueError("response span end must be >= start")
            clipped_start = min(int(start), active_len)
            clipped_end = min(int(end), active_len)
            if clipped_end <= clipped_start:
                continue
            abs_start = min(active_start + clipped_start, seq_len)
            abs_end = min(active_start + clipped_end, seq_len)
            labels[row_idx, abs_start:abs_end] = original[row_idx, abs_start:abs_end]
    return labels


def validate_non_overlapping_spans(spans: list[TokenSpan], sequence_length: int) -> None:
    """Validate spans when callers require a strict non-overlapping layout."""

    occupied: set[int] = set()
    for span in spans:
        span.validate(sequence_length)
        for idx in range(span.start, span.end):
            if idx in occupied:
                raise ValueError(f"overlapping token span at index {idx}")
            occupied.add(idx)
