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


def validate_non_overlapping_spans(spans: list[TokenSpan], sequence_length: int) -> None:
    """Validate spans when callers require a strict non-overlapping layout."""

    occupied: set[int] = set()
    for span in spans:
        span.validate(sequence_length)
        for idx in range(span.start, span.end):
            if idx in occupied:
                raise ValueError(f"overlapping token span at index {idx}")
            occupied.add(idx)
