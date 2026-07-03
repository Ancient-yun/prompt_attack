"""MAELS-inspired anchor/axis token split for strength-scheduled UAP attacks.

Splits the learnable textual-inversion tokens into two groups: identity-preserving
``anchor`` tokens and a continuous ``axis`` direction. The effective axis embedding at
strength ``t`` is ``axis_base + t * axis_direction``, so sweeping ``t`` from 0 to 1 walks
a straight-line path from the token's initial value toward the learned attack direction,
mirroring how arXiv:2402.03095 (MAELS) walks its unsupervised semantic code ``z3`` in small
steps and rewards the earliest legitimate misclassification rather than the strongest one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil

from prompt_attack.attacks.learnable_tokens import build_prompt
from prompt_attack.data.imagenet import ImageRecord
from prompt_attack.generators.base import LearnablePrompt, LearnablePromptBatch


@dataclass(frozen=True)
class AxisPromptState:
    """Prompt state split into an identity-preserving anchor and a continuous axis."""

    token_texts: tuple[str, ...]
    token_ids: tuple[int, ...]
    num_anchor_tokens: int
    num_axis_tokens: int
    anchor_embeddings: object  # torch.nn.Parameter, shape (A, D)
    axis_base: object  # torch.Tensor buffer, shape (X, D), requires_grad=False
    axis_direction: object  # torch.nn.Parameter, shape (X, D)


def split_anchor_axis_tokens(
    num_learnable_tokens: int, num_anchor_tokens: int | None
) -> tuple[int, int]:
    """Return ``(num_anchor, num_axis)`` derived from config, defaulting to a half split."""
    if num_learnable_tokens < 2:
        raise ValueError(
            "strength_schedule requires at least 2 learnable tokens "
            f"(1 anchor + 1 axis), got {num_learnable_tokens}."
        )
    anchor = (
        ceil(num_learnable_tokens / 2) if num_anchor_tokens is None else num_anchor_tokens
    )
    if not 0 < anchor < num_learnable_tokens:
        raise ValueError(
            "num_anchor_tokens must leave room for at least one axis token: "
            f"got num_anchor_tokens={anchor}, num_learnable_tokens={num_learnable_tokens}."
        )
    return anchor, num_learnable_tokens - anchor


def embeddings_at(state: AxisPromptState, t: float):
    """Return the concatenated ``(N, D)`` embedding tensor at strength ``t``.

    Row order matches ``state.token_ids`` order, so this feeds directly into the
    existing FLUX.2 embedding hook with no changes to the generator's injection code.
    """
    import torch

    axis_values = state.axis_base + t * state.axis_direction
    return torch.cat([state.anchor_embeddings, axis_values], dim=0)


def build_axis_prompt(state: AxisPromptState, *, class_label: str, t: float) -> LearnablePrompt:
    """Wrap the strength-``t`` embedding in a single-image ``LearnablePrompt``."""
    return LearnablePrompt(
        prompt_text=build_prompt(class_label, len(state.token_texts)),
        token_texts=state.token_texts,
        token_ids=state.token_ids,
        learnable_embeddings=embeddings_at(state, t),
    )


def build_axis_prompt_batch(
    state: AxisPromptState, records: Sequence[ImageRecord], *, t: float
) -> LearnablePromptBatch:
    """Build a batch prompt sharing one axis-prompt tensor at strength ``t``.

    Mirrors ``LearnableTokenAttackRunner._shared_prompt_batch`` for the legacy flat-tensor
    prompt state.
    """
    return LearnablePromptBatch(
        prompt_texts=tuple(
            build_prompt(record.class_label, len(state.token_texts)) for record in records
        ),
        token_texts=state.token_texts,
        token_ids=state.token_ids,
        learnable_embeddings=embeddings_at(state, t),
    )


def rank_weighted_strength_weights(strengths: Sequence[float]) -> list[float]:
    """Return weights (aligned to ``strengths`` order) favoring small-``t`` successes.

    Sorts ascending and assigns triangular rank weights: for ``K`` strengths, rank
    ``r = 1..K`` (``r=1`` is the smallest ``t``) gets ``weight(r) = 2*(K-r+1) / (K*(K+1))``.
    Weights sum to 1 and decrease monotonically as ``t`` grows, so an early (small-``t``)
    successful, legitimate crossing is rewarded more than pushing the same crossing out to
    ``t=1`` -- the training-time analogue of MAELS' "earliest valid step" search. A ``1/t``
    weighting was considered and rejected: it diverges as ``t -> 0``.
    """
    if not strengths:
        raise ValueError("strengths must contain at least one value.")
    order = sorted(range(len(strengths)), key=lambda index: strengths[index])
    total_ranks = len(strengths)
    denominator = total_ranks * (total_ranks + 1)
    weights = [0.0] * total_ranks
    for rank, original_index in enumerate(order, start=1):
        weights[original_index] = 2.0 * (total_ranks - rank + 1) / denominator
    return weights
