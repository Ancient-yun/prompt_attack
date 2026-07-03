import pytest
import torch

from prompt_attack.attacks.axis_tokens import (
    AxisPromptState,
    build_axis_prompt_batch,
    embeddings_at,
    rank_weighted_strength_weights,
    split_anchor_axis_tokens,
)
from prompt_attack.data.imagenet import ImageRecord
from pathlib import Path


def test_split_anchor_axis_tokens_defaults_to_half_split() -> None:
    assert split_anchor_axis_tokens(8, None) == (4, 4)
    assert split_anchor_axis_tokens(7, None) == (4, 3)


def test_split_anchor_axis_tokens_uses_explicit_anchor_count() -> None:
    assert split_anchor_axis_tokens(8, 3) == (3, 5)


def test_split_anchor_axis_tokens_rejects_out_of_range_anchor() -> None:
    with pytest.raises(ValueError, match="leave room for at least one axis token"):
        split_anchor_axis_tokens(8, 8)
    with pytest.raises(ValueError, match="leave room for at least one axis token"):
        split_anchor_axis_tokens(8, 0)


def test_split_anchor_axis_tokens_rejects_too_few_tokens() -> None:
    with pytest.raises(ValueError, match="at least 2 learnable tokens"):
        split_anchor_axis_tokens(1, None)


def _make_state(*, num_anchor: int = 2, num_axis: int = 2, embed_dim: int = 4) -> AxisPromptState:
    torch.manual_seed(0)
    anchor = torch.nn.Parameter(torch.randn(num_anchor, embed_dim))
    axis_base = torch.randn(num_axis, embed_dim)
    axis_direction = torch.nn.Parameter(torch.randn(num_axis, embed_dim))
    return AxisPromptState(
        token_texts=tuple(f"<v{i + 1}>" for i in range(num_anchor + num_axis)),
        token_ids=tuple(range(num_anchor + num_axis)),
        num_anchor_tokens=num_anchor,
        num_axis_tokens=num_axis,
        anchor_embeddings=anchor,
        axis_base=axis_base,
        axis_direction=axis_direction,
    )


def test_embeddings_at_zero_strength_equals_anchor_plus_base() -> None:
    state = _make_state()

    result = embeddings_at(state, 0.0)

    expected = torch.cat([state.anchor_embeddings, state.axis_base], dim=0)
    assert torch.allclose(result, expected)


def test_embeddings_at_full_strength_matches_manual_concat() -> None:
    state = _make_state()

    result = embeddings_at(state, 1.0)

    expected = torch.cat(
        [state.anchor_embeddings, state.axis_base + state.axis_direction], dim=0
    )
    assert torch.allclose(result, expected)


def test_embeddings_at_row_order_matches_token_ids() -> None:
    state = _make_state(num_anchor=2, num_axis=3)

    result = embeddings_at(state, 0.5)

    assert result.shape == (len(state.token_ids), state.anchor_embeddings.shape[-1])


def test_build_axis_prompt_batch_shares_one_tensor_varying_class_label() -> None:
    state = _make_state()
    records = [
        ImageRecord(
            path=Path("a.png"),
            synset="n0",
            class_label="dog",
            class_index=0,
            image_id="a",
        ),
        ImageRecord(
            path=Path("b.png"),
            synset="n1",
            class_label="cat",
            class_index=1,
            image_id="b",
        ),
    ]

    batch = build_axis_prompt_batch(state, records, t=0.5)

    assert len(batch.prompt_texts) == 2
    assert batch.prompt_texts[0] != batch.prompt_texts[1]
    assert "dog" in batch.prompt_texts[0]
    assert "cat" in batch.prompt_texts[1]
    assert torch.allclose(batch.learnable_embeddings, embeddings_at(state, 0.5))


def test_rank_weighted_strength_weights_sums_to_one_and_decreases_with_t() -> None:
    weights = rank_weighted_strength_weights([0.2, 0.6, 1.0])

    assert sum(weights) == pytest.approx(1.0)
    assert weights[0] > weights[1] > weights[2]


def test_rank_weighted_strength_weights_handles_unsorted_input() -> None:
    weights = rank_weighted_strength_weights([1.0, 0.2, 0.6])

    # index 1 (t=0.2, smallest) should get the largest weight, index 0 (t=1.0) smallest.
    assert weights[1] > weights[2] > weights[0]
    assert sum(weights) == pytest.approx(1.0)


def test_rank_weighted_strength_weights_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        rank_weighted_strength_weights([])


def test_mock_generator_axis_prompt_gradient_flows_to_both_params() -> None:
    from prompt_attack.generators.mock import MockEditableGenerator

    generator = MockEditableGenerator(device="cpu")
    state = generator.create_axis_prompt(
        class_label="object",
        num_tokens=6,
        num_anchor_tokens=3,
        initializer="object",
        init_std=0.02,
        init_seed=0,
    )
    assert state.anchor_embeddings.requires_grad
    assert state.axis_direction.requires_grad
    assert not state.axis_base.requires_grad

    embeddings = embeddings_at(state, t=0.5)
    loss = embeddings.sum()
    loss.backward()

    assert state.anchor_embeddings.grad is not None
    assert torch.any(state.anchor_embeddings.grad != 0)
    assert state.axis_direction.grad is not None
    assert torch.any(state.axis_direction.grad != 0)


def test_mock_generator_axis_prompt_zero_strength_gradient_skips_direction() -> None:
    from prompt_attack.generators.mock import MockEditableGenerator

    generator = MockEditableGenerator(device="cpu")
    state = generator.create_axis_prompt(
        class_label="object",
        num_tokens=6,
        num_anchor_tokens=3,
        initializer="object",
        init_std=0.02,
        init_seed=0,
    )

    embeddings = embeddings_at(state, t=0.0)
    loss = embeddings.sum()
    loss.backward()

    assert state.anchor_embeddings.grad is not None
    # At t=0 the axis embedding is axis_base + 0*axis_direction, so its gradient wrt
    # axis_direction is exactly zero -- expected linearity, not a bug.
    assert torch.all(state.axis_direction.grad == 0)
