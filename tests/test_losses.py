import pytest
import torch
import torch.nn.functional as F

from prompt_attack.attacks.losses import (
    attack_semantic_loss_weights,
    attack_loss_from_objective,
    cr_loss,
    is_cr_dino_objective,
    is_cr_objective,
    negative_cross_entropy_loss,
    untargeted_margin_loss,
    validate_lambda_sem,
    weighted_attack_semantic_loss,
)


def test_negative_cross_entropy_loss() -> None:
    logits = torch.tensor([[1.0, 2.0, -1.0]])
    expected = -F.cross_entropy(logits, torch.tensor([1]))
    actual = negative_cross_entropy_loss(logits, true_label=1)
    assert torch.allclose(actual, expected)


def test_attack_loss_dispatch() -> None:
    logits = torch.tensor([[1.0, 2.0, -1.0]])
    assert torch.allclose(
        attack_loss_from_objective(logits, 1, "untargeted_margin"),
        untargeted_margin_loss(logits, 1),
    )
    assert torch.allclose(
        attack_loss_from_objective(logits, 1, "negative_cross_entropy"),
        negative_cross_entropy_loss(logits, 1),
    )
    assert torch.allclose(
        attack_loss_from_objective(logits, 1, "cr"),
        cr_loss(logits, 1),
    )
    assert torch.allclose(attack_loss_from_objective(logits, 1, "cr_dino"), cr_loss(logits, 1))


def test_cr_loss_matches_classification_rejection_term() -> None:
    logits = torch.tensor([[1.0, 2.0, -1.0]])

    assert torch.allclose(cr_loss(logits, true_label=1), negative_cross_entropy_loss(logits, 1))
    assert is_cr_objective("cr")
    assert is_cr_objective("classification-rejection")
    assert is_cr_dino_objective("cr-dino")


def test_cr_objective_disables_semantic_loss_weight() -> None:
    assert attack_semantic_loss_weights("cr", lambda_sem=0.9) == (1.0, 0.0)
    assert attack_semantic_loss_weights("cr_dino", lambda_sem=0.9) == pytest.approx((0.1, 0.9))


def test_weighted_attack_semantic_loss_uses_complementary_weights() -> None:
    attack_loss = torch.tensor(-8.0)
    semantic_loss = torch.tensor(0.5)

    actual = weighted_attack_semantic_loss(attack_loss, semantic_loss, lambda_sem=0.25)

    assert torch.allclose(actual, torch.tensor(-5.875))


def test_lambda_sem_must_be_unit_interval() -> None:
    for value in (-0.1, 1.1):
        try:
            validate_lambda_sem(value)
        except ValueError as exc:
            assert "1 - lambda_sem" in str(exc)
        else:
            raise AssertionError(f"validate_lambda_sem accepted {value}")
