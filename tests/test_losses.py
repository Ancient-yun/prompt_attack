import torch
import torch.nn.functional as F

from prompt_attack.attacks.losses import (
    attack_loss_from_objective,
    negative_cross_entropy_loss,
    untargeted_margin_loss,
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
