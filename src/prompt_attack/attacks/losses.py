"""Attack and semantic losses."""

from __future__ import annotations

from typing import Any


def _target_tensor(logits, true_label: Any):
    import torch

    if torch.is_tensor(true_label):
        target = true_label.to(device=logits.device, dtype=torch.long)
    else:
        target = torch.tensor([true_label], device=logits.device, dtype=torch.long)
    if target.ndim == 0:
        target = target.unsqueeze(0)
    return target


def _reduce_loss(loss, reduction: str):
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"Unsupported loss reduction: {reduction}")


def untargeted_margin_loss(logits, true_label: Any, *, reduction: str = "mean"):
    """Return logit_true - max(logit_other) for minimization."""
    import torch

    target = _target_tensor(logits, true_label)
    true_logit = logits.gather(1, target.view(-1, 1)).squeeze(1)
    other_logits = logits.clone()
    other_logits.scatter_(1, target.view(-1, 1), -torch.inf)
    other_max = other_logits.max(dim=-1).values
    return _reduce_loss(true_logit - other_max, reduction)


def negative_cross_entropy_loss(logits, true_label: Any, *, reduction: str = "mean"):
    """Return -CE(logits, true_label) so minimization maximizes true-label CE."""
    import torch.nn.functional as F

    target = _target_tensor(logits, true_label)
    return -F.cross_entropy(logits, target, reduction=reduction)


def cr_loss(logits, true_label: Any, *, reduction: str = "mean"):
    """Return the classification-rejection loss."""
    return negative_cross_entropy_loss(logits, true_label, reduction=reduction)


def _normalized_objective(objective: str) -> str:
    return objective.lower().replace("-", "_")


def is_cr_objective(objective: str) -> bool:
    """Return whether the configured objective should use CR without semantic loss."""
    return _normalized_objective(objective) in {
        "classification_rejection",
        "cr",
    }


def is_cr_dino_objective(objective: str) -> bool:
    """Return whether the configured objective should use CR with DINO semantic loss."""
    return _normalized_objective(objective) in {
        "classification_rejection_dino",
        "classification_rejection_with_dino",
        "cr_dino",
    }


def attack_loss_from_objective(logits, true_label: Any, objective: str, *, reduction: str = "mean"):
    """Dispatch the configured untargeted attack objective."""
    normalized = _normalized_objective(objective)
    if normalized == "untargeted_margin":
        return untargeted_margin_loss(logits, true_label, reduction=reduction)
    if is_cr_objective(objective) or is_cr_dino_objective(objective):
        return cr_loss(logits, true_label, reduction=reduction)
    if normalized in {
        "negative_cross_entropy",
        "neg_cross_entropy",
        "negce",
        "untargeted_negative_cross_entropy",
    }:
        return negative_cross_entropy_loss(logits, true_label, reduction=reduction)
    raise ValueError(f"Unsupported attack objective: {objective}")


def validate_lambda_sem(lambda_sem: float) -> None:
    """Validate the semantic/classification tradeoff weight."""
    if not 0.0 <= lambda_sem <= 1.0:
        raise ValueError(
            "attack.lambda_sem must be in [0, 1] because attack loss weight is "
            f"defined as 1 - lambda_sem, got {lambda_sem}."
        )


def attack_semantic_loss_weights(objective: str, lambda_sem: float) -> tuple[float, float]:
    """Return attack and semantic weights for the configured objective."""
    validate_lambda_sem(lambda_sem)
    if is_cr_objective(objective):
        return 1.0, 0.0
    return 1.0 - lambda_sem, lambda_sem


def weighted_attack_semantic_loss(attack_loss, semantic_loss, lambda_sem: float):
    """Blend attack and semantic losses with complementary weights."""
    validate_lambda_sem(lambda_sem)
    return (1.0 - lambda_sem) * attack_loss + lambda_sem * semantic_loss


def dino_loss(similarity):
    """Return 1 - cosine similarity."""
    return 1.0 - similarity.mean()
