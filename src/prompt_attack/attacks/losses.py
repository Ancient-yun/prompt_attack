"""Attack and semantic losses."""

from __future__ import annotations


def untargeted_margin_loss(logits, true_label: int):
    """Return logit_true - max(logit_other) for minimization."""
    import torch

    true_logit = logits[0, true_label]
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[0, true_label] = False
    other_max = logits.masked_select(mask).view(1, -1).max(dim=-1).values[0]
    return true_logit - other_max


def negative_cross_entropy_loss(logits, true_label: int):
    """Return -CE(logits, true_label) so minimization maximizes true-label CE."""
    import torch
    import torch.nn.functional as F

    target = torch.tensor([true_label], device=logits.device, dtype=torch.long)
    return -F.cross_entropy(logits, target)


def cr_loss(logits, true_label: int):
    """Return the classification-rejection loss."""
    return negative_cross_entropy_loss(logits, true_label)


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


def attack_loss_from_objective(logits, true_label: int, objective: str):
    """Dispatch the configured untargeted attack objective."""
    normalized = _normalized_objective(objective)
    if normalized == "untargeted_margin":
        return untargeted_margin_loss(logits, true_label)
    if is_cr_objective(objective) or is_cr_dino_objective(objective):
        return cr_loss(logits, true_label)
    if normalized in {
        "negative_cross_entropy",
        "neg_cross_entropy",
        "negce",
        "untargeted_negative_cross_entropy",
    }:
        return negative_cross_entropy_loss(logits, true_label)
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
