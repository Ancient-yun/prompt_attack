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


def margin_hinge_loss(
    logits,
    true_label: Any,
    *,
    margin: float = 0.0,
    reduction: str = "mean",
):
    """Return a bounded untargeted margin hinge loss for minimization."""
    import torch

    target = _target_tensor(logits, true_label)
    true_logit = logits.gather(1, target.view(-1, 1)).squeeze(1)
    other_logits = logits.clone()
    other_logits.scatter_(1, target.view(-1, 1), -torch.inf)
    other_max = other_logits.max(dim=-1).values
    return _reduce_loss(torch.relu(true_logit - other_max + margin), reduction)


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


def is_margin_dino_constraint_objective(objective: str) -> bool:
    """Return whether the objective should use bounded attack plus image similarity."""
    return _normalized_objective(objective) in {
        "margin_dino",
        "margin_dino_img2img",
        "margin_dino_constraint",
        "hinge_dino_constraint",
        "margin_clip",
        "margin_clip_img2img",
        "margin_clip_image",
        "margin_clip_image_to_image",
    }


def uses_clip_image_similarity(objective: str) -> bool:
    """Return whether the objective expects CLIP image-to-image similarity."""
    return _normalized_objective(objective) in {
        "margin_clip",
        "margin_clip_img2img",
        "margin_clip_image",
        "margin_clip_image_to_image",
    }


def uses_dino_similarity(objective: str) -> bool:
    """Return whether the objective expects DINO image-to-image similarity."""
    normalized = _normalized_objective(objective)
    return normalized in {
        "classification_rejection_dino",
        "classification_rejection_with_dino",
        "cr_dino",
        "margin_dino",
        "margin_dino_img2img",
        "margin_dino_constraint",
        "hinge_dino_constraint",
    }


def attack_loss_from_objective(logits, true_label: Any, objective: str, *, reduction: str = "mean"):
    """Dispatch the configured untargeted attack objective."""
    normalized = _normalized_objective(objective)
    if normalized == "untargeted_margin":
        return untargeted_margin_loss(logits, true_label, reduction=reduction)
    if is_cr_objective(objective) or is_cr_dino_objective(objective):
        return cr_loss(logits, true_label, reduction=reduction)
    if is_margin_dino_constraint_objective(objective):
        return margin_hinge_loss(logits, true_label, reduction=reduction)
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
    if is_margin_dino_constraint_objective(objective):
        return 1.0, 1.0
    return 1.0 - lambda_sem, lambda_sem


def weighted_attack_semantic_loss(attack_loss, semantic_loss, lambda_sem: float):
    """Blend attack and semantic losses with complementary weights."""
    validate_lambda_sem(lambda_sem)
    return (1.0 - lambda_sem) * attack_loss + lambda_sem * semantic_loss


def semantic_image_loss(similarity, *, reduction: str = "mean"):
    """Return a continuous image-to-image preservation loss."""
    return _reduce_loss(1.0 - similarity, reduction)


def objective_loss_components(
    logits,
    true_label: Any,
    objective: str,
    *,
    semantic_similarity=None,
    lambda_sem: float,
    semantic_loss_weight: float,
    attack_margin: float,
) -> tuple[Any, Any, Any, Any]:
    """Return per-sample attack, semantic, weighted-semantic, and total losses."""
    import torch

    if is_margin_dino_constraint_objective(objective):
        if semantic_similarity is None:
            raise ValueError(f"{objective} requires image-to-image semantic similarity.")
        attack_losses = margin_hinge_loss(
            logits,
            true_label,
            margin=attack_margin,
            reduction="none",
        )
        semantic_losses = semantic_image_loss(semantic_similarity, reduction="none")
        weighted_semantic_losses = semantic_loss_weight * semantic_losses
        total_losses = attack_losses + weighted_semantic_losses
        return attack_losses, semantic_losses, weighted_semantic_losses, total_losses

    attack_losses = attack_loss_from_objective(logits, true_label, objective, reduction="none")
    attack_loss_weight, semantic_loss_weight = attack_semantic_loss_weights(objective, lambda_sem)
    if semantic_similarity is not None:
        semantic_losses = semantic_image_loss(semantic_similarity, reduction="none")
    else:
        semantic_losses = torch.zeros_like(attack_losses)
    if semantic_loss_weight > 0:
        if semantic_similarity is None:
            raise ValueError(f"{objective} requires image-to-image semantic similarity.")
        weighted_semantic_losses = semantic_loss_weight * semantic_losses
        total_losses = attack_loss_weight * attack_losses + weighted_semantic_losses
    else:
        weighted_semantic_losses = torch.zeros_like(attack_losses)
        total_losses = attack_losses
    return attack_losses, semantic_losses, weighted_semantic_losses, total_losses
