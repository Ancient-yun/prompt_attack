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


def attack_loss_from_objective(logits, true_label: int, objective: str):
    """Dispatch the configured untargeted attack objective."""
    normalized = objective.lower().replace("-", "_")
    if normalized == "untargeted_margin":
        return untargeted_margin_loss(logits, true_label)
    if normalized in {
        "negative_cross_entropy",
        "neg_cross_entropy",
        "negce",
        "untargeted_negative_cross_entropy",
    }:
        return negative_cross_entropy_loss(logits, true_label)
    raise ValueError(f"Unsupported attack objective: {objective}")


def dino_loss(similarity):
    """Return 1 - cosine similarity."""
    return 1.0 - similarity.mean()
