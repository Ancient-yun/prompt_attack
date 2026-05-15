"""Image-level quality and distance metrics."""

from __future__ import annotations


def _as_batched_tensor(tensor):
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    return tensor


def align_reference_to_candidate(reference, candidate):
    """Resize reference tensor to the candidate spatial size."""
    import torch.nn.functional as F

    reference = _as_batched_tensor(reference)
    candidate = _as_batched_tensor(candidate)
    if reference.shape[-2:] == candidate.shape[-2:]:
        return reference
    return F.interpolate(reference, size=candidate.shape[-2:], mode="bilinear", align_corners=False)


def pixel_distance_metrics(reference, candidate) -> dict[str, float]:
    """Compute pixel distances between two [0, 1] tensors after spatial alignment."""
    import torch

    reference = align_reference_to_candidate(reference.detach(), candidate.detach())
    candidate = _as_batched_tensor(candidate.detach())
    diff = candidate - reference
    return {
        "pixel_l1_mean": float(diff.abs().mean().cpu().item()),
        "pixel_l2": float(torch.linalg.vector_norm(diff).cpu().item()),
        "pixel_l2_mean": float(torch.sqrt(torch.mean(diff.square())).cpu().item()),
        "pixel_linf": float(diff.abs().max().cpu().item()),
    }


def global_ssim(reference, candidate) -> float:
    """Compute a simple global SSIM score after spatial alignment."""
    import torch

    reference = align_reference_to_candidate(reference.detach(), candidate.detach()).clamp(0, 1)
    candidate = _as_batched_tensor(candidate.detach()).clamp(0, 1)
    dims = (-3, -2, -1)
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = reference.mean(dim=dims)
    mu_y = candidate.mean(dim=dims)
    var_x = ((reference - mu_x.view(-1, 1, 1, 1)) ** 2).mean(dim=dims)
    var_y = ((candidate - mu_y.view(-1, 1, 1, 1)) ** 2).mean(dim=dims)
    cov_xy = (
        (reference - mu_x.view(-1, 1, 1, 1)) * (candidate - mu_y.view(-1, 1, 1, 1))
    ).mean(dim=dims)
    score = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    )
    return float(torch.nan_to_num(score).mean().cpu().item())
