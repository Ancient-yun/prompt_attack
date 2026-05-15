"""Learnable soft token initialization."""

from __future__ import annotations


def initialize_soft_tokens(num_tokens: int, token_dim: int, *, device: str, init_std: float):
    """Create learnable soft-token embeddings."""
    import torch

    if init_std <= 0:
        raise ValueError(f"Soft-token init std must be positive, got {init_std}.")
    values = torch.randn(num_tokens, token_dim, device=device, dtype=torch.float32) * init_std
    return torch.nn.Parameter(values)


def build_prompt(class_label: str, num_tokens: int) -> str:
    """Build the visible prompt string used for logging.

    The actual FLUX.2 path passes continuous prompt embeddings directly. The
    human-readable placeholders are kept only so each run records the intended
    soft-token layout with the class label at the end.
    """
    tokens = " ".join(f"[V{i + 1}]" for i in range(num_tokens))
    return f"{tokens} a photo of {class_label}"
