"""Learnable textual-inversion token prompt helpers."""

from __future__ import annotations


def validate_token_init_std(init_std: float) -> None:
    """Validate the noise scale used for learnable token initialization."""
    if init_std <= 0:
        raise ValueError(f"Soft-token init std must be positive, got {init_std}.")


def build_token_texts(num_tokens: int) -> tuple[str, ...]:
    """Return the fixed textual-inversion token strings for an attack."""
    if num_tokens <= 0:
        raise ValueError(f"num_soft_tokens must be positive, got {num_tokens}.")
    return tuple(f"<v{i + 1}>" for i in range(num_tokens))


def build_prompt(class_label: str, num_tokens: int) -> str:
    """Build the prompt string that goes through the tokenizer."""
    tokens = " ".join(build_token_texts(num_tokens))
    return f"{tokens} a photo of {class_label}"
