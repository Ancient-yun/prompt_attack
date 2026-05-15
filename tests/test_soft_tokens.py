import pytest
import torch

from prompt_attack.attacks.soft_tokens import initialize_soft_tokens


def test_initialize_soft_tokens_uses_configured_std() -> None:
    soft_tokens = initialize_soft_tokens(4, 16, device="cpu", init_std=0.1)

    assert isinstance(soft_tokens, torch.nn.Parameter)
    assert soft_tokens.shape == (4, 16)
    assert soft_tokens.requires_grad


def test_initialize_soft_tokens_rejects_non_positive_std() -> None:
    with pytest.raises(ValueError, match="positive"):
        initialize_soft_tokens(4, 16, device="cpu", init_std=0.0)
