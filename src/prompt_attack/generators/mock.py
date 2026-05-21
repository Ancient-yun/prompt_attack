"""Small differentiable generator used for smoke tests."""

from __future__ import annotations

from PIL import Image

from prompt_attack.attacks.learnable_tokens import (
    build_prompt,
    build_token_texts,
    validate_token_init_std,
)
from prompt_attack.generators.base import GenerationResult, LearnablePrompt
from prompt_attack.utils.image import tensor_to_pil


class MockEditableGenerator:
    """A differentiable generator that perturbs the input from learnable tokens."""

    supports_gradient = True
    embedding_dim = 64

    def __init__(self, *, device: str) -> None:
        self.device = device

    def create_learnable_prompt(
        self,
        *,
        class_label: str,
        num_tokens: int,
        initializer: str,
        init_std: float,
    ) -> LearnablePrompt:
        """Create mock learnable token embeddings for smoke tests."""
        import torch

        del initializer
        validate_token_init_std(init_std)
        token_texts = build_token_texts(num_tokens)
        values = torch.randn(
            num_tokens,
            self.embedding_dim,
            device=self.device,
            dtype=torch.float32,
        ) * init_std
        return LearnablePrompt(
            prompt_text=build_prompt(class_label, num_tokens),
            token_texts=token_texts,
            token_ids=tuple(range(num_tokens)),
            learnable_embeddings=torch.nn.Parameter(values),
        )

    def sync_learnable_prompt(self, prompt_state: LearnablePrompt) -> None:
        """Mock generator has no embedding table to synchronize."""
        del prompt_state

    def generate(
        self,
        *,
        input_image: Image.Image,
        input_tensor: object,
        prompt_state: LearnablePrompt,
        seed: int,
        require_grad: bool,
    ) -> GenerationResult:
        """Return a differentiable color-shifted version of the input tensor."""
        import torch

        del input_image, seed
        learnable_tokens = prompt_state.learnable_embeddings
        if not isinstance(input_tensor, torch.Tensor) or not isinstance(learnable_tokens, torch.Tensor):
            raise TypeError("MockEditableGenerator expects torch tensors.")
        token_summary = torch.tanh(learnable_tokens.mean(dim=0))
        if token_summary.numel() < 3:
            token_summary = torch.nn.functional.pad(token_summary, (0, 3 - token_summary.numel()))
        color_shift = token_summary[:3].view(1, 3, 1, 1) * 0.08
        image_tensor = (input_tensor + color_shift).clamp(0, 1)
        if not require_grad:
            image_tensor = image_tensor.detach()
        return GenerationResult(image_tensor=image_tensor, pil_image=tensor_to_pil(image_tensor))
