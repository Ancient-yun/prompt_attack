"""Small differentiable generator used for smoke tests."""

from __future__ import annotations

from collections.abc import Sequence

from PIL import Image

from prompt_attack.attacks.learnable_tokens import (
    build_prompt,
    build_token_texts,
    validate_token_init_std,
)
from prompt_attack.generators.base import GenerationBatchResult, GenerationResult, LearnablePrompt, LearnablePromptBatch
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

    def create_learnable_prompt_batch(
        self,
        *,
        class_labels: Sequence[str],
        num_tokens: int,
        initializer: str,
        init_std: float,
    ) -> LearnablePromptBatch:
        """Create per-sample mock learnable token embeddings for batch smoke tests."""
        import torch

        del initializer
        validate_token_init_std(init_std)
        token_texts = build_token_texts(num_tokens)
        values = torch.randn(
            len(class_labels),
            num_tokens,
            self.embedding_dim,
            device=self.device,
            dtype=torch.float32,
        ) * init_std
        return LearnablePromptBatch(
            prompt_texts=tuple(build_prompt(label, num_tokens) for label in class_labels),
            token_texts=token_texts,
            token_ids=tuple(range(num_tokens)),
            learnable_embeddings=torch.nn.Parameter(values),
        )

    def sync_learnable_prompt(self, prompt_state: LearnablePrompt) -> None:
        """Mock generator has no embedding table to synchronize."""
        del prompt_state

    def sync_learnable_prompt_batch(self, prompt_state: LearnablePromptBatch) -> None:
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

    def generate_batch(
        self,
        *,
        input_images: Sequence[Image.Image],
        input_tensor: object,
        prompt_state: LearnablePromptBatch,
        seeds: Sequence[int],
        require_grad: bool,
    ) -> GenerationBatchResult:
        """Return a differentiable batch of color-shifted input tensors."""
        import torch

        del input_images, seeds
        learnable_tokens = prompt_state.learnable_embeddings
        if not isinstance(input_tensor, torch.Tensor) or not isinstance(learnable_tokens, torch.Tensor):
            raise TypeError("MockEditableGenerator expects torch tensors.")
        if learnable_tokens.ndim == 2:
            token_summary = torch.tanh(learnable_tokens.mean(dim=0)).expand(input_tensor.shape[0], -1)
        elif learnable_tokens.ndim == 3:
            token_summary = torch.tanh(learnable_tokens.mean(dim=1))
        else:
            raise ValueError("Mock batch prompt embeddings must have shape [N, D] or [B, N, D].")
        if token_summary.shape[-1] < 3:
            token_summary = torch.nn.functional.pad(token_summary, (0, 3 - token_summary.shape[-1]))
        color_shift = token_summary[:, :3].view(input_tensor.shape[0], 3, 1, 1) * 0.08
        image_tensor = (input_tensor + color_shift).clamp(0, 1)
        if not require_grad:
            image_tensor = image_tensor.detach()
        return GenerationBatchResult(
            image_tensor=image_tensor,
            pil_images=[tensor_to_pil(image_tensor[index]) for index in range(image_tensor.shape[0])],
        )
