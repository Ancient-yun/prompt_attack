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
        init_seed: int = 0,
    ) -> LearnablePrompt:
        """Create mock learnable token embeddings for smoke tests."""
        import torch

        validate_token_init_std(init_std)
        token_texts = build_token_texts(num_tokens)
        values = self._initial_values(
            initializer=initializer,
            shape=(num_tokens, self.embedding_dim),
            init_std=init_std,
            init_seed=init_seed,
        )
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
        init_seed: int = 0,
    ) -> LearnablePromptBatch:
        """Create per-sample mock learnable token embeddings for batch smoke tests."""
        import torch

        validate_token_init_std(init_std)
        token_texts = build_token_texts(num_tokens)
        values = self._initial_values(
            initializer=initializer,
            shape=(len(class_labels), num_tokens, self.embedding_dim),
            init_std=init_std,
            init_seed=init_seed,
        )
        return LearnablePromptBatch(
            prompt_texts=tuple(build_prompt(label, num_tokens) for label in class_labels),
            token_texts=token_texts,
            token_ids=tuple(range(num_tokens)),
            learnable_embeddings=torch.nn.Parameter(values),
        )

    def _initial_values(
        self,
        *,
        initializer: str,
        shape: tuple[int, ...],
        init_std: float,
        init_seed: int,
    ):
        """Create deterministic mock initial embeddings for supported initializer names."""
        import torch

        normalized = initializer.lower().replace("-", "_")
        generator = torch.Generator(device=self.device).manual_seed(init_seed)
        noise = torch.randn(shape, generator=generator, device=self.device, dtype=torch.float32)
        if normalized == "random_real_tokens":
            base = torch.empty(shape, device=self.device, dtype=torch.float32).uniform_(
                -0.05,
                0.05,
                generator=generator,
            )
        elif normalized == "fixed10_class_average":
            base = torch.full(shape, 0.01, device=self.device, dtype=torch.float32)
        else:
            base = torch.zeros(shape, device=self.device, dtype=torch.float32)
        return base + noise * init_std

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
        pil_images = []
        if not require_grad:
            pil_images = [tensor_to_pil(image_tensor[index]) for index in range(image_tensor.shape[0])]
        return GenerationBatchResult(
            image_tensor=image_tensor,
            pil_images=pil_images,
        )
