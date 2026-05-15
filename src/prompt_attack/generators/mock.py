"""Small differentiable generator used for smoke tests."""

from __future__ import annotations

from PIL import Image

from prompt_attack.generators.base import GenerationResult
from prompt_attack.utils.image import tensor_to_pil


class MockEditableGenerator:
    """A differentiable generator that perturbs the input from soft tokens."""

    supports_gradient = True

    def __init__(self, *, device: str) -> None:
        self.device = device

    def soft_token_dim(self, class_label: str) -> int:
        """Return the smoke-test token dimension."""
        del class_label
        return 64

    def generate(
        self,
        *,
        input_image: Image.Image,
        input_tensor: object,
        prompt: str,
        soft_tokens: object,
        seed: int,
        require_grad: bool,
    ) -> GenerationResult:
        """Return a differentiable color-shifted version of the input tensor."""
        import torch

        del input_image, prompt, seed
        if not isinstance(input_tensor, torch.Tensor) or not isinstance(soft_tokens, torch.Tensor):
            raise TypeError("MockEditableGenerator expects torch tensors.")
        token_summary = torch.tanh(soft_tokens.mean(dim=0))
        if token_summary.numel() < 3:
            token_summary = torch.nn.functional.pad(token_summary, (0, 3 - token_summary.numel()))
        color_shift = token_summary[:3].view(1, 3, 1, 1) * 0.08
        image_tensor = (input_tensor + color_shift).clamp(0, 1)
        if not require_grad:
            image_tensor = image_tensor.detach()
        return GenerationResult(image_tensor=image_tensor, pil_image=tensor_to_pil(image_tensor))
