"""Generator adapter interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image


@dataclass(frozen=True)
class GenerationResult:
    image_tensor: object
    pil_image: Image.Image


class EditableGenerator(Protocol):
    supports_gradient: bool

    def soft_token_dim(self, class_label: str) -> int:
        """Return the conditioning dimension expected by this generator."""

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
        """Generate or edit an image."""
