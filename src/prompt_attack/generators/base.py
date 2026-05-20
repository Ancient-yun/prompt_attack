"""Generator adapter interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image


@dataclass(frozen=True)
class GenerationResult:
    image_tensor: object
    pil_image: Image.Image


@dataclass(frozen=True)
class LearnablePrompt:
    """Prompt text plus the small trainable embedding state it contains."""

    prompt_text: str
    token_texts: tuple[str, ...]
    token_ids: tuple[int, ...]
    learnable_embeddings: object


class EditableGenerator(Protocol):
    supports_gradient: bool

    def create_learnable_prompt(
        self,
        *,
        class_label: str,
        num_tokens: int,
        initializer: str,
        init_std: float,
    ) -> LearnablePrompt:
        """Create prompt text and trainable token embeddings for one attack."""

    def sync_learnable_prompt(self, prompt_state: LearnablePrompt) -> None:
        """Synchronize generator-owned token rows after optimizer updates."""

    def generate(
        self,
        *,
        input_image: Image.Image,
        input_tensor: object,
        prompt_state: LearnablePrompt,
        seed: int,
        require_grad: bool,
    ) -> GenerationResult:
        """Generate or edit an image."""
