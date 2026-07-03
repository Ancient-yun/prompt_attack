"""Generator adapter interfaces."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from PIL import Image

if TYPE_CHECKING:
    from prompt_attack.attacks.axis_tokens import AxisPromptState


@dataclass(frozen=True)
class GenerationResult:
    image_tensor: object
    pil_image: Image.Image


@dataclass(frozen=True)
class GenerationBatchResult:
    image_tensor: object
    pil_images: list[Image.Image]


@dataclass(frozen=True)
class LearnablePrompt:
    """Prompt text plus the small trainable embedding state it contains."""

    prompt_text: str
    token_texts: tuple[str, ...]
    token_ids: tuple[int, ...]
    learnable_embeddings: object


@dataclass(frozen=True)
class LearnablePromptBatch:
    """Batch prompts plus trainable embedding state.

    ``learnable_embeddings`` may be per-sample ``[B, N, D]`` for imagewise attacks or shared
    ``[N, D]`` for universal attacks.
    """

    prompt_texts: tuple[str, ...]
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
        init_seed: int = 0,
    ) -> LearnablePrompt:
        """Create prompt text and trainable token embeddings for one attack."""

    def create_learnable_prompt_batch(
        self,
        *,
        class_labels: Sequence[str],
        num_tokens: int,
        initializer: str,
        init_std: float,
        init_seed: int = 0,
    ) -> LearnablePromptBatch:
        """Create prompt texts and trainable token embeddings for a batch attack."""

    def sync_learnable_prompt(self, prompt_state: LearnablePrompt) -> None:
        """Synchronize generator-owned token rows after optimizer updates."""

    def sync_learnable_prompt_batch(self, prompt_state: LearnablePromptBatch) -> None:
        """Synchronize generator-owned token rows after batch optimizer updates."""

    def create_axis_prompt(
        self,
        *,
        class_label: str,
        num_tokens: int,
        num_anchor_tokens: int,
        initializer: str,
        init_std: float,
        init_seed: int = 0,
    ) -> "AxisPromptState":
        """Create anchor/axis token embeddings for a strength-scheduled attack."""

    def sync_axis_prompt(self, state: "AxisPromptState", *, t: float = 1.0) -> None:
        """Synchronize generator-owned token rows for an axis prompt at strength ``t``."""

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

    def generate_batch(
        self,
        *,
        input_images: Sequence[Image.Image],
        input_tensor: object,
        prompt_state: LearnablePromptBatch,
        seeds: Sequence[int],
        require_grad: bool,
    ) -> GenerationBatchResult:
        """Generate or edit a batch of images."""
