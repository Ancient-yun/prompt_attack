"""Generator factory."""

from __future__ import annotations

from prompt_attack.config import GeneratorConfig
from prompt_attack.generators.flux2 import Flux2Adapter
from prompt_attack.generators.mock import MockEditableGenerator


def build_generator(config: GeneratorConfig, *, device: str):
    """Build an editable generator adapter."""
    if config.name == "mock":
        return MockEditableGenerator(device=device)
    if config.name == "flux2_klein_4b":
        return Flux2Adapter(config, device=device)
    raise ValueError(f"Unsupported generator: {config.name}")

