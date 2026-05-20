from pathlib import Path

import pytest
import torch
from PIL import Image

from prompt_attack.attacks.soft_tokens import build_prompt, build_token_texts, validate_token_init_std
from prompt_attack.config import GeneratorConfig
from prompt_attack.generators.flux2 import Flux2Adapter
from prompt_attack.generators.mock import MockEditableGenerator


def test_build_prompt_uses_textual_inversion_tokens() -> None:
    assert build_token_texts(3) == ("<v1>", "<v2>", "<v3>")
    assert build_prompt("folding chair", 3) == "<v1> <v2> <v3> a photo of folding chair"


def test_build_token_texts_rejects_non_positive_count() -> None:
    with pytest.raises(ValueError, match="positive"):
        build_token_texts(0)


def test_token_init_std_rejects_non_positive_std() -> None:
    with pytest.raises(ValueError, match="positive"):
        validate_token_init_std(0.0)


def test_mock_generator_learnable_embeddings_receive_grad_and_update() -> None:
    generator = MockEditableGenerator(device="cpu")
    prompt_state = generator.create_learnable_prompt(
        class_label="dummy",
        num_tokens=4,
        initializer="object",
        init_std=0.1,
    )
    learnable_embeddings = prompt_state.learnable_embeddings
    assert isinstance(learnable_embeddings, torch.Tensor)
    input_tensor = torch.full((1, 3, 8, 8), 0.5)
    image = Image.new("RGB", (8, 8), color=(128, 128, 128))
    before = learnable_embeddings.detach().clone()
    optimizer = torch.optim.Adam([learnable_embeddings], lr=0.1)

    generated = generator.generate(
        input_image=image,
        input_tensor=input_tensor,
        prompt_state=prompt_state,
        seed=0,
        require_grad=True,
    )
    image_tensor = generated.image_tensor
    assert isinstance(image_tensor, torch.Tensor)
    image_tensor.sum().backward()
    optimizer.step()

    assert learnable_embeddings.grad is not None
    assert not torch.allclose(learnable_embeddings.detach(), before)


class FakeTokenizer:
    def __init__(self) -> None:
        self.vocab = {"object": 0}
        self.unk_token_id = None

    def __len__(self) -> int:
        return len(self.vocab)

    def add_tokens(self, tokens: list[str]) -> int:
        added = 0
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
        return added

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self.vocab.get(token)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        return [self.vocab[token] for token in text.split()]


class FakeTextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = torch.nn.Embedding(1, 5)
        self.resize_calls = 0

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.embeddings

    def resize_token_embeddings(self, size: int) -> torch.nn.Embedding:
        previous = self.embeddings
        self.embeddings = torch.nn.Embedding(size, previous.embedding_dim)
        with torch.no_grad():
            self.embeddings.weight[: previous.num_embeddings].copy_(previous.weight)
        self.resize_calls += 1
        return self.embeddings


class FakePipe:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.text_encoder = FakeTextEncoder()
        self.transformer = None
        self.vae = None


def test_flux2_token_addition_is_idempotent() -> None:
    pipe = FakePipe()
    adapter = Flux2Adapter(
        GeneratorConfig(name="flux2_klein_4b", model_id="fake"),
        device="cpu",
    )
    adapter._pipe = pipe

    first = adapter.create_learnable_prompt(
        class_label="dummy",
        num_tokens=2,
        initializer="object",
        init_std=0.01,
    )
    second = adapter.create_learnable_prompt(
        class_label="dummy",
        num_tokens=2,
        initializer="object",
        init_std=0.01,
    )

    assert first.token_texts == ("<v1>", "<v2>")
    assert first.token_ids == second.token_ids
    assert len(pipe.tokenizer) == 3
    assert pipe.text_encoder.resize_calls == 1


def test_flux2_embedding_hook_routes_gradient_to_learnable_tokens() -> None:
    pipe = FakePipe()
    adapter = Flux2Adapter(
        GeneratorConfig(name="flux2_klein_4b", model_id="fake"),
        device="cpu",
    )
    adapter._pipe = pipe
    prompt_state = adapter.create_learnable_prompt(
        class_label="dummy",
        num_tokens=1,
        initializer="object",
        init_std=0.01,
    )
    learnable_embeddings = prompt_state.learnable_embeddings
    assert isinstance(learnable_embeddings, torch.Tensor)

    embedding_layer = pipe.text_encoder.get_input_embeddings()
    input_ids = torch.tensor([[prompt_state.token_ids[0], 0]])
    handle = adapter._register_learnable_embedding_hook(prompt_state)
    try:
        embedding_layer(input_ids).sum().backward()
    finally:
        handle.remove()

    assert learnable_embeddings.grad is not None
    assert learnable_embeddings.grad.abs().sum() > 0


def test_time_attack_does_not_reference_removed_soft_token_dim_api() -> None:
    text = Path("scripts/time_attack.py").read_text(encoding="utf-8")
    assert "soft_token_dim" not in text
