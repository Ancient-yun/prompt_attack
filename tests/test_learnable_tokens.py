import pytest
import torch
from PIL import Image

from prompt_attack.attacks.learnable_tokens import (
    build_prompt,
    build_token_texts,
    validate_token_init_std,
)
from prompt_attack.config import GeneratorConfig
from prompt_attack.data.imagenet import FixedClass
from prompt_attack.generators.base import LearnablePromptBatch
from prompt_attack.generators import flux2 as flux2_module
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


def test_mock_generator_batch_learnable_embeddings_receive_grad_and_update() -> None:
    generator = MockEditableGenerator(device="cpu")
    prompt_state = generator.create_learnable_prompt_batch(
        class_labels=["dummy one", "dummy two"],
        num_tokens=4,
        initializer="object",
        init_std=0.1,
    )
    learnable_embeddings = prompt_state.learnable_embeddings
    assert isinstance(learnable_embeddings, torch.Tensor)
    assert learnable_embeddings.shape[0] == 2
    input_tensor = torch.full((2, 3, 8, 8), 0.5)
    images = [
        Image.new("RGB", (8, 8), color=(128, 128, 128)),
        Image.new("RGB", (8, 8), color=(128, 128, 128)),
    ]
    before = learnable_embeddings.detach().clone()
    optimizer = torch.optim.Adam([learnable_embeddings], lr=0.1)

    generated = generator.generate_batch(
        input_images=images,
        input_tensor=input_tensor,
        prompt_state=prompt_state,
        seeds=[0, 1],
        require_grad=True,
    )
    image_tensor = generated.image_tensor
    assert isinstance(image_tensor, torch.Tensor)
    assert image_tensor.shape == input_tensor.shape
    image_tensor.sum().backward()
    optimizer.step()

    assert learnable_embeddings.grad is not None
    assert not torch.allclose(learnable_embeddings.detach(), before)


def test_mock_generator_shared_batch_embeddings_receive_grad_and_update() -> None:
    generator = MockEditableGenerator(device="cpu")
    prompt_state = generator.create_learnable_prompt(
        class_label="dummy",
        num_tokens=4,
        initializer="object",
        init_std=0.1,
    )
    learnable_embeddings = prompt_state.learnable_embeddings
    assert isinstance(learnable_embeddings, torch.Tensor)
    input_tensor = torch.full((2, 3, 8, 8), 0.5)
    images = [
        Image.new("RGB", (8, 8), color=(128, 128, 128)),
        Image.new("RGB", (8, 8), color=(128, 128, 128)),
    ]
    before = learnable_embeddings.detach().clone()
    optimizer = torch.optim.Adam([learnable_embeddings], lr=0.1)
    batch_state = LearnablePromptBatch(
        prompt_texts=(prompt_state.prompt_text, prompt_state.prompt_text),
        token_texts=prompt_state.token_texts,
        token_ids=prompt_state.token_ids,
        learnable_embeddings=prompt_state.learnable_embeddings,
    )

    generated = generator.generate_batch(
        input_images=images,
        input_tensor=input_tensor,
        prompt_state=batch_state,
        seeds=[0, 1],
        require_grad=True,
    )
    image_tensor = generated.image_tensor
    assert isinstance(image_tensor, torch.Tensor)
    assert image_tensor.shape == input_tensor.shape
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


def test_flux2_random_real_token_initializer_is_seeded() -> None:
    pipe = FakePipe()
    for token in ("alpha", "beta", "gamma", "delta", "epsilon"):
        pipe.tokenizer.vocab[token] = len(pipe.tokenizer.vocab)
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    adapter = Flux2Adapter(
        GeneratorConfig(name="flux2_klein_4b", model_id="fake"),
        device="cpu",
    )
    adapter._pipe = pipe

    first = adapter.create_learnable_prompt(
        class_label="dummy",
        num_tokens=2,
        initializer="random_real_tokens",
        init_std=1e-8,
        init_seed=123,
    )
    second = adapter.create_learnable_prompt(
        class_label="dummy",
        num_tokens=2,
        initializer="random_real_tokens",
        init_std=1e-8,
        init_seed=123,
    )
    third = adapter.create_learnable_prompt(
        class_label="dummy",
        num_tokens=2,
        initializer="random_real_tokens",
        init_std=1e-8,
        init_seed=456,
    )

    first_embeddings = first.learnable_embeddings
    second_embeddings = second.learnable_embeddings
    third_embeddings = third.learnable_embeddings
    assert isinstance(first_embeddings, torch.Tensor)
    assert isinstance(second_embeddings, torch.Tensor)
    assert isinstance(third_embeddings, torch.Tensor)
    assert torch.allclose(first_embeddings, second_embeddings)
    assert not torch.allclose(first_embeddings, third_embeddings)


def test_flux2_fixed10_class_average_initializer(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = FakePipe()
    for token in ("alpha", "beta"):
        pipe.tokenizer.vocab[token] = len(pipe.tokenizer.vocab)
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    monkeypatch.setattr(
        flux2_module,
        "FIXED_10_CLASSES",
        (
            FixedClass("n00000001", "alpha"),
            FixedClass("n00000002", "beta"),
        ),
    )
    adapter = Flux2Adapter(
        GeneratorConfig(name="flux2_klein_4b", model_id="fake"),
        device="cpu",
    )
    adapter._pipe = pipe

    prompt = adapter.create_learnable_prompt(
        class_label="dummy",
        num_tokens=1,
        initializer="fixed10_class_average",
        init_std=1e-8,
        init_seed=0,
    )
    embedding = pipe.text_encoder.get_input_embeddings().weight
    expected = embedding[
        torch.tensor([pipe.tokenizer.vocab["alpha"], pipe.tokenizer.vocab["beta"]])
    ].mean(dim=0)

    prompt_embeddings = prompt.learnable_embeddings
    assert isinstance(prompt_embeddings, torch.Tensor)
    assert torch.allclose(prompt_embeddings[0], expected, atol=1e-6)


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


def test_flux2_batch_embedding_hook_routes_per_row_gradient_to_learnable_tokens() -> None:
    pipe = FakePipe()
    adapter = Flux2Adapter(
        GeneratorConfig(name="flux2_klein_4b", model_id="fake"),
        device="cpu",
    )
    adapter._pipe = pipe
    prompt_state = adapter.create_learnable_prompt_batch(
        class_labels=["dummy one", "dummy two"],
        num_tokens=1,
        initializer="object",
        init_std=0.01,
    )
    learnable_embeddings = prompt_state.learnable_embeddings
    assert isinstance(learnable_embeddings, torch.Tensor)

    embedding_layer = pipe.text_encoder.get_input_embeddings()
    input_ids = torch.tensor(
        [
            [prompt_state.token_ids[0], 0],
            [0, prompt_state.token_ids[0]],
        ]
    )
    handle = adapter._register_learnable_embedding_hook(prompt_state)
    try:
        embedding_layer(input_ids).sum().backward()
    finally:
        handle.remove()

    assert learnable_embeddings.grad is not None
    assert learnable_embeddings.grad.shape == learnable_embeddings.shape
    assert torch.all(learnable_embeddings.grad.abs().sum(dim=(1, 2)) > 0)


class FakeLatentPipe:
    def _encode_vae_image(self, image: torch.Tensor, generator: object) -> torch.Tensor:
        del generator
        return image

    def _pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = latents.shape
        return latents.reshape(batch, channels, height * width).permute(0, 2, 1)


def test_flux2_per_sample_image_latents_do_not_share_references() -> None:
    first = torch.zeros((1, 3, 4, 4))
    second = torch.ones((1, 3, 4, 4))

    latents, latent_ids = Flux2Adapter._prepare_per_sample_image_latents(
        FakeLatentPipe(),
        [first, second],
        batch_size=2,
        generator=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert latents.shape == (2, 16, 3)
    assert latent_ids.shape == (2, 16, 4)
    assert torch.all(latents[0] == 0)
    assert torch.all(latents[1] == 1)
