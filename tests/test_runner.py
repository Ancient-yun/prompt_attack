from collections.abc import Sequence
from dataclasses import replace
import json
from pathlib import Path

import torch
import pytest
from PIL import Image

from prompt_attack.attacks import runner as runner_module
from prompt_attack.attacks.runner import AttackComponents
from prompt_attack.attacks.runner import LearnableTokenAttackRunner
from prompt_attack.config import load_config
from prompt_attack.data.imagenet import ImageRecord
from prompt_attack.generators.base import (
    GenerationBatchResult,
    GenerationResult,
    LearnablePrompt,
    LearnablePromptBatch,
)
from prompt_attack.generators.mock import MockEditableGenerator
from prompt_attack.models.victim import ClassificationResult


class RaisingVictim:
    def evaluate_pil(self, image: object, true_label: int) -> object:
        del image, true_label
        raise AssertionError("victim should not be evaluated when clean_correct_only is false")


class BatchCleanVictim:
    def __init__(self, predictions: Sequence[int]) -> None:
        self.predictions = list(predictions)
        self.batch_sizes: list[int] = []

    def evaluate_pil_batch(
        self,
        images: list[Image.Image],
        true_labels: list[int],
    ) -> list[ClassificationResult]:
        del true_labels
        self.batch_sizes.append(len(images))
        offset = sum(self.batch_sizes[:-1])
        return [
            ClassificationResult(
                pred=self.predictions[offset + index],
                pred_conf=1.0,
                true_conf=1.0,
                margin=1.0,
            )
            for index in range(len(images))
        ]

    def evaluate_pil(self, image: object, true_label: int) -> object:
        del image, true_label
        raise AssertionError("clean-correct filtering should use batched victim evaluation")


def test_clean_correct_filter_skips_victim_when_disabled() -> None:
    config = load_config(Path("configs/flux2_resnet18.yaml"))
    config = replace(
        config,
        data=replace(config.data, clean_correct_only=False, images_per_class=1),
    )
    records = [
        ImageRecord(
            path=Path("does-not-exist.png"),
            synset="class_0000",
            class_label="dummy",
            class_index=0,
            image_id="dummy",
        )
    ]

    selected = LearnableTokenAttackRunner(config, device="cpu")._clean_correct_records(
        records,
        RaisingVictim(),
    )

    assert selected == records


def test_clean_correct_filter_uses_batched_victim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner_module, "CLEAN_FILTER_BATCH_SIZE", 2)
    monkeypatch.setattr(runner_module, "CLEAN_FILTER_CACHE_DIR", tmp_path / "cache")
    config = load_config(Path("configs/flux2_resnet18.yaml"))
    config = replace(
        config,
        data=replace(config.data, clean_correct_only=True, images_per_class=None),
    )
    records = []
    for index in range(3):
        image_path = tmp_path / f"sample_{index}.png"
        Image.new("RGB", (8, 8), color=(index, index, index)).save(image_path)
        records.append(
            ImageRecord(
                path=image_path,
                synset="class_0000",
                class_label="dummy",
                class_index=1,
                image_id=f"dummy_{index}",
            )
        )
    victim = BatchCleanVictim(predictions=[1, 0, 1])

    selected = LearnableTokenAttackRunner(config, device="cpu")._clean_correct_records(
        records,
        victim,
    )

    assert [record.image_id for record in selected] == ["dummy_0", "dummy_2"]
    assert victim.batch_sizes == [2, 1]


def test_clean_correct_filter_reuses_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner_module, "CLEAN_FILTER_BATCH_SIZE", 2)
    monkeypatch.setattr(runner_module, "CLEAN_FILTER_CACHE_DIR", tmp_path / "cache")
    config = load_config(Path("configs/flux2_resnet18.yaml"))
    config = replace(
        config,
        data=replace(config.data, clean_correct_only=True, images_per_class=None),
    )
    records = []
    for index in range(3):
        image_path = tmp_path / f"cached_{index}.png"
        Image.new("RGB", (8, 8), color=(index, index, index)).save(image_path)
        records.append(
            ImageRecord(
                path=image_path,
                synset="class_0000",
                class_label="dummy",
                class_index=1,
                image_id=f"cached_{index}",
            )
        )
    victim = BatchCleanVictim(predictions=[1, 0, 1])
    runner = LearnableTokenAttackRunner(config, device="cpu")

    selected = runner._clean_correct_records(records, victim)
    cached_selected = runner._clean_correct_records(records, RaisingVictim())

    assert [record.image_id for record in selected] == ["cached_0", "cached_2"]
    assert [record.image_id for record in cached_selected] == ["cached_0", "cached_2"]


def test_clean_correct_cache_metadata_contains_version(tmp_path: Path) -> None:
    config = load_config(Path("configs/flux2_resnet18.yaml"))
    config = replace(
        config,
        data=replace(config.data, imagenet_root=tmp_path, clean_correct_only=True),
    )

    _, metadata = LearnableTokenAttackRunner(config, device="cpu")._clean_correct_cache_path()

    assert metadata["cache_version"] == runner_module.CLEAN_FILTER_CACHE_VERSION
    assert metadata["preprocess_signature"] == runner_module.CLEAN_FILTER_PREPROCESS_SIGNATURE


def test_clean_correct_cache_ignores_stale_metadata(tmp_path: Path) -> None:
    config = load_config(Path("configs/flux2_resnet18.yaml"))
    config = replace(
        config,
        data=replace(config.data, imagenet_root=tmp_path, clean_correct_only=True),
    )
    runner = LearnableTokenAttackRunner(config, device="cpu")
    cache_path, metadata = runner._clean_correct_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    stale_metadata = dict(metadata)
    stale_metadata.pop("cache_version")
    cache_path.write_text(
        json.dumps(
            {
                "metadata": stale_metadata,
                "checked_keys": ["class/image"],
                "clean_keys": ["class/image"],
            },
        ),
        encoding="utf-8",
    )

    checked, clean = runner._load_clean_correct_cache(cache_path, metadata)

    assert checked == set()
    assert clean == set()


def test_clean_correct_filter_allows_uncapped_per_class_selection() -> None:
    config = load_config(Path("configs/flux2_resnet18.yaml"))
    config = replace(
        config,
        data=replace(config.data, clean_correct_only=False, images_per_class=None),
    )
    records = [
        ImageRecord(
            path=Path(f"does-not-exist-{index}.png"),
            synset="class_0000",
            class_label="dummy",
            class_index=0,
            image_id=f"dummy_{index}",
        )
        for index in range(3)
    ]

    selected = LearnableTokenAttackRunner(config, device="cpu")._clean_correct_records(
        records,
        RaisingVictim(),
    )

    assert selected == records


def test_universal_train_batches_shuffle_deterministically_each_epoch() -> None:
    config = load_config(Path("configs/flux2_resnet18.yaml"))
    runner = LearnableTokenAttackRunner(config, device="cpu")
    records = [
        ImageRecord(
            path=Path(f"image_{index}.png"),
            synset=f"class_{index:04d}",
            class_label=f"class {index}",
            class_index=index,
            image_id=f"image_{index}",
        )
        for index in range(8)
    ]

    epoch0 = runner._shuffled_record_batches(records, batch_size=2, epoch=0)
    epoch0_repeat = runner._shuffled_record_batches(records, batch_size=2, epoch=0)
    epoch1 = runner._shuffled_record_batches(records, batch_size=2, epoch=1)
    epoch0_ids = [record.image_id for batch in epoch0 for record in batch]
    epoch1_ids = [record.image_id for batch in epoch1 for record in batch]
    original_ids = [record.image_id for record in records]

    assert epoch0 == epoch0_repeat
    assert epoch0_ids != original_ids
    assert epoch1_ids != epoch0_ids
    assert sorted(epoch0_ids) == sorted(original_ids)
    assert sorted(epoch1_ids) == sorted(original_ids)


class CountingBatchGenerator(MockEditableGenerator):
    def __init__(self, *, device: str) -> None:
        super().__init__(device=device)
        self.batch_calls = 0

    def generate(
        self,
        *,
        input_image: Image.Image,
        input_tensor: object,
        prompt_state: LearnablePrompt,
        seed: int,
        require_grad: bool,
    ) -> GenerationResult:
        del input_image, input_tensor, prompt_state, seed, require_grad
        raise AssertionError("single-image generate should not be used for batch attacks")

    def generate_batch(
        self,
        *,
        input_images: Sequence[Image.Image],
        input_tensor: object,
        prompt_state: LearnablePromptBatch,
        seeds: Sequence[int],
        require_grad: bool,
    ) -> GenerationBatchResult:
        self.batch_calls += 1
        return super().generate_batch(
            input_images=input_images,
            input_tensor=input_tensor,
            prompt_state=prompt_state,
            seeds=seeds,
            require_grad=require_grad,
        )


class TinyVictim:
    categories = [f"class_{index}" for index in range(1000)]

    def logits_from_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        means = image_tensor.mean(dim=(1, 2, 3))
        logits = torch.zeros((image_tensor.shape[0], 1000), device=image_tensor.device)
        logits[:, 0] = means
        logits[:, 1] = 1.0 - means
        return logits

    def evaluate_logits(self, logits: torch.Tensor, true_label: int) -> ClassificationResult:
        probs = torch.softmax(logits, dim=-1)
        pred = int(logits.argmax(dim=-1).item())
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[0, true_label] = False
        other_max = logits.masked_select(mask).view(1, -1).max(dim=-1).values[0]
        return ClassificationResult(
            pred=pred,
            pred_conf=float(probs[0, pred].detach().cpu().item()),
            true_conf=float(probs[0, true_label].detach().cpu().item()),
            margin=float((logits[0, true_label] - other_max).detach().cpu().item()),
        )

    def evaluate_logits_batch(self, logits: torch.Tensor, true_labels: list[int]) -> list[ClassificationResult]:
        return [
            self.evaluate_logits(logits[index : index + 1], true_label)
            for index, true_label in enumerate(true_labels)
        ]


class TinySemantic:
    def similarity(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        del right
        return torch.ones(left.shape[0], device=left.device)


class EmptyQuality:
    def score_tensor(self, image_tensor: torch.Tensor) -> dict[str, float]:
        del image_tensor
        return {}


def test_attack_batch_uses_generator_batch_forward(tmp_path: Path) -> None:
    image_paths = []
    for index in range(2):
        path = tmp_path / f"image_{index}.png"
        Image.new("RGB", (16, 16), color=(128 + index, 128, 128)).save(path)
        image_paths.append(path)
    records = [
        ImageRecord(
            path=image_paths[0],
            synset="class_0000",
            class_label="class zero",
            class_index=0,
            image_id="image_0",
        ),
        ImageRecord(
            path=image_paths[1],
            synset="class_0001",
            class_label="class one",
            class_index=1,
            image_id="image_1",
        ),
    ]
    config = load_config(Path("configs/flux2_resnet18.yaml"))
    config = replace(
        config,
        generator=replace(config.generator, name="mock", model_id="mock", batch_size=2),
        attack=replace(config.attack, steps=1, num_learnable_tokens=4),
        output=replace(config.output, root=tmp_path / "outputs"),
        quality=replace(config.quality, nriqa=replace(config.quality.nriqa, enabled=False)),
    )
    generator = CountingBatchGenerator(device="cpu")
    components = AttackComponents(
        victim=TinyVictim(),
        semantic=TinySemantic(),
        dino_metric=None,
        generator=generator,
        quality_evaluator=EmptyQuality(),
    )

    rows = LearnableTokenAttackRunner(config, device="cpu").attack_batch(records, components)

    assert generator.batch_calls == 1
    assert len(rows) == 2
    assert all(row["generator_batch_size"] == 2 for row in rows)


def test_universal_prompt_trains_shared_embedding_and_evaluates(tmp_path: Path) -> None:
    image_paths = []
    for index in range(2):
        path = tmp_path / f"uap_image_{index}.png"
        Image.new("RGB", (16, 16), color=(128 + index, 128, 128)).save(path)
        image_paths.append(path)
    records = [
        ImageRecord(
            path=image_paths[0],
            synset="class_0000",
            class_label="class zero",
            class_index=0,
            image_id="uap_image_0",
        ),
        ImageRecord(
            path=image_paths[1],
            synset="class_0001",
            class_label="class one",
            class_index=1,
            image_id="uap_image_1",
        ),
    ]
    config = load_config(Path("configs/flux2_resnet18.yaml"))
    config = replace(
        config,
        generator=replace(config.generator, name="mock", model_id="mock", batch_size=1),
        attack=replace(
            config.attack,
            training_mode="universal",
            batch_size=2,
            steps=2,
            num_learnable_tokens=4,
        ),
        output=replace(config.output, root=tmp_path / "outputs"),
        quality=replace(config.quality, nriqa=replace(config.quality.nriqa, enabled=False)),
    )
    components = AttackComponents(
        victim=TinyVictim(),
        semantic=TinySemantic(),
        dino_metric=None,
        generator=MockEditableGenerator(device="cpu"),
        quality_evaluator=EmptyQuality(),
    )
    runner = LearnableTokenAttackRunner(config, device="cpu")

    prompt_state, history = runner.train_universal_prompt(
        records,
        components,
        history_path=tmp_path / "history.csv",
    )
    rows = runner.evaluate_universal_prompt(
        records,
        components,
        prompt_state,
        stage="test",
        metrics_path=tmp_path / "results.csv",
    )

    assert len(history) == 2
    assert isinstance(prompt_state.learnable_embeddings, torch.Tensor)
    assert prompt_state.learnable_embeddings.ndim == 2
    assert len(rows) == 2
    assert all(row["clean_correct"] in {True, False} for row in rows)
    assert all(row["training_mode"] == "universal" for row in rows)
    assert all(row["stage"] == "test" for row in rows)
    assert all(row["world_size"] == 1 for row in history)


def test_initial_prompt_checkpoint_replaces_universal_embeddings(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "learned_prompt.pt"
    generator = MockEditableGenerator(device="cpu")
    values = torch.full((4, generator.embedding_dim), 0.25, dtype=torch.float32)
    torch.save(
        {
            "token_texts": ("<v1>", "<v2>", "<v3>", "<v4>"),
            "token_ids": (0, 1, 2, 3),
            "learnable_embeddings": values,
            "metadata": {"run_name": "source"},
        },
        checkpoint_path,
    )
    config = load_config(Path("configs/flux2_resnet18.yaml"))
    config = replace(
        config,
        generator=replace(config.generator, name="mock", model_id="mock"),
        attack=replace(
            config.attack,
            num_learnable_tokens=4,
            init_prompt_path=checkpoint_path,
        ),
    )
    runner = LearnableTokenAttackRunner(config, device="cpu")
    prompt_state = generator.create_learnable_prompt(
        class_label="object",
        num_tokens=4,
        initializer="object",
        init_std=0.02,
    )

    loaded = runner._apply_initial_prompt_checkpoint(prompt_state)

    assert isinstance(loaded.learnable_embeddings, torch.nn.Parameter)
    assert loaded.learnable_embeddings.requires_grad
    assert torch.allclose(loaded.learnable_embeddings.detach(), values)


def test_initial_prompt_checkpoint_rejects_shape_mismatch(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "learned_prompt.pt"
    generator = MockEditableGenerator(device="cpu")
    torch.save(
        {
            "token_texts": ("<v1>", "<v2>"),
            "token_ids": (0, 1),
            "learnable_embeddings": torch.zeros((2, generator.embedding_dim)),
            "metadata": {"run_name": "source"},
        },
        checkpoint_path,
    )
    config = load_config(Path("configs/flux2_resnet18.yaml"))
    config = replace(
        config,
        generator=replace(config.generator, name="mock", model_id="mock"),
        attack=replace(
            config.attack,
            num_learnable_tokens=4,
            init_prompt_path=checkpoint_path,
        ),
    )
    runner = LearnableTokenAttackRunner(config, device="cpu")
    prompt_state = generator.create_learnable_prompt(
        class_label="object",
        num_tokens=4,
        initializer="object",
        init_std=0.02,
    )

    with pytest.raises(ValueError, match="token count mismatch"):
        runner._apply_initial_prompt_checkpoint(prompt_state)
