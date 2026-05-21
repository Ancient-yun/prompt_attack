from pathlib import Path

import pytest

from prompt_attack.config import load_config, with_smoke_overrides
from prompt_attack.config import DataConfig
from prompt_attack.data import imagenet as imagenet_module
from prompt_attack.data.imagenet import FIXED_10_CLASSES, build_candidate_records


def test_load_config() -> None:
    config = load_config(Path("configs/flux2_resnet18.yaml"))
    assert config.generator.name == "flux2_klein_4b"
    assert config.generator.guidance_scale == 1.0
    assert config.generator.num_inference_steps == 4
    assert config.data.split == "train"
    assert config.data.class_mode == "imagenet_folder"
    assert config.victim.name == "resnet18"
    assert config.attack.num_soft_tokens == 64
    assert config.attack.soft_token_initializer == "object"
    assert config.attack.soft_token_init_std == 0.02
    assert config.attack.lr_scheduler.name == "cosine"
    assert config.attack.lr_scheduler.warmup_steps == 5
    assert config.attack.lr_scheduler.min_lr == 1.0e-4
    assert config.attack.steps == 100
    assert config.attack.lambda_sem == 0.0
    assert config.attack.objective == "cr"
    assert config.quality.fid.enabled
    assert str(config.quality.fid.fid_root).replace("\\", "/") == "external/pytorch_fid"
    assert config.quality.nriqa.enabled
    assert "musiq_koniq" in config.quality.nriqa.metrics
    assert len(FIXED_10_CLASSES) == 10


def test_load_eval_config_uses_imagenet_val_folder() -> None:
    config = load_config(Path("configs/flux2_resnet18_eval.yaml"))

    assert config.data.split == "val"
    assert config.data.class_mode == "imagenet_folder"
    assert config.output.root.name.endswith("imagenet_val")


def test_smoke_override_uses_mock() -> None:
    config = load_config(Path("configs/flux2_resnet18.yaml"))
    smoke = with_smoke_overrides(config, use_mock_generator=True)
    assert smoke.generator.name == "mock"
    assert smoke.attack.steps == 2
    assert smoke.attack.soft_token_initializer == config.attack.soft_token_initializer
    assert smoke.attack.soft_token_init_std == config.attack.soft_token_init_std
    assert smoke.attack.lr_scheduler.name == "cosine"
    assert not smoke.quality.fid.enabled
    assert not smoke.quality.nriqa.enabled
    assert smoke.data.images_per_class == 2


def test_imagenet_folder_records_use_sorted_synset_indices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split = tmp_path / "val"
    split.mkdir()
    synsets = [f"n{i:08d}" for i in range(1000)]
    for synset in synsets:
        class_dir = split / synset
        class_dir.mkdir()
        (class_dir / f"{synset}_sample.JPEG").write_bytes(b"not-opened")
    labels = tuple(f"label_{index}" for index in range(1000))
    monkeypatch.setattr(imagenet_module, "imagenet_categories", lambda: labels)

    records = build_candidate_records(
        DataConfig(
            imagenet_root=tmp_path,
            split="val",
            class_mode="imagenet_folder",
            images_per_class=1,
            candidate_multiplier=1,
        )
    )

    assert len(records) == 1000
    assert records[0].synset == synsets[0]
    assert records[0].class_index == 0
    assert records[0].class_label == "label_0"
    assert records[-1].synset == synsets[-1]
    assert records[-1].class_index == 999
    assert records[-1].class_label == "label_999"


def test_imagenet_folder_requires_1000_class_dirs(tmp_path: Path) -> None:
    split = tmp_path / "train"
    split.mkdir()
    for index in range(999):
        (split / f"n{index:08d}").mkdir()

    with pytest.raises(ValueError, match="Expected 1000 ImageNet class folders"):
        build_candidate_records(
            DataConfig(
                imagenet_root=tmp_path,
                split="train",
                class_mode="imagenet_folder",
            )
        )


def test_csv_images_mode_still_loads_existing_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (tmp_path / "images.csv").write_text("ImageId,TrueLabel\nsample,2\n", encoding="utf-8")
    (image_dir / "sample.png").write_bytes(b"not-opened")
    monkeypatch.setattr(
        imagenet_module,
        "imagenet_categories",
        lambda: tuple(f"label_{index}" for index in range(1000)),
    )

    records = build_candidate_records(
        DataConfig(
            imagenet_root=tmp_path,
            split="",
            class_mode="csv_images",
        )
    )

    assert len(records) == 1
    assert records[0].image_id == "sample"
    assert records[0].class_index == 1
    assert records[0].class_label == "label_1"
