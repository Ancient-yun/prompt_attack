from pathlib import Path

import pytest

from prompt_attack.config import DataConfig, load_config, parse_images_per_class, with_smoke_overrides
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
    assert config.attack.training_mode == "imagewise"
    assert config.attack.batch_size == 1
    assert config.attack.num_learnable_tokens == 64
    assert config.attack.learnable_token_initializer == "object"
    assert config.attack.learnable_token_init_std == 0.02
    assert config.attack.learnable_token_init_seed == 0
    assert config.attack.init_prompt_path is None
    assert config.attack.lr_scheduler.name == "cosine"
    assert config.attack.lr_scheduler.warmup_steps == 5
    assert config.attack.lr_scheduler.min_lr == 1.0e-4
    assert config.attack.steps == 100
    assert config.attack.lambda_sem == 0.0
    assert config.attack.semantic_loss_weight == 10.0
    assert config.attack.attack_margin == 0.0
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


def test_imagenet_root_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROMPT_ATTACK_IMAGENET_ROOT", "/tmp/b200-imagenet")

    config = load_config(Path("configs/flux2_resnet18.yaml"))

    assert config.data.imagenet_root == Path("/tmp/b200-imagenet")


def test_smoke_override_uses_mock() -> None:
    config = load_config(Path("configs/flux2_resnet18.yaml"))
    smoke = with_smoke_overrides(config, use_mock_generator=True)
    assert smoke.generator.name == "mock"
    assert smoke.attack.steps == 2
    assert smoke.attack.training_mode == config.attack.training_mode
    assert smoke.attack.batch_size == config.attack.batch_size
    assert smoke.attack.learnable_token_initializer == config.attack.learnable_token_initializer
    assert smoke.attack.learnable_token_init_std == config.attack.learnable_token_init_std
    assert smoke.attack.learnable_token_init_seed == config.attack.learnable_token_init_seed
    assert smoke.attack.init_prompt_path == config.attack.init_prompt_path
    assert smoke.attack.semantic_loss_weight == config.attack.semantic_loss_weight
    assert smoke.attack.attack_margin == config.attack.attack_margin
    assert smoke.attack.lr_scheduler.name == "cosine"
    assert not smoke.quality.fid.enabled
    assert not smoke.quality.nriqa.enabled
    assert smoke.data.images_per_class == 2
    # Regression guard: with_smoke_overrides constructs a second explicit AttackConfig(...)
    # that must thread through every new attack field, or smoke runs silently lose them.
    assert smoke.attack.eot_train_seeds == config.attack.eot_train_seeds
    assert smoke.attack.strength_schedule == config.attack.strength_schedule
    assert smoke.attack.num_anchor_tokens == config.attack.num_anchor_tokens
    assert smoke.attack.axis_lr == config.attack.axis_lr
    assert smoke.attack.train_strengths == config.attack.train_strengths
    assert smoke.attack.eval_strengths == config.attack.eval_strengths
    assert smoke.attack.legitimacy_ssim_threshold == config.attack.legitimacy_ssim_threshold
    assert (
        smoke.attack.legitimacy_semantic_threshold == config.attack.legitimacy_semantic_threshold
    )


def test_load_config_parses_strength_schedule_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
data:
  imagenet_root: /data/imagenet
generator:
  name: flux2_klein_4b
  model_id: dummy
attack:
  strength_schedule: true
  num_anchor_tokens: 3
  axis_lr: 0.05
  train_strengths: "0.3,0.7,1.0"
  eval_strengths: "0.2,0.5,1.0"
  legitimacy_ssim_threshold: 0.4
  legitimacy_semantic_threshold: 0.6
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.attack.strength_schedule is True
    assert config.attack.num_anchor_tokens == 3
    assert config.attack.axis_lr == 0.05
    assert config.attack.train_strengths == (0.3, 0.7, 1.0)
    assert config.attack.eval_strengths == (0.2, 0.5, 1.0)
    assert config.attack.legitimacy_ssim_threshold == 0.4
    assert config.attack.legitimacy_semantic_threshold == 0.6


def test_attack_config_rejects_anchor_tokens_leaving_no_axis_room() -> None:
    from prompt_attack.config import AttackConfig

    with pytest.raises(ValueError, match="leave room for at least one axis token"):
        AttackConfig(strength_schedule=True, num_learnable_tokens=4, num_anchor_tokens=4)


def test_attack_config_auto_appends_missing_full_strength(recwarn: pytest.WarningsRecorder) -> None:
    from prompt_attack.config import AttackConfig

    config = AttackConfig(strength_schedule=True, eval_strengths=(0.2, 0.5))

    assert config.eval_strengths == (0.2, 0.5, 1.0)
    assert any("eval_strengths" in str(warning.message) for warning in recwarn.list)


def test_parse_images_per_class_accepts_all() -> None:
    assert parse_images_per_class("all") is None
    assert parse_images_per_class("ALL") is None
    assert parse_images_per_class("3") == 3
    with pytest.raises(ValueError, match="positive or 'all'"):
        parse_images_per_class("0")


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


def test_fixed_10_all_uses_every_image_without_candidate_cap(tmp_path: Path) -> None:
    split = tmp_path / "train"
    split.mkdir()
    fixed_synsets = {cls.synset for cls in FIXED_10_CLASSES}
    synsets = [cls.synset for cls in FIXED_10_CLASSES]
    filler_index = 0
    while len(synsets) < 1000:
        synset = f"n{filler_index:08d}"
        filler_index += 1
        if synset in fixed_synsets:
            continue
        synsets.append(synset)
    for synset in synsets:
        class_dir = split / synset
        class_dir.mkdir()
    for cls in FIXED_10_CLASSES:
        class_dir = split / cls.synset
        for image_index in range(3):
            (class_dir / f"{cls.synset}_{image_index}.JPEG").write_bytes(b"not-opened")

    records = build_candidate_records(
        DataConfig(
            imagenet_root=tmp_path,
            split="train",
            class_mode="fixed_10",
            images_per_class=None,
            candidate_multiplier=1,
        )
    )

    assert len(records) == len(FIXED_10_CLASSES) * 3
    assert {record.synset for record in records} == fixed_synsets


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
