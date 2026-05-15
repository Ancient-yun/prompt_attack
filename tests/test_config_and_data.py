from pathlib import Path

from prompt_attack.config import load_config, with_smoke_overrides
from prompt_attack.data.imagenet import FIXED_10_CLASSES


def test_load_config() -> None:
    config = load_config(Path("configs/flux2_resnet18_imagenet10.yaml"))
    assert config.generator.name == "flux2_klein_4b"
    assert config.generator.num_inference_steps == 8
    assert config.victim.name == "resnet18"
    assert config.attack.num_soft_tokens == 8
    assert config.attack.soft_token_init_std == 0.02
    assert config.attack.lr_scheduler.name == "cosine"
    assert config.attack.lr_scheduler.warmup_steps == 5
    assert config.attack.lr_scheduler.min_lr == 1.0e-4
    assert config.attack.objective == "negative_cross_entropy"
    assert config.quality.fid.enabled
    assert str(config.quality.fid.fid_root).replace("\\", "/") == "external/pytorch_fid"
    assert config.quality.nriqa.enabled
    assert "musiq_koniq" in config.quality.nriqa.metrics
    assert len(FIXED_10_CLASSES) == 10


def test_smoke_override_uses_mock() -> None:
    config = load_config(Path("configs/flux2_resnet18_imagenet10.yaml"))
    smoke = with_smoke_overrides(config, use_mock_generator=True)
    assert smoke.generator.name == "mock"
    assert smoke.attack.steps == 2
    assert smoke.attack.soft_token_init_std == config.attack.soft_token_init_std
    assert smoke.attack.lr_scheduler.name == "cosine"
    assert not smoke.quality.fid.enabled
    assert not smoke.quality.nriqa.enabled
    assert smoke.data.images_per_class == 2
