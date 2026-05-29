import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

from prompt_attack.config import load_config
from prompt_attack.utils.wandb_logger import WandbLogger


class FakeRun:
    def __init__(self) -> None:
        self.logs: list[dict[str, Any]] = []
        self.summary: dict[str, Any] = {}

    def define_metric(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def log(self, payload: dict[str, Any], *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.logs.append(payload)

    def finish(self) -> None:
        pass


def test_wandb_config_name_takes_precedence_over_stale_env(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = load_config(Path("configs/flux2_resnet18.yaml"))
    captured: dict[str, Any] = {}

    def fake_init(**kwargs: Any) -> FakeRun:
        captured.update(kwargs)
        return FakeRun()

    fake_wandb = SimpleNamespace(init=fake_init)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setenv("WANDB_NAME", "stale-run-name")
    config = load_config(Path("configs/flux2_resnet18.yaml"))
    config = config.__class__(
        data=config.data,
        generator=config.generator,
        victim=config.victim,
        semantic=config.semantic,
        attack=config.attack,
        quality=config.quality,
        output=config.output,
        logging=config.logging.__class__(
            wandb=config.logging.wandb.__class__(
                enabled=True,
                project=config.logging.wandb.project,
                entity=config.logging.wandb.entity,
                name=config.logging.wandb.name,
                mode="disabled",
                dir=tmp_path,
                tags=config.logging.wandb.tags,
                log_every_steps=config.logging.wandb.log_every_steps,
                log_images=config.logging.wandb.log_images,
            )
        ),
    )

    logger = WandbLogger(config)
    logger.start()
    logger.finish()

    assert captured["name"] == config.logging.wandb.name


def test_wandb_image_table_flushes_once_at_finish(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = load_config(Path("configs/flux2_resnet18.yaml"))
    fake_run = FakeRun()

    def fake_init(**kwargs: Any) -> FakeRun:
        del kwargs
        return fake_run

    fake_wandb = SimpleNamespace(
        init=fake_init,
        Table=lambda **kwargs: {"table": kwargs},
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    config = config.__class__(
        data=config.data,
        generator=config.generator,
        victim=config.victim,
        semantic=config.semantic,
        attack=config.attack,
        quality=config.quality,
        output=config.output,
        logging=config.logging.__class__(
            wandb=config.logging.wandb.__class__(
                enabled=True,
                project=config.logging.wandb.project,
                entity=config.logging.wandb.entity,
                name=config.logging.wandb.name,
                mode="disabled",
                dir=tmp_path,
                tags=config.logging.wandb.tags,
                log_every_steps=config.logging.wandb.log_every_steps,
                log_images=False,
            )
        ),
    )
    row = {
        "success": True,
        "clean_true_conf": 0.9,
        "adv_true_conf": 0.1,
        "confidence_drop": 0.8,
        "clean_top1_conf": 0.9,
        "adv_top1_conf": 0.7,
        "clean_margin": 1.0,
        "adv_margin": -1.0,
        "margin_drop": 2.0,
        "semantic_similarity": 0.8,
        "dino_similarity": "",
        "clip_image_similarity": 0.8,
        "ssim": 0.7,
        "pixel_l1_mean": 0.1,
        "pixel_l2": 1.0,
        "pixel_l2_mean": 0.2,
        "pixel_linf": 0.3,
        "best_step": 1,
        "best_attack_step": 1,
        "first_success_step": 1,
        "min_adv_true_conf": 0.1,
        "min_adv_margin": -1.0,
        "runtime_seconds": 2.0,
        "class_label": "class zero",
        "clean_pred_label": "class zero",
        "adv_pred_label": "class one",
        "image_id": "img0",
        "objective": "margin_clip_img2img",
        "semantic_model": "clip_vit_b32",
        "semantic_metric": "clip_image_similarity",
        "original_image_path": "original.jpg",
        "output_image_path": "adv.jpg",
        "iqa_nima_ava": "",
        "iqa_hyperiqa": "",
        "iqa_musiq_ava": "",
        "iqa_musiq_koniq": "",
        "iqa_tres": "",
    }

    logger = WandbLogger(config)
    logger.start()
    image = Image.new("RGB", (4, 4), color=(128, 128, 128))
    logger.log_image_result(row=row, original=image, adversarial=image)

    assert not any("image_results/table" in payload for payload in fake_run.logs)

    logger.finish()

    table_logs = [payload for payload in fake_run.logs if "image_results/table" in payload]
    assert len(table_logs) == 1
