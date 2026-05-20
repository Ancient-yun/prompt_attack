import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from prompt_attack.config import load_config
from prompt_attack.utils.wandb_logger import WandbLogger


class FakeRun:
    def define_metric(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

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
