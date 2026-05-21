"""Experiment configuration loading and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    imagenet_root: Path
    split: str = "val"
    class_mode: str = "fixed_10"
    images_per_class: int = 20
    clean_correct_only: bool = True
    candidate_multiplier: int = 5


@dataclass(frozen=True)
class GeneratorConfig:
    name: str
    model_id: str
    precision: str = "bf16"
    batch_size: int = 1
    seed_per_image: bool = True
    memory_mode: str = "vram24"
    height: int = 1024
    width: int = 1024
    guidance_scale: float = 1.0
    num_inference_steps: int = 4
    use_cpu_offload: bool = True
    gradient_checkpointing: bool = True


@dataclass(frozen=True)
class VictimConfig:
    name: str = "resnet50"
    weights: str = "IMAGENET1K_V2"


@dataclass(frozen=True)
class SemanticConfig:
    name: str = "dinov2_vitb14"
    feature: str = "cls"


@dataclass(frozen=True)
class LRSchedulerConfig:
    name: str = "fixed"
    warmup_steps: int = 0
    min_lr: float = 0.0


@dataclass(frozen=True)
class AttackConfig:
    num_learnable_tokens: int = 8
    learnable_token_initializer: str = "object"
    learnable_token_init_std: float = 0.02
    lr: float = 1.0e-2
    steps: int = 100
    lambda_sem: float = 0.5
    semantic_threshold: float = 0.85
    objective: str = "untargeted_margin"
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)


@dataclass(frozen=True)
class OutputConfig:
    root: Path = Path("outputs/flux2_klein_4b")
    save_grids: bool = True
    save_images: bool = True


@dataclass(frozen=True)
class FIDConfig:
    enabled: bool = False
    fid_root: Path = Path("external/pytorch_fid")
    batch_size: int = 50
    dims: int = 2048
    fail_on_error: bool = False


@dataclass(frozen=True)
class NRIQAConfig:
    enabled: bool = False
    metrics: tuple[str, ...] = (
        "nima_ava",
        "hyperiqa",
        "musiq_ava",
        "musiq_koniq",
        "tres",
    )
    fail_on_error: bool = False


@dataclass(frozen=True)
class QualityConfig:
    fid: FIDConfig = field(default_factory=FIDConfig)
    nriqa: NRIQAConfig = field(default_factory=NRIQAConfig)


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool = False
    project: str = "prompt-attack"
    entity: str | None = None
    name: str | None = None
    mode: str = "offline"
    dir: Path = Path("outputs/wandb")
    tags: tuple[str, ...] = ()
    log_every_steps: int = 1
    log_images: bool = False


@dataclass(frozen=True)
class LoggingConfig:
    wandb: WandbConfig = WandbConfig()


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    generator: GeneratorConfig
    victim: VictimConfig
    semantic: SemanticConfig
    attack: AttackConfig
    quality: QualityConfig
    output: OutputConfig
    logging: LoggingConfig


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{name}' must be a mapping.")
    return value


def load_config(path: Path) -> ExperimentConfig:
    """Load an experiment config from YAML."""
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {path}")

    data_raw = _section(raw, "data")
    generator_raw = _section(raw, "generator")
    victim_raw = _section(raw, "victim")
    semantic_raw = _section(raw, "semantic")
    attack_raw = _section(raw, "attack")
    output_raw = _section(raw, "output")
    quality_raw = _section(raw, "quality")
    fid_raw = _section(quality_raw, "fid")
    nriqa_raw = _section(quality_raw, "nriqa")
    logging_raw = _section(raw, "logging")
    wandb_raw = _section(logging_raw, "wandb")
    lr_scheduler_raw = attack_raw.get("lr_scheduler", {})
    if lr_scheduler_raw is None:
        lr_scheduler_raw = {}
    if not isinstance(lr_scheduler_raw, dict):
        raise ValueError("Config field 'attack.lr_scheduler' must be a mapping.")
    wandb_tags_raw = wandb_raw.get("tags", ())
    if wandb_tags_raw is None:
        wandb_tags_raw = ()
    wandb_tags: tuple[str, ...]
    if isinstance(wandb_tags_raw, str):
        wandb_tags = (wandb_tags_raw,)
    else:
        wandb_tags = tuple(str(tag) for tag in wandb_tags_raw)
    nriqa_metrics_raw = nriqa_raw.get(
        "metrics",
        ("nima_ava", "hyperiqa", "musiq_ava", "musiq_koniq", "tres"),
    )
    if nriqa_metrics_raw is None:
        nriqa_metrics_raw = ()
    if isinstance(nriqa_metrics_raw, str):
        nriqa_metrics = tuple(
            metric.strip() for metric in nriqa_metrics_raw.split(",") if metric.strip()
        )
    else:
        nriqa_metrics = tuple(str(metric) for metric in nriqa_metrics_raw)

    split_raw = data_raw.get("split", "val")
    imagenet_root_raw = os.environ.get("PROMPT_ATTACK_IMAGENET_ROOT", data_raw["imagenet_root"])
    data = DataConfig(
        imagenet_root=Path(os.path.expandvars(str(imagenet_root_raw))),
        split="" if split_raw in {None, ""} else str(split_raw),
        class_mode=str(data_raw.get("class_mode", "fixed_10")),
        images_per_class=int(data_raw.get("images_per_class", 20)),
        clean_correct_only=bool(data_raw.get("clean_correct_only", True)),
        candidate_multiplier=int(data_raw.get("candidate_multiplier", 5)),
    )
    generator = GeneratorConfig(
        name=str(generator_raw["name"]),
        model_id=str(generator_raw["model_id"]),
        precision=str(generator_raw.get("precision", "bf16")),
        batch_size=int(generator_raw.get("batch_size", 1)),
        seed_per_image=bool(generator_raw.get("seed_per_image", True)),
        memory_mode=str(generator_raw.get("memory_mode", "vram24")),
        height=int(generator_raw.get("height", 1024)),
        width=int(generator_raw.get("width", 1024)),
        guidance_scale=float(generator_raw.get("guidance_scale", 1.0)),
        num_inference_steps=int(generator_raw.get("num_inference_steps", 4)),
        use_cpu_offload=bool(generator_raw.get("use_cpu_offload", True)),
        gradient_checkpointing=bool(generator_raw.get("gradient_checkpointing", True)),
    )
    return ExperimentConfig(
        data=data,
        generator=generator,
        victim=VictimConfig(
            name=str(victim_raw.get("name", "resnet50")),
            weights=str(victim_raw.get("weights", "IMAGENET1K_V2")),
        ),
        semantic=SemanticConfig(
            name=str(semantic_raw.get("name", "dinov2_vitb14")),
            feature=str(semantic_raw.get("feature", "cls")),
        ),
        attack=AttackConfig(
            num_learnable_tokens=int(attack_raw.get("num_learnable_tokens", 8)),
            learnable_token_initializer=str(
                attack_raw.get("learnable_token_initializer", "object")
            ),
            learnable_token_init_std=float(attack_raw.get("learnable_token_init_std", 0.02)),
            lr=float(attack_raw.get("lr", 1.0e-2)),
            steps=int(attack_raw.get("steps", 100)),
            lambda_sem=float(attack_raw.get("lambda_sem", 0.5)),
            semantic_threshold=float(attack_raw.get("semantic_threshold", 0.85)),
            objective=str(attack_raw.get("objective", "untargeted_margin")),
            lr_scheduler=LRSchedulerConfig(
                name=str(lr_scheduler_raw.get("name", "fixed")),
                warmup_steps=max(0, int(lr_scheduler_raw.get("warmup_steps", 0))),
                min_lr=float(lr_scheduler_raw.get("min_lr", 0.0)),
            ),
        ),
        quality=QualityConfig(
            fid=FIDConfig(
                enabled=bool(fid_raw.get("enabled", False)),
                fid_root=Path(str(fid_raw.get("fid_root", "external/pytorch_fid"))),
                batch_size=max(1, int(fid_raw.get("batch_size", 50))),
                dims=int(fid_raw.get("dims", 2048)),
                fail_on_error=bool(fid_raw.get("fail_on_error", False)),
            ),
            nriqa=NRIQAConfig(
                enabled=bool(nriqa_raw.get("enabled", False)),
                metrics=nriqa_metrics,
                fail_on_error=bool(nriqa_raw.get("fail_on_error", False)),
            ),
        ),
        output=OutputConfig(
            root=Path(str(output_raw.get("root", "outputs/flux2_klein_4b"))),
            save_grids=bool(output_raw.get("save_grids", True)),
            save_images=bool(output_raw.get("save_images", True)),
        ),
        logging=LoggingConfig(
            wandb=WandbConfig(
                enabled=bool(wandb_raw.get("enabled", False)),
                project=str(wandb_raw.get("project", "prompt-attack")),
                entity=(
                    None
                    if wandb_raw.get("entity") in {None, ""}
                    else str(wandb_raw.get("entity"))
                ),
                name=None if wandb_raw.get("name") in {None, ""} else str(wandb_raw.get("name")),
                mode=str(wandb_raw.get("mode", "offline")),
                dir=Path(str(wandb_raw.get("dir", "outputs/wandb"))),
                tags=wandb_tags,
                log_every_steps=max(1, int(wandb_raw.get("log_every_steps", 1))),
                log_images=bool(wandb_raw.get("log_images", False)),
            )
        ),
    )


def with_smoke_overrides(config: ExperimentConfig, *, use_mock_generator: bool) -> ExperimentConfig:
    """Return a tiny config for smoke tests."""
    generator = config.generator
    if use_mock_generator:
        generator = GeneratorConfig(
            name="mock",
            model_id="mock",
            precision=config.generator.precision,
            batch_size=1,
            seed_per_image=True,
            memory_mode="mock",
            height=224,
            width=224,
            guidance_scale=1.0,
            num_inference_steps=1,
            use_cpu_offload=False,
            gradient_checkpointing=False,
        )
    return ExperimentConfig(
        data=DataConfig(
            imagenet_root=config.data.imagenet_root,
            split=config.data.split,
            class_mode=config.data.class_mode,
            images_per_class=2,
            clean_correct_only=config.data.clean_correct_only,
            candidate_multiplier=2,
        ),
        generator=generator,
        victim=config.victim,
        semantic=config.semantic,
        attack=AttackConfig(
            num_learnable_tokens=config.attack.num_learnable_tokens,
            learnable_token_initializer=config.attack.learnable_token_initializer,
            learnable_token_init_std=config.attack.learnable_token_init_std,
            lr=config.attack.lr,
            steps=2,
            lambda_sem=config.attack.lambda_sem,
            semantic_threshold=config.attack.semantic_threshold,
            objective=config.attack.objective,
            lr_scheduler=config.attack.lr_scheduler,
        ),
        quality=QualityConfig(
            fid=FIDConfig(
                enabled=False,
                fid_root=config.quality.fid.fid_root,
                batch_size=config.quality.fid.batch_size,
                dims=config.quality.fid.dims,
                fail_on_error=config.quality.fid.fail_on_error,
            ),
            nriqa=NRIQAConfig(
                enabled=False,
                metrics=config.quality.nriqa.metrics,
                fail_on_error=config.quality.nriqa.fail_on_error,
            ),
        ),
        output=OutputConfig(
            root=config.output.root / "smoke",
            save_grids=config.output.save_grids,
            save_images=config.output.save_images,
        ),
        logging=LoggingConfig(
            wandb=WandbConfig(
                enabled=config.logging.wandb.enabled,
                project=config.logging.wandb.project,
                entity=config.logging.wandb.entity,
                name=(
                    f"{config.logging.wandb.name}-smoke"
                    if config.logging.wandb.name
                    else "smoke"
                ),
                mode=config.logging.wandb.mode,
                dir=config.logging.wandb.dir,
                tags=(*config.logging.wandb.tags, "smoke"),
                log_every_steps=config.logging.wandb.log_every_steps,
                log_images=config.logging.wandb.log_images,
            )
        ),
    )
