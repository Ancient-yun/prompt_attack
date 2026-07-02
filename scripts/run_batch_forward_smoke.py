"""Run a small FLUX batch-forward train/val smoke experiment."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import torch

from prompt_attack.attacks.runner import LearnableTokenAttackRunner
from prompt_attack.config import (
    DataConfig,
    FIDConfig,
    LoggingConfig,
    NRIQAConfig,
    OutputConfig,
    QualityConfig,
    WandbConfig,
    load_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/flux2_resnet18.yaml"))
    parser.add_argument("--root", type=Path, default=Path("outputs/local_batch_forward"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--images-per-class", type=int, default=1)
    parser.add_argument("--candidate-multiplier", type=int, default=5)
    parser.add_argument("--train-max-images", type=int, default=2)
    parser.add_argument("--val-max-images", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-mode", default="online", choices=("online", "offline", "disabled"))
    parser.add_argument("--wandb-project", default="prompt-learnable-token-attack")
    return parser.parse_args()


def nvidia_smi(label: str) -> None:
    print(f"\n===== nvidia-smi: {label} =====", flush=True)
    subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used",
            "--format=csv,noheader",
        ],
        check=False,
    )
    print("================================\n", flush=True)


def build_stage_config(
    args: argparse.Namespace,
    *,
    split: str,
    name: str,
):
    base = load_config(args.config)
    wandb = base.logging.wandb
    return replace(
        base,
        data=DataConfig(
            imagenet_root=base.data.imagenet_root,
            split=split,
            class_mode="fixed_10",
            images_per_class=args.images_per_class,
            clean_correct_only=True,
            candidate_multiplier=args.candidate_multiplier,
        ),
        generator=replace(
            base.generator,
            batch_size=args.batch_size,
            height=512,
            width=512,
            num_inference_steps=args.num_inference_steps,
            use_cpu_offload=False,
        ),
        attack=replace(base.attack, batch_size=args.batch_size, steps=args.steps),
        output=OutputConfig(root=args.root / name, save_grids=True, save_images=True),
        quality=QualityConfig(
            fid=FIDConfig(enabled=False, fid_root=base.quality.fid.fid_root),
            nriqa=NRIQAConfig(enabled=False, metrics=base.quality.nriqa.metrics),
        ),
        logging=LoggingConfig(
            wandb=WandbConfig(
                enabled=args.wandb_mode != "disabled",
                project=args.wandb_project,
                entity=wandb.entity,
                name=name,
                mode=args.wandb_mode,
                dir=wandb.dir,
                tags=(
                    "batch-forward",
                    split,
                    f"batch-size-{args.batch_size}",
                    f"steps-{args.steps}",
                    f"inference-steps-{args.num_inference_steps}",
                ),
                log_every_steps=wandb.log_every_steps,
                log_images=wandb.log_images,
            )
        ),
    )


def run_stage(args: argparse.Namespace, *, split: str, max_images: int) -> None:
    name = (
        f"batch_forward_{split}_bs{args.batch_size}_max{max_images}_"
        f"steps{args.steps}_nis{args.num_inference_steps}"
    )
    print(f"\n===== stage start: {name} =====", flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    nvidia_smi(f"before {name}")
    started = time.perf_counter()
    config = build_stage_config(args, split=split, name=name)
    rows = LearnableTokenAttackRunner(config, device=args.device).run(max_images=max_images)
    elapsed = time.perf_counter() - started
    peak_gb = 0.0
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    nvidia_smi(f"after {name}")
    print(
        f"===== stage done: {name} | rows={len(rows)} | elapsed_sec={elapsed:.1f} | "
        f"peak_allocated_gb={peak_gb:.2f} | output={config.output.root} =====",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_MODE"] = args.wandb_mode
    print("batch-forward train/val smoke", flush=True)
    print(f"cwd={Path.cwd()} batch_size={args.batch_size}", flush=True)
    run_stage(args, split="train", max_images=args.train_max_images)
    run_stage(args, split="val", max_images=args.val_max_images)


if __name__ == "__main__":
    main()
