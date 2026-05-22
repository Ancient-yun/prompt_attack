"""Train and evaluate a fixed-10-class universal learnable-token prompt."""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from prompt_attack.attacks.runner import LearnableTokenAttackRunner
from prompt_attack.config import (
    DataConfig,
    FIDConfig,
    GeneratorConfig,
    LoggingConfig,
    NRIQAConfig,
    OutputConfig,
    QualityConfig,
    WandbConfig,
    load_config,
    parse_images_per_class,
)
from prompt_attack.metrics.fid import compute_fid_for_rows
from prompt_attack.metrics.summary import summarize_rows
from prompt_attack.utils.distributed import DistributedContext
from prompt_attack.utils.io import append_csv_row, ensure_dir, write_json
from prompt_attack.utils.process_title import set_process_title
from prompt_attack.utils.wandb_logger import WandbLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/flux2_resnet18.yaml"))
    parser.add_argument("--root", type=Path, default=Path("outputs/uap_fixed10"))
    parser.add_argument("--imagenet-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-images-per-class", default="10")
    parser.add_argument("--test-images-per-class", default="10")
    parser.add_argument("--train-max-images", type=int)
    parser.add_argument("--test-max-images", type=int)
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--attack-batch-size", type=int)
    parser.add_argument("--generator-batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--num-tokens", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--lambda-sem", type=float, default=0.0)
    parser.add_argument("--objective", default="cr")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--generator", default="flux2", choices=("flux2", "mock"))
    parser.add_argument("--save-train-images", action="store_true")
    parser.add_argument(
        "--include-clean-incorrect",
        action="store_true",
        help="Include images the victim misclassifies before attack. Default excludes them.",
    )
    parser.add_argument("--wandb-mode", default="online", choices=("online", "offline", "disabled"))
    parser.add_argument("--wandb-project", default="prompt-learnable-token-attack")
    parser.add_argument("--log-images", action="store_true")
    return parser.parse_args()


def images_per_class_label(value: int | None) -> str:
    return "all" if value is None else str(value)


def global_batch_size(args: argparse.Namespace) -> int:
    batch_size = args.global_batch_size if args.global_batch_size is not None else args.attack_batch_size
    if batch_size is None:
        batch_size = 2
    return max(1, int(batch_size))


def nvidia_smi(label: str, *, enabled: bool = True) -> None:
    if not enabled:
        return
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


def terminal_section(title: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}", flush=True)


def terminal_kv(title: str, rows: list[tuple[str, object]]) -> None:
    terminal_section(title)
    width = max(len(key) for key, _ in rows) if rows else 0
    for key, value in rows:
        print(f"{key:<{width}} : {value}", flush=True)


def terminal_stage(index: int, total: int, title: str) -> None:
    print(f"\n[stage {index}/{total}] {title}", flush=True)


def run_name(args: argparse.Namespace, *, train_ipc: int | None, test_ipc: int | None) -> str:
    train_label = images_per_class_label(train_ipc)
    test_label = images_per_class_label(test_ipc)
    return (
        f"uap_fixed10_train{train_label}_test{test_label}_"
        f"gb{global_batch_size(args)}_gbs{args.generator_batch_size}_"
        f"epochs{args.epochs}_nis{args.num_inference_steps}_tokens{args.num_tokens}"
    )


def configured_generator(base: GeneratorConfig, args: argparse.Namespace) -> GeneratorConfig:
    if args.generator == "mock":
        return GeneratorConfig(
            name="mock",
            model_id="mock",
            precision=base.precision,
            batch_size=max(1, args.generator_batch_size),
            seed_per_image=True,
            memory_mode="mock",
            height=min(args.height, 224),
            width=min(args.width, 224),
            guidance_scale=1.0,
            num_inference_steps=1,
            use_cpu_offload=False,
            gradient_checkpointing=False,
        )
    return replace(
        base,
        batch_size=max(1, args.generator_batch_size),
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        use_cpu_offload=False,
    )


def build_stage_config(
    args: argparse.Namespace,
    *,
    split: str,
    images_per_class: int | None,
    output_root: Path,
    name: str,
    steps: int,
):
    base = load_config(args.config)
    wandb = base.logging.wandb
    imagenet_root = args.imagenet_root or base.data.imagenet_root
    return replace(
        base,
        data=DataConfig(
            imagenet_root=imagenet_root,
            split=split,
            class_mode="fixed_10",
            images_per_class=images_per_class,
            clean_correct_only=not args.include_clean_incorrect,
            candidate_multiplier=base.data.candidate_multiplier,
        ),
        generator=configured_generator(base.generator, args),
        attack=replace(
            base.attack,
            training_mode="universal",
            batch_size=global_batch_size(args),
            num_learnable_tokens=args.num_tokens,
            lr=args.lr,
            steps=steps,
            lambda_sem=args.lambda_sem,
            objective=args.objective,
        ),
        output=OutputConfig(root=output_root, save_grids=True, save_images=True),
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
                    "uap",
                    "fixed-10",
                    "universal",
                    "ddp" if int(os.environ.get("WORLD_SIZE", "1")) > 1 else "single-gpu",
                    f"train-{images_per_class_label(images_per_class)}",
                    f"batch-size-{global_batch_size(args)}",
                    f"generator-batch-size-{args.generator_batch_size}",
                    f"epochs-{args.epochs}",
                    f"inference-steps-{args.num_inference_steps}",
                ),
                log_every_steps=wandb.log_every_steps,
                log_images=args.log_images,
            )
        ),
    )


def trim_records(records: list[Any], max_records: int | None) -> list[Any]:
    if max_records is None:
        return records
    return records[:max_records]


def compute_steps(record_count: int, args: argparse.Namespace) -> int:
    if args.steps is not None:
        return max(1, int(args.steps))
    updates_per_epoch = max(1, math.ceil(record_count / global_batch_size(args)))
    return updates_per_epoch * max(1, args.epochs)


def batch_count(record_count: int, batch_size: int) -> int:
    return max(1, math.ceil(record_count / max(1, batch_size)))


def print_run_overview(
    args: argparse.Namespace,
    *,
    name: str,
    output_root: Path,
    metrics_dir: Path,
    train_records: int,
    test_records: int,
    steps: int,
    dist_context: DistributedContext,
) -> None:
    stages = 4 + int(args.save_train_images)
    rows = [
        ("run", name),
        ("output", output_root),
        ("train records", train_records),
        ("test records", test_records),
        ("clean-correct only", not args.include_clean_incorrect),
        ("train updates", steps),
        ("global batch", global_batch_size(args)),
        ("generator batch", args.generator_batch_size),
        ("train batches/epoch", batch_count(train_records, global_batch_size(args))),
        ("test eval batches", batch_count(test_records, args.generator_batch_size)),
        (
            "train eval batches",
            batch_count(train_records, args.generator_batch_size) if args.save_train_images else "skipped",
        ),
        ("world size", dist_context.world_size),
        ("device", dist_context.device),
        ("wandb mode", args.wandb_mode),
        ("stage count", stages),
        ("history csv", metrics_dir / "train_history.csv"),
        ("test csv", metrics_dir / "test_results.csv"),
        ("summary json", metrics_dir / "summary.json"),
    ]
    terminal_kv("Fixed10 UAP Run Overview", rows)
    print(
        "\nProgress lines show step/batch, percent, ASR, elapsed time, and ETA. "
        "In tmux, keep this pane open or tail the log file.",
        flush=True,
    )


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def merge_csv_files(paths: list[Path], output_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if output_path.exists():
        output_path.unlink()
    for path in paths:
        for row in read_csv_rows(path):
            rows.append(row)
            append_csv_row(output_path, row)
    return rows


def stage_summary(rows: list[dict[str, Any]], *, fid: float | None = None) -> dict[str, Any]:
    summary = asdict(summarize_rows(rows, fid=fid))
    clean_rows = [row for row in rows if str(row.get("clean_correct", "")).lower() == "true"]
    summary["clean_correct_count"] = len(clean_rows)
    summary["clean_correct"] = asdict(summarize_rows(clean_rows, fid=None))
    return summary


def remove_previous_outputs(metrics_dir: Path, *, dist_context: DistributedContext) -> None:
    if not dist_context.is_rank0:
        return
    for pattern in (
        "train_history*.csv",
        "train_results*.csv",
        "test_results*.csv",
        "summary.json",
    ):
        for path in metrics_dir.glob(pattern):
            path.unlink()


def main() -> None:
    args = parse_args()
    train_ipc = parse_images_per_class(args.train_images_per_class)
    test_ipc = parse_images_per_class(args.test_images_per_class)
    dist_context = DistributedContext.from_env(args.device)
    name = run_name(args, train_ipc=train_ipc, test_ipc=test_ipc)
    output_root = args.root / name
    metrics_dir = output_root / "metrics"
    stage_total = 4 + int(args.save_train_images)
    set_process_title(f"prompt_attack setup {name}")
    if dist_context.is_rank0:
        ensure_dir(metrics_dir)
        remove_previous_outputs(metrics_dir, dist_context=dist_context)
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_MODE"] = args.wandb_mode
        terminal_kv(
            "Fixed10 UAP Startup",
            [
                ("run", name),
                ("output", output_root),
                ("rank", dist_context.rank),
                ("world size", dist_context.world_size),
                ("device", dist_context.device),
            ],
        )
        nvidia_smi("before uap")
    dist_context.barrier()

    if dist_context.is_rank0:
        terminal_stage(1, stage_total, "load models and select train/test records")
        set_process_title(f"prompt_attack load-records {name}")
    train_probe_config = build_stage_config(
        args,
        split="train",
        images_per_class=train_ipc,
        output_root=output_root,
        name=name,
        steps=1,
    )
    train_probe_runner = LearnableTokenAttackRunner(train_probe_config, device=dist_context.device)
    components = train_probe_runner.build_components()
    train_records = trim_records(
        train_probe_runner.prepare_records(
            components.victim,
            max_records=args.train_max_images,
        ),
        args.train_max_images,
    )
    steps = compute_steps(len(train_records), args)
    train_config = build_stage_config(
        args,
        split="train",
        images_per_class=train_ipc,
        output_root=output_root,
        name=name,
        steps=steps,
    )
    test_config = build_stage_config(
        args,
        split="val",
        images_per_class=test_ipc,
        output_root=output_root,
        name=name,
        steps=steps,
    )
    train_runner = LearnableTokenAttackRunner(train_config, device=dist_context.device)
    test_runner = LearnableTokenAttackRunner(test_config, device=dist_context.device)
    test_records = trim_records(
        test_runner.prepare_records(
            components.victim,
            max_records=args.test_max_images,
        ),
        args.test_max_images,
    )
    if dist_context.is_rank0:
        print_run_overview(
            args,
            name=name,
            output_root=output_root,
            metrics_dir=metrics_dir,
            train_records=len(train_records),
            test_records=len(test_records),
            steps=steps,
            dist_context=dist_context,
        )

    logger = WandbLogger(train_config) if dist_context.is_rank0 else None
    started = time.perf_counter()
    if logger is not None:
        logger.start()
    try:
        if dist_context.is_rank0:
            terminal_stage(2, stage_total, "train shared learnable-token prompt")
            set_process_title(f"prompt_attack train-start {name}")
        prompt_state, history = train_runner.train_universal_prompt(
            train_records,
            components,
            logger=logger,
            history_path=metrics_dir / "train_history.csv" if dist_context.is_rank0 else None,
            dist_context=dist_context,
        )
        train_rows: list[dict[str, Any]] = []
        if args.save_train_images:
            if dist_context.is_rank0:
                terminal_stage(3, stage_total, "frozen train-set image save/eval")
                set_process_title(f"prompt_attack train-eval-start {name}")
            train_rank_path = (
                metrics_dir / f"train_results_rank{dist_context.rank}.csv"
                if dist_context.is_distributed
                else metrics_dir / "train_results.csv"
            )
            train_rows = train_runner.evaluate_universal_prompt(
                dist_context.shard(train_records),
                components,
                prompt_state,
                stage="train",
                metrics_path=train_rank_path,
                logger=logger if not dist_context.is_distributed or dist_context.is_rank0 else None,
                show_progress=not dist_context.is_distributed or dist_context.is_rank0,
            )

        if dist_context.is_rank0:
            test_stage = 4 if args.save_train_images else 3
            terminal_stage(test_stage, stage_total, "frozen val/test eval")
            set_process_title(f"prompt_attack test-eval-start {name}")
        test_rank_path = (
            metrics_dir / f"test_results_rank{dist_context.rank}.csv"
            if dist_context.is_distributed
            else metrics_dir / "test_results.csv"
        )
        test_rows = test_runner.evaluate_universal_prompt(
            dist_context.shard(test_records),
            components,
            prompt_state,
            stage="test",
            metrics_path=test_rank_path,
            logger=logger if not dist_context.is_distributed or dist_context.is_rank0 else None,
            show_progress=not dist_context.is_distributed or dist_context.is_rank0,
        )
        dist_context.barrier()

        if dist_context.is_rank0:
            merge_stage = 5 if args.save_train_images else 4
            terminal_stage(merge_stage, stage_total, "merge metrics and write summary")
            set_process_title(f"prompt_attack summarize {name}")
            if dist_context.is_distributed:
                if args.save_train_images:
                    train_rows = merge_csv_files(
                        [metrics_dir / f"train_results_rank{rank}.csv" for rank in range(dist_context.world_size)],
                        metrics_dir / "train_results.csv",
                    )
                test_rows = merge_csv_files(
                    [metrics_dir / f"test_results_rank{rank}.csv" for rank in range(dist_context.world_size)],
                    metrics_dir / "test_results.csv",
                )
            else:
                train_rows = read_csv_rows(metrics_dir / "train_results.csv") if args.save_train_images else []
                test_rows = read_csv_rows(metrics_dir / "test_results.csv")

            train_fid = compute_fid_for_rows(train_rows, train_config.quality.fid) if train_rows else None
            test_fid = compute_fid_for_rows(test_rows, test_config.quality.fid)
            train_summary = stage_summary(train_rows, fid=train_fid) if train_rows else None
            test_summary = stage_summary(test_rows, fid=test_fid)
            elapsed = time.perf_counter() - started
            summary = {
                "run_name": name,
                "training_mode": "universal",
                "world_size": dist_context.world_size,
                "elapsed_seconds": elapsed,
                "train_record_count": len(train_records),
                "test_record_count": len(test_records),
                "train_updates": len(history),
                "epochs": args.epochs,
                "global_batch_size": global_batch_size(args),
                "generator_batch_size": args.generator_batch_size,
                "save_train_images": args.save_train_images,
                "train": train_summary,
                "test": test_summary,
                "final_train_history": history[-1] if history else None,
            }
            write_json(metrics_dir / "summary.json", summary)
            train_runner.save_universal_prompt(
                prompt_state,
                metadata={
                    "run_name": name,
                    "training_mode": "universal",
                    "world_size": dist_context.world_size,
                    "train_split": "train",
                    "test_split": "val",
                    "train_record_count": len(train_records),
                    "test_record_count": len(test_records),
                    "steps": steps,
                    "epochs": args.epochs,
                    "global_batch_size": global_batch_size(args),
                    "generator_batch_size": args.generator_batch_size,
                    "num_inference_steps": args.num_inference_steps,
                    "num_learnable_tokens": args.num_tokens,
                    "objective": args.objective,
                    "lambda_sem": args.lambda_sem,
                    "lr": args.lr,
                    "output_root": str(output_root),
                },
            )
            if logger is not None:
                logger.log_summary(
                    {
                        "test_asr": test_summary["asr"],
                        "test_clean_correct_asr": test_summary["clean_correct"]["asr"],
                        "test_mean_dino_similarity": test_summary["mean_dino_similarity"],
                        "train_updates": len(history),
                        "elapsed_seconds": elapsed,
                    }
                )
            print(
                "done | "
                f"test_asr={test_summary['asr']:.3f} | "
                f"test_clean_correct_asr={test_summary['clean_correct']['asr']:.3f} | "
                f"elapsed_sec={elapsed:.1f}",
                flush=True,
            )
            set_process_title(f"prompt_attack done {name}")
            if dist_context.is_distributed:
                for rank in range(dist_context.world_size):
                    for prefix in ("train_results", "test_results"):
                        rank_path = metrics_dir / f"{prefix}_rank{rank}.csv"
                        if rank_path.exists():
                            rank_path.unlink()
    finally:
        if logger is not None:
            logger.finish()
        dist_context.barrier()
        if dist_context.is_rank0:
            nvidia_smi("after uap")
        dist_context.close()


if __name__ == "__main__":
    main()
