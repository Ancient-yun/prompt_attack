"""Measure wall-clock time for a tiny real-generator attack run."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from prompt_attack.attacks.runner import SoftTokenAttackRunner
from prompt_attack.config import DataConfig, OutputConfig, load_config
from prompt_attack.utils.io import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--no-cpu-offload", action="store_true")
    parser.add_argument("--clean-correct-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/timing/timing.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_config(args.config)
    config = replace(
        base,
        data=DataConfig(
            imagenet_root=base.data.imagenet_root,
            split=base.data.split,
            class_mode=base.data.class_mode,
            images_per_class=1,
            clean_correct_only=args.clean_correct_only,
            candidate_multiplier=1,
        ),
        generator=replace(
            base.generator,
            height=args.height or base.generator.height,
            width=args.width or base.generator.width,
            num_inference_steps=args.num_inference_steps or base.generator.num_inference_steps,
            use_cpu_offload=False if args.no_cpu_offload else base.generator.use_cpu_offload,
        ),
        attack=replace(base.attack, steps=args.steps),
        output=OutputConfig(root=args.output.parent / "images", save_grids=True, save_images=True),
    )

    timings: dict[str, float | int | str | list[dict[str, float | str | int | bool]]] = {
        "device": args.device,
        "generator": config.generator.name,
        "height": config.generator.height,
        "width": config.generator.width,
        "num_inference_steps": config.generator.num_inference_steps,
        "use_cpu_offload": config.generator.use_cpu_offload,
        "steps": config.attack.steps,
        "max_images": args.max_images,
        "images": [],
    }

    t0 = time.perf_counter()
    runner = SoftTokenAttackRunner(config, device=args.device)
    components = runner.build_components()
    timings["setup_seconds"] = time.perf_counter() - t0

    record_t0 = time.perf_counter()
    records = runner.prepare_records(components.victim, max_records=args.max_images)
    timings["record_selection_seconds"] = time.perf_counter() - record_t0
    if records:
        load_t0 = time.perf_counter()
        prompt_state = components.generator.create_learnable_prompt(
            class_label=records[0].class_label,
            num_tokens=config.attack.num_soft_tokens,
            initializer=config.attack.soft_token_initializer,
            init_std=config.attack.soft_token_init_std,
        )
        timings["learnable_token_count"] = len(prompt_state.token_texts)
        timings["learnable_prompt_text"] = prompt_state.prompt_text
        timings["generator_prompt_setup_seconds"] = time.perf_counter() - load_t0

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    total_attack_seconds = 0.0
    for record in records:
        image_t0 = time.perf_counter()
        row = runner.attack_one(record, components)
        elapsed = time.perf_counter() - image_t0
        total_attack_seconds += elapsed
        images = timings["images"]
        assert isinstance(images, list)
        images.append(
            {
                "image_id": record.image_id,
                "class_label": record.class_label,
                "seconds": elapsed,
                "seconds_per_step": elapsed / config.attack.steps,
                "success": bool(row["success"]),
                "dino_similarity": float(row["dino_similarity"]),
            }
        )

    timings["attack_seconds"] = total_attack_seconds
    timings["seconds_per_image"] = total_attack_seconds / max(len(records), 1)
    timings["seconds_per_step"] = total_attack_seconds / max(len(records) * config.attack.steps, 1)
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        timings["peak_cuda_memory_gb"] = torch.cuda.max_memory_allocated() / (1024**3)

    ensure_dir(args.output.parent)
    write_json(args.output, timings)
    print(json.dumps(timings, indent=2, default=str))


if __name__ == "__main__":
    main()
