"""End-to-end soft-token attack runner."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import asdict
from math import cos, pi
from typing import Any

from tqdm import tqdm

from prompt_attack.attacks.losses import attack_loss_from_objective, dino_loss
from prompt_attack.attacks.soft_tokens import build_prompt, initialize_soft_tokens
from prompt_attack.config import ExperimentConfig
from prompt_attack.data.imagenet import ImageRecord, build_candidate_records, load_image
from prompt_attack.generators.factory import build_generator
from prompt_attack.metrics.fid import compute_fid_for_rows
from prompt_attack.metrics.image_quality import global_ssim, pixel_distance_metrics
from prompt_attack.metrics.nr_iqa import NoReferenceIQAEvaluator
from prompt_attack.metrics.summary import summarize_rows
from prompt_attack.models.semantic import build_semantic_model
from prompt_attack.models.victim import build_victim
from prompt_attack.utils.image import make_side_by_side, pil_to_tensor, save_image, tensor_to_pil
from prompt_attack.utils.io import append_csv_row, ensure_dir, write_json
from prompt_attack.utils.seed import stable_image_seed
from prompt_attack.utils.wandb_logger import WandbLogger


class SoftTokenAttackRunner:
    """Run proposed soft-token attacks sequentially image by image."""

    def __init__(self, config: ExperimentConfig, *, device: str) -> None:
        self.config = config
        self.device = device

    def _build_lr_scheduler(self, optimizer):
        import torch

        scheduler_config = self.config.attack.lr_scheduler
        name = scheduler_config.name.lower()
        if name in {"fixed", "none", "constant"}:
            return None
        if name != "cosine":
            raise ValueError(f"Unsupported lr scheduler: {scheduler_config.name}")

        base_lr = self.config.attack.lr
        if base_lr <= 0:
            raise ValueError("attack.lr must be positive when using an lr scheduler.")
        min_factor = max(0.0, min(1.0, scheduler_config.min_lr / base_lr))
        warmup_steps = scheduler_config.warmup_steps
        total_steps = max(1, self.config.attack.steps)

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return max(min_factor, float(step + 1) / float(warmup_steps))
            decay_steps = max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, float(step - warmup_steps + 1) / float(decay_steps)))
            cosine = 0.5 * (1.0 + cos(pi * progress))
            return min_factor + (1.0 - min_factor) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def _clean_correct_records(
        self,
        records: list[ImageRecord],
        victim,
        *,
        max_records: int | None = None,
    ) -> list[ImageRecord]:
        selected: list[ImageRecord] = []
        per_class: dict[str, int] = defaultdict(int)
        for record in tqdm(records, desc="clean-correct filter"):
            if max_records is not None and len(selected) >= max_records:
                break
            if per_class[record.synset] >= self.config.data.images_per_class:
                continue
            image = load_image(record.path)
            result = victim.evaluate_pil(image, record.class_index)
            if not self.config.data.clean_correct_only or result.pred == record.class_index:
                selected.append(record)
                per_class[record.synset] += 1
        return selected

    def prepare_records(self, victim, *, max_records: int | None = None) -> list[ImageRecord]:
        """Build and optionally clean-correct filter records."""
        candidates = build_candidate_records(self.config.data)
        records = self._clean_correct_records(candidates, victim, max_records=max_records)
        if not records:
            raise RuntimeError("No attack records selected.")
        return records

    def run(self, *, max_images: int | None = None) -> list[dict[str, Any]]:
        """Run the configured attack and persist outputs."""
        ensure_dir(self.config.output.root)
        metrics_path = self.config.output.root / "metrics" / "results.csv"
        summary_path = self.config.output.root / "metrics" / "summary.json"
        for path in (metrics_path, summary_path):
            if path.exists():
                path.unlink()

        logger = WandbLogger(self.config)
        logger.start()
        try:
            victim = build_victim(
                self.config.victim.name,
                weights=self.config.victim.weights,
                device=self.device,
            )
            semantic = build_semantic_model(self.config.semantic.name, device=self.device)
            generator = build_generator(self.config.generator, device=self.device)
            quality_evaluator = NoReferenceIQAEvaluator(
                self.config.quality.nriqa,
                device=self.device,
            )
            if not generator.supports_gradient:
                raise RuntimeError(
                    f"Generator '{self.config.generator.name}' is inference-only in this scaffold. "
                    "Run scripts/smoke_test.py for a differentiable mock attack path, then implement "
                    "the FLUX.2 soft-token conditioning hook before the full run."
                )

            records = self.prepare_records(victim, max_records=max_images)
            if max_images is not None:
                records = records[:max_images]
            rows: list[dict[str, Any]] = []
            for image_index, record in enumerate(tqdm(records, desc="attack")):
                row = self._attack_one(
                    record,
                    victim,
                    semantic,
                    generator,
                    quality_evaluator=quality_evaluator,
                    logger=logger,
                    image_index=image_index,
                )
                append_csv_row(metrics_path, row)
                rows.append(row)

            fid_value = compute_fid_for_rows(rows, self.config.quality.fid)
            summary = asdict(summarize_rows(rows, fid=fid_value))
            write_json(summary_path, summary)
            logger.log_summary(summary)
            return rows
        finally:
            logger.finish()

    def _attack_one(
        self,
        record: ImageRecord,
        victim,
        semantic,
        generator,
        *,
        quality_evaluator: NoReferenceIQAEvaluator | None = None,
        logger: WandbLogger | None = None,
        image_index: int = 0,
    ) -> dict[str, Any]:
        import torch

        started_at = time.perf_counter()
        image = load_image(record.path)
        original_tensor = pil_to_tensor(image, device=self.device)
        clean_logits = victim.logits_from_tensor(original_tensor)
        clean_eval = victim.evaluate_logits(clean_logits, record.class_index)
        prompt = build_prompt(record.class_label, self.config.attack.num_soft_tokens)
        expected_dim = generator.soft_token_dim(prompt)
        token_dim = self.config.attack.soft_token_dim or expected_dim
        if token_dim != expected_dim:
            raise ValueError(
                f"Configured soft_token_dim={token_dim} does not match generator "
                f"conditioning dim={expected_dim} for '{prompt}'."
            )
        soft_tokens = initialize_soft_tokens(
            self.config.attack.num_soft_tokens,
            token_dim,
            device=self.device,
        )
        if not soft_tokens.requires_grad:
            raise RuntimeError("Soft-token parameter must require gradients.")
        optimizer = torch.optim.Adam([soft_tokens], lr=self.config.attack.lr)
        lr_scheduler = self._build_lr_scheduler(optimizer)
        seed = stable_image_seed(0, record.image_id)
        best: dict[str, Any] | None = None
        best_attack: dict[str, Any] | None = None
        best_semantic_success: dict[str, Any] | None = None
        min_true_conf: dict[str, Any] | None = None
        first_success_step = -1
        first_semantic_success_step = -1

        for step in range(self.config.attack.steps):
            optimizer.zero_grad(set_to_none=True)
            current_lr = float(optimizer.param_groups[0]["lr"])
            generated = generator.generate(
                input_image=image,
                input_tensor=original_tensor,
                prompt=prompt,
                soft_tokens=soft_tokens,
                seed=seed,
                require_grad=True,
            )
            logits = victim.logits_from_tensor(generated.image_tensor)
            attack_loss = attack_loss_from_objective(
                logits,
                record.class_index,
                self.config.attack.objective,
            )
            dino_sim = None
            sem_loss = None
            if self.config.attack.lambda_sem > 0:
                dino_sim = semantic.similarity(original_tensor, generated.image_tensor)
                sem_loss = dino_loss(dino_sim)
                total_loss = attack_loss + self.config.attack.lambda_sem * sem_loss
            else:
                total_loss = attack_loss
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"Non-finite loss for {record.image_id} at step {step}")
            total_loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            eval_result = victim.evaluate_logits(logits.detach(), record.class_index)
            success = eval_result.pred != record.class_index
            if dino_sim is None or sem_loss is None:
                with torch.no_grad():
                    dino_sim = semantic.similarity(original_tensor, generated.image_tensor.detach())
                    sem_loss = dino_loss(dino_sim)
            semantic_constrained_success = (
                success and float(dino_sim.detach().cpu().item()) >= self.config.attack.semantic_threshold
            )
            current = {
                "step": step,
                "image_tensor": generated.image_tensor.detach(),
                "attack_loss": float(attack_loss.detach().cpu().item()),
                "dino_loss": float(sem_loss.detach().cpu().item()),
                "total_loss": float(total_loss.detach().cpu().item()),
                "adv_pred": eval_result.pred,
                "adv_top1_conf": eval_result.pred_conf,
                "adv_true_conf": eval_result.true_conf,
                "adv_margin": eval_result.margin,
                "dino_similarity": float(dino_sim.detach().cpu().item()),
                "success": success,
                "semantic_constrained_success": semantic_constrained_success,
            }
            if success and first_success_step < 0:
                first_success_step = step
            if semantic_constrained_success and first_semantic_success_step < 0:
                first_semantic_success_step = step
            if logger is not None:
                logger.log_step(
                    image_index=image_index,
                    image_id=record.image_id,
                    class_label=record.class_label,
                    attack_step=step,
                    values={
                        "attack_loss": current["attack_loss"],
                        "attack_objective_loss": current["attack_loss"],
                        "semantic_loss": current["dino_loss"],
                        "total_loss": current["total_loss"],
                        "lr": current_lr,
                        "true_conf": current["adv_true_conf"],
                        "confidence_drop": clean_eval.true_conf - current["adv_true_conf"],
                        "top1_conf": current["adv_top1_conf"],
                        "logit_gap_true_vs_other": current["adv_margin"],
                        "dino_similarity": current["dino_similarity"],
                        "success": int(success),
                        "semantic_constrained_success": int(semantic_constrained_success),
                    },
                )
            if best is None or current["total_loss"] < best["total_loss"]:
                best = current
            if best_attack is None or current["adv_margin"] < best_attack["adv_margin"]:
                best_attack = current
            if min_true_conf is None or current["adv_true_conf"] < min_true_conf["adv_true_conf"]:
                min_true_conf = current
            if semantic_constrained_success and (
                best_semantic_success is None
                or current["adv_margin"] < best_semantic_success["adv_margin"]
            ):
                best_semantic_success = current

        if best is None or best_attack is None or min_true_conf is None:
            raise RuntimeError(f"No optimization step ran for {record.image_id}")

        adv_image = tensor_to_pil(best["image_tensor"])
        output_dir = self.config.output.root / "images" / record.class_label / record.image_id
        if self.config.output.save_images:
            save_image(image, output_dir / "original.png")
            save_image(adv_image, output_dir / "adv.png")
        if self.config.output.save_grids:
            grid = make_side_by_side(image, adv_image, "original", "adv")
            save_image(grid, self.config.output.root / "grids" / f"{record.class_label}_{record.image_id}.png")

        success = int(best["adv_pred"]) != record.class_index
        confidence_drop = clean_eval.true_conf - float(best["adv_true_conf"])
        semantic_constrained_success = (
            success and float(best["dino_similarity"]) >= self.config.attack.semantic_threshold
        )
        pixel_metrics = pixel_distance_metrics(original_tensor, best["image_tensor"])
        ssim = global_ssim(original_tensor, best["image_tensor"])
        nriqa_metrics = (
            quality_evaluator.score_tensor(best["image_tensor"])
            if quality_evaluator is not None
            else {}
        )
        runtime_seconds = time.perf_counter() - started_at
        row = {
            "image_id": record.image_id,
            "class_id": record.class_index,
            "class_label": record.class_label,
            "run_name": self.config.generator.name,
            "seed": seed,
            "prompt_text": prompt,
            "num_soft_tokens": self.config.attack.num_soft_tokens,
            "lr": self.config.attack.lr,
            "lr_scheduler": self.config.attack.lr_scheduler.name,
            "lr_warmup_steps": self.config.attack.lr_scheduler.warmup_steps,
            "lr_min": self.config.attack.lr_scheduler.min_lr,
            "steps": self.config.attack.steps,
            "lambda_sem": self.config.attack.lambda_sem,
            "semantic_threshold": self.config.attack.semantic_threshold,
            "objective": self.config.attack.objective,
            "generator_height": self.config.generator.height,
            "generator_width": self.config.generator.width,
            "num_inference_steps": self.config.generator.num_inference_steps,
            "clean_pred": clean_eval.pred,
            "clean_pred_label": victim.categories[clean_eval.pred],
            "clean_top1_conf": clean_eval.pred_conf,
            "adv_pred": int(best["adv_pred"]),
            "adv_pred_label": victim.categories[int(best["adv_pred"])],
            "adv_top1_conf": float(best["adv_top1_conf"]),
            "success": success,
            "semantic_constrained_success": semantic_constrained_success,
            "clean_true_conf": clean_eval.true_conf,
            "adv_true_conf": float(best["adv_true_conf"]),
            "confidence_drop": confidence_drop,
            "clean_margin": clean_eval.margin,
            "adv_margin": float(best["adv_margin"]),
            "margin_drop": clean_eval.margin - float(best["adv_margin"]),
            "dino_similarity": float(best["dino_similarity"]),
            "ssim": ssim,
            **pixel_metrics,
            **nriqa_metrics,
            "best_step": int(best["step"]),
            "best_attack_step": int(best_attack["step"]),
            "best_semantic_success_step": (
                -1 if best_semantic_success is None else int(best_semantic_success["step"])
            ),
            "first_success_step": first_success_step,
            "first_semantic_success_step": first_semantic_success_step,
            "min_adv_true_conf": float(min_true_conf["adv_true_conf"]),
            "min_adv_true_conf_step": int(min_true_conf["step"]),
            "min_adv_margin": float(best_attack["adv_margin"]),
            "min_adv_margin_step": int(best_attack["step"]),
            "best_attack_loss": float(best["attack_loss"]),
            "best_dino_loss": float(best["dino_loss"]),
            "best_total_loss": float(best["total_loss"]),
            "runtime_seconds": runtime_seconds,
            "output_image_path": str(output_dir / "adv.png"),
            "original_image_path": str(output_dir / "original.png"),
            "grid_image_path": str(
                self.config.output.root / "grids" / f"{record.class_label}_{record.image_id}.png"
            ),
        }
        if logger is not None:
            logger.log_image_result(row=row, original=image, adversarial=adv_image)
        return row
