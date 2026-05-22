"""End-to-end textual-inversion token attack runner."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from math import cos, pi
from pathlib import Path
from typing import Any

from tqdm import tqdm

from prompt_attack.attacks.learnable_tokens import build_prompt
from prompt_attack.attacks.losses import (
    attack_semantic_loss_weights,
    attack_loss_from_objective,
    dino_loss,
    weighted_attack_semantic_loss,
)
from prompt_attack.config import ExperimentConfig
from prompt_attack.data.imagenet import ImageRecord, build_candidate_records, load_image
from prompt_attack.generators.base import LearnablePrompt, LearnablePromptBatch
from prompt_attack.generators.factory import build_generator
from prompt_attack.metrics.fid import compute_fid_for_rows
from prompt_attack.metrics.image_quality import global_ssim, pixel_distance_metrics
from prompt_attack.metrics.nr_iqa import NoReferenceIQAEvaluator
from prompt_attack.metrics.summary import summarize_rows
from prompt_attack.models.semantic import build_semantic_model
from prompt_attack.models.victim import build_victim
from prompt_attack.utils.distributed import DistributedContext
from prompt_attack.utils.image import make_side_by_side, pil_to_tensor, save_image, tensor_to_pil
from prompt_attack.utils.io import append_csv_row, ensure_dir, write_json
from prompt_attack.utils.process_title import set_process_title
from prompt_attack.utils.seed import stable_image_seed
from prompt_attack.utils.wandb_logger import WandbLogger


CLEAN_FILTER_BATCH_SIZE = 512


def _format_duration(seconds: float) -> str:
    """Return a compact human-readable duration."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _progress_eta(*, started_at: float, completed: int, total: int) -> tuple[float, float]:
    """Return elapsed seconds and ETA seconds for a progress counter."""
    elapsed = time.perf_counter() - started_at
    if completed <= 0 or total <= 0:
        return elapsed, 0.0
    remaining = max(total - completed, 0)
    eta = elapsed / completed * remaining
    return elapsed, eta


@dataclass(frozen=True)
class AttackComponents:
    """Loaded model components shared across attack entry points."""

    victim: Any
    semantic: Any
    generator: Any
    quality_evaluator: Any


class LearnableTokenAttackRunner:
    """Run textual-inversion token attacks."""

    def __init__(self, config: ExperimentConfig, *, device: str) -> None:
        self.config = config
        self.device = device

    def build_components(self) -> AttackComponents:
        """Load reusable attack models and evaluators."""
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
                f"Generator '{self.config.generator.name}' does not support differentiable "
                "generation required for learnable-token optimization."
            )
        return AttackComponents(
            victim=victim,
            semantic=semantic,
            generator=generator,
            quality_evaluator=quality_evaluator,
        )

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
        cap = self.config.data.images_per_class

        def can_select(record: ImageRecord) -> bool:
            return cap is None or per_class[record.synset] < cap

        if not self.config.data.clean_correct_only:
            for record in tqdm(records, desc="clean-correct filter"):
                if max_records is not None and len(selected) >= max_records:
                    break
                if not can_select(record):
                    continue
                selected.append(record)
                per_class[record.synset] += 1
            return selected

        filter_batch_size = CLEAN_FILTER_BATCH_SIZE
        pending: list[ImageRecord] = []
        progress = tqdm(total=len(records), desc="clean-correct filter")

        def flush_pending() -> None:
            nonlocal pending
            if not pending:
                return
            batch = pending
            pending = []
            images = [load_image(record.path) for record in batch]
            labels = [record.class_index for record in batch]
            if hasattr(victim, "evaluate_pil_batch"):
                results = victim.evaluate_pil_batch(images, labels)
            else:
                results = [
                    victim.evaluate_pil(image, label)
                    for image, label in zip(images, labels)
                ]
            for record, result in zip(batch, results):
                if max_records is not None and len(selected) >= max_records:
                    break
                if not can_select(record):
                    continue
                if result.pred == record.class_index:
                    selected.append(record)
                    per_class[record.synset] += 1
            progress.update(len(batch))

        try:
            for record in records:
                if max_records is not None and len(selected) >= max_records:
                    break
                if not can_select(record):
                    progress.update(1)
                    continue
                pending.append(record)
                if len(pending) >= filter_batch_size:
                    flush_pending()
            flush_pending()
        finally:
            progress.close()
        return selected

    def prepare_records(self, victim, *, max_records: int | None = None) -> list[ImageRecord]:
        """Build and optionally clean-correct filter records."""
        candidates = build_candidate_records(self.config.data)
        records = self._clean_correct_records(candidates, victim, max_records=max_records)
        if not records:
            raise RuntimeError("No attack records selected.")
        return records

    def _record_batches(
        self,
        records: list[ImageRecord],
        *,
        batch_size: int,
    ) -> list[list[ImageRecord]]:
        """Split records into non-empty batches."""
        batch_size = max(1, batch_size)
        return [records[index : index + batch_size] for index in range(0, len(records), batch_size)]

    def _shared_prompt_batch(
        self,
        prompt_state: LearnablePrompt,
        records: list[ImageRecord],
    ) -> LearnablePromptBatch:
        """Build per-class prompt texts backed by one shared learnable token tensor."""
        return LearnablePromptBatch(
            prompt_texts=tuple(
                build_prompt(record.class_label, len(prompt_state.token_texts)) for record in records
            ),
            token_texts=prompt_state.token_texts,
            token_ids=prompt_state.token_ids,
            learnable_embeddings=prompt_state.learnable_embeddings,
        )

    def _load_batch_inputs(self, records: list[ImageRecord]):
        """Load PIL images and matching resized tensors for one attack batch."""
        import torch

        images = [load_image(record.path) for record in records]
        reference_images = [
            image.resize((self.config.generator.width, self.config.generator.height))
            for image in images
        ]
        original_tensor = torch.cat(
            [pil_to_tensor(image, device=self.device) for image in reference_images],
            dim=0,
        )
        true_labels = torch.tensor(
            [record.class_index for record in records],
            device=self.device,
            dtype=torch.long,
        )
        return images, original_tensor, true_labels

    def run(self, *, max_images: int | None = None) -> list[dict[str, Any]]:
        """Run the configured attack and persist outputs."""
        if self.config.attack.training_mode == "universal":
            return self.run_universal(max_images=max_images)

        ensure_dir(self.config.output.root)
        metrics_path = self.config.output.root / "metrics" / "results.csv"
        summary_path = self.config.output.root / "metrics" / "summary.json"
        for path in (metrics_path, summary_path):
            if path.exists():
                path.unlink()

        logger = WandbLogger(self.config)
        logger.start()
        try:
            components = self.build_components()
            records = self.prepare_records(components.victim, max_records=max_images)
            if max_images is not None:
                records = records[:max_images]
            rows: list[dict[str, Any]] = []
            batch_size = max(1, self.config.generator.batch_size)
            if batch_size == 1:
                for image_index, record in enumerate(tqdm(records, desc="attack")):
                    row = self.attack_one(
                        record,
                        components,
                        logger=logger,
                        image_index=image_index,
                    )
                    append_csv_row(metrics_path, row)
                    rows.append(row)
            else:
                batches = self._record_batches(records, batch_size=batch_size)
                for batch_index, batch in enumerate(tqdm(batches, desc="attack-batches")):
                    batch_rows = self.attack_batch(
                        batch,
                        components,
                        logger=logger,
                        image_start_index=batch_index * batch_size,
                    )
                    for row in batch_rows:
                        append_csv_row(metrics_path, row)
                        rows.append(row)

            fid_value = compute_fid_for_rows(rows, self.config.quality.fid)
            summary = asdict(summarize_rows(rows, fid=fid_value))
            write_json(summary_path, summary)
            logger.log_summary(summary)
            return rows
        finally:
            logger.finish()

    def create_universal_prompt(self, components: AttackComponents) -> LearnablePrompt:
        """Create the single shared prompt parameter used by universal attacks."""
        return components.generator.create_learnable_prompt(
            class_label="object",
            num_tokens=self.config.attack.num_learnable_tokens,
            initializer=self.config.attack.learnable_token_initializer,
            init_std=self.config.attack.learnable_token_init_std,
        )

    def train_universal_prompt(
        self,
        records: list[ImageRecord],
        components: AttackComponents,
        *,
        logger: WandbLogger | None = None,
        history_path: Path | None = None,
        dist_context: DistributedContext | None = None,
    ) -> tuple[LearnablePrompt, list[dict[str, Any]]]:
        """Optimize one shared learnable-token tensor over cyclic train batches."""
        import torch

        dist_context = dist_context or DistributedContext(
            rank=0,
            local_rank=0,
            world_size=1,
            device=self.device,
            backend="none",
        )
        if not records:
            raise ValueError("Universal training requires at least one record.")
        attack_loss_weight, semantic_loss_weight = attack_semantic_loss_weights(
            self.config.attack.objective,
            self.config.attack.lambda_sem,
        )
        prompt_state = self.create_universal_prompt(components)
        learnable_embeddings = prompt_state.learnable_embeddings
        if not isinstance(learnable_embeddings, torch.Tensor):
            raise TypeError("Universal prompt state must expose torch.Tensor learnable embeddings.")
        if learnable_embeddings.ndim != 2:
            raise ValueError("Universal prompt embeddings must have shape [N, D].")
        if not learnable_embeddings.requires_grad:
            raise RuntimeError("Universal prompt embeddings must require gradients.")
        dist_context.broadcast_tensor(learnable_embeddings.data)
        components.generator.sync_learnable_prompt(prompt_state)

        optimizer = torch.optim.Adam([learnable_embeddings], lr=self.config.attack.lr)
        lr_scheduler = self._build_lr_scheduler(optimizer)
        effective_batch_size = max(1, self.config.attack.batch_size)
        micro_batch_size = max(1, self.config.generator.batch_size)
        batches = self._record_batches(records, batch_size=effective_batch_size)
        history: list[dict[str, Any]] = []
        train_started_at = time.perf_counter()

        progress = tqdm(
            range(self.config.attack.steps),
            desc="uap-train",
            disable=not dist_context.is_rank0,
            dynamic_ncols=True,
        )
        for step in progress:
            batch = batches[step % len(batches)]
            local_batch = dist_context.shard(batch)
            optimizer.zero_grad(set_to_none=True)
            current_lr = float(optimizer.param_groups[0]["lr"])
            step_attack_loss = 0.0
            step_dino_loss = 0.0
            step_total_loss = 0.0
            step_success = 0
            processed = 0

            for micro_batch in self._record_batches(local_batch, batch_size=micro_batch_size):
                images, original_tensor, true_labels = self._load_batch_inputs(micro_batch)
                prompt_batch = self._shared_prompt_batch(prompt_state, micro_batch)
                seeds = [stable_image_seed(0, record.image_id) for record in micro_batch]
                generated = components.generator.generate_batch(
                    input_images=images,
                    input_tensor=original_tensor,
                    prompt_state=prompt_batch,
                    seeds=seeds,
                    require_grad=True,
                )
                image_tensor = generated.image_tensor
                if not isinstance(image_tensor, torch.Tensor):
                    raise TypeError("Generator batch result must expose torch.Tensor image_tensor.")
                logits = components.victim.logits_from_tensor(image_tensor)
                attack_losses = attack_loss_from_objective(
                    logits,
                    true_labels,
                    self.config.attack.objective,
                    reduction="none",
                )
                if semantic_loss_weight > 0:
                    dino_sim = components.semantic.similarity(original_tensor, image_tensor)
                    sem_losses = 1.0 - dino_sim
                    total_losses = attack_loss_weight * attack_losses + semantic_loss_weight * sem_losses
                else:
                    sem_losses = torch.zeros_like(attack_losses)
                    total_losses = attack_losses

                total_loss = total_losses.sum() / len(batch)
                if not torch.isfinite(total_loss):
                    raise FloatingPointError(f"Non-finite universal loss at step {step}")
                total_loss.backward()

                with torch.no_grad():
                    eval_results = components.victim.evaluate_logits_batch(
                        logits.detach(),
                        true_labels.detach().cpu().tolist(),
                    )
                step_attack_loss += float(attack_losses.detach().sum().cpu().item())
                step_dino_loss += float(sem_losses.detach().sum().cpu().item())
                step_total_loss += float(total_losses.detach().sum().cpu().item())
                step_success += sum(
                    int(result.pred != record.class_index)
                    for result, record in zip(eval_results, micro_batch)
                )
                processed += len(micro_batch)

            if learnable_embeddings.grad is None:
                learnable_embeddings.grad = torch.zeros_like(learnable_embeddings)
            dist_context.all_reduce_sum(learnable_embeddings.grad)
            stats = torch.tensor(
                [step_attack_loss, step_dino_loss, step_total_loss, step_success, processed],
                device=learnable_embeddings.device,
                dtype=torch.float64,
            )
            dist_context.all_reduce_sum(stats)
            global_processed = max(1.0, float(stats[4].detach().cpu().item()))
            optimizer.step()
            components.generator.sync_learnable_prompt(prompt_state)
            if lr_scheduler is not None:
                lr_scheduler.step()

            history_row = {
                "step": step,
                "batch_index": step % len(batches),
                "lr": current_lr,
                "attack_loss": float(stats[0].detach().cpu().item()) / global_processed,
                "dino_loss": float(stats[1].detach().cpu().item()) / global_processed,
                "total_loss": float(stats[2].detach().cpu().item()) / global_processed,
                "success_rate": float(stats[3].detach().cpu().item()) / global_processed,
                "effective_batch_size": len(batch),
                "micro_batch_size": micro_batch_size,
                "world_size": dist_context.world_size,
            }
            completed_steps = step + 1
            elapsed, eta = _progress_eta(
                started_at=train_started_at,
                completed=completed_steps,
                total=self.config.attack.steps,
            )
            if dist_context.is_rank0:
                history.append(history_row)
                set_process_title(
                    f"prompt_attack uap-train {self.config.output.root.name} "
                    f"{completed_steps}/{self.config.attack.steps} "
                    f"asr={history_row['success_rate']:.3f} eta={_format_duration(eta)}"
                )
                progress.set_postfix(
                    {
                        "loss": f"{history_row['total_loss']:.4f}",
                        "asr": f"{history_row['success_rate']:.3f}",
                        "lr": f"{current_lr:.2e}",
                        "eta": _format_duration(eta),
                    },
                    refresh=True,
                )
                progress.write(
                    "[train] "
                    f"step {completed_steps}/{self.config.attack.steps} "
                    f"({completed_steps / self.config.attack.steps:.1%}) | "
                    f"batch={history_row['batch_index'] + 1}/{len(batches)} | "
                    f"loss={history_row['total_loss']:.4f} | "
                    f"attack_loss={history_row['attack_loss']:.4f} | "
                    f"asr={history_row['success_rate']:.3f} | "
                    f"lr={current_lr:.2e} | "
                    f"elapsed={_format_duration(elapsed)} | "
                    f"eta={_format_duration(eta)}",
                )
            if history_path is not None and dist_context.is_rank0:
                append_csv_row(history_path, history_row)
            if logger is not None and dist_context.is_rank0:
                logger.log_universal_step(
                    attack_step=step,
                    values={
                        "attack_loss": history_row["attack_loss"],
                        "semantic_loss": history_row["dino_loss"],
                        "total_loss": history_row["total_loss"],
                        "lr": current_lr,
                        "success_rate": history_row["success_rate"],
                        "effective_batch_size": len(batch),
                        "micro_batch_size": micro_batch_size,
                    },
                )

        return prompt_state, history

    def evaluate_universal_prompt(
        self,
        records: list[ImageRecord],
        components: AttackComponents,
        prompt_state: LearnablePrompt,
        *,
        stage: str,
        metrics_path: Path | None = None,
        logger: WandbLogger | None = None,
        show_progress: bool = True,
    ) -> list[dict[str, Any]]:
        """Evaluate a frozen universal prompt over records without optimizer updates."""
        import torch

        rows: list[dict[str, Any]] = []
        attack_loss_weight, semantic_loss_weight = attack_semantic_loss_weights(
            self.config.attack.objective,
            self.config.attack.lambda_sem,
        )
        batch_size = max(1, self.config.generator.batch_size)
        batches = self._record_batches(records, batch_size=batch_size)
        eval_started_at = time.perf_counter()
        seen = 0
        clean_correct_count = 0
        success_count = 0
        clean_success_count = 0
        semantic_success_count = 0
        progress = tqdm(
            batches,
            desc=f"{stage}-eval",
            disable=not show_progress,
            dynamic_ncols=True,
        )
        for batch_index, batch in enumerate(progress):
            started_at = time.perf_counter()
            images, original_tensor, true_labels = self._load_batch_inputs(batch)
            prompt_batch = self._shared_prompt_batch(prompt_state, batch)
            seeds = [stable_image_seed(0, record.image_id) for record in batch]
            generated = components.generator.generate_batch(
                input_images=images,
                input_tensor=original_tensor,
                prompt_state=prompt_batch,
                seeds=seeds,
                require_grad=False,
            )
            image_tensor = generated.image_tensor
            if not isinstance(image_tensor, torch.Tensor):
                raise TypeError("Generator batch result must expose torch.Tensor image_tensor.")
            with torch.no_grad():
                clean_logits = components.victim.logits_from_tensor(original_tensor)
                adv_logits = components.victim.logits_from_tensor(image_tensor)
                clean_evals = components.victim.evaluate_logits_batch(
                    clean_logits,
                    true_labels.detach().cpu().tolist(),
                )
                adv_evals = components.victim.evaluate_logits_batch(
                    adv_logits,
                    true_labels.detach().cpu().tolist(),
                )
                attack_losses = attack_loss_from_objective(
                    adv_logits,
                    true_labels,
                    self.config.attack.objective,
                    reduction="none",
                )
                dino_sim = components.semantic.similarity(original_tensor, image_tensor)
                dino_losses = 1.0 - dino_sim
                total_losses = (
                    attack_loss_weight * attack_losses + semantic_loss_weight * dino_losses
                    if semantic_loss_weight > 0
                    else attack_losses
                )

            runtime_seconds = time.perf_counter() - started_at
            for index, record in enumerate(batch):
                clean_eval = clean_evals[index]
                adv_eval = adv_evals[index]
                adv_image = tensor_to_pil(image_tensor[index : index + 1])
                output_dir = self.config.output.root / "images" / stage / record.class_label / record.image_id
                if self.config.output.save_images:
                    save_image(images[index], output_dir / "original.png")
                    save_image(adv_image, output_dir / "adv.png")
                grid_path = (
                    self.config.output.root
                    / "grids"
                    / f"{stage}_{record.class_label}_{record.image_id}.png"
                )
                if self.config.output.save_grids:
                    grid = make_side_by_side(images[index], adv_image, "original", "adv")
                    save_image(grid, grid_path)

                original_one = original_tensor[index : index + 1]
                adv_one = image_tensor[index : index + 1]
                success = adv_eval.pred != record.class_index
                clean_correct = clean_eval.pred == record.class_index
                dino_value = float(dino_sim[index].detach().cpu().item())
                semantic_constrained_success = success and dino_value >= self.config.attack.semantic_threshold
                pixel_metrics = pixel_distance_metrics(original_one, adv_one)
                ssim = global_ssim(original_one, adv_one)
                nriqa_metrics = (
                    components.quality_evaluator.score_tensor(adv_one)
                    if components.quality_evaluator is not None
                    else {}
                )
                final_step = self.config.attack.steps
                row = {
                    "stage": stage,
                    "training_mode": "universal",
                    "image_id": record.image_id,
                    "class_id": record.class_index,
                    "class_label": record.class_label,
                    "run_name": self.config.generator.name,
                    "seed": seeds[index],
                    "prompt_text": prompt_batch.prompt_texts[index],
                    "num_learnable_tokens": self.config.attack.num_learnable_tokens,
                    "learnable_token_initializer": self.config.attack.learnable_token_initializer,
                    "learnable_token_init_std": self.config.attack.learnable_token_init_std,
                    "learnable_token_texts": " ".join(prompt_state.token_texts),
                    "lr": self.config.attack.lr,
                    "lr_scheduler": self.config.attack.lr_scheduler.name,
                    "lr_warmup_steps": self.config.attack.lr_scheduler.warmup_steps,
                    "lr_min": self.config.attack.lr_scheduler.min_lr,
                    "steps": self.config.attack.steps,
                    "lambda_sem": self.config.attack.lambda_sem,
                    "attack_loss_weight": attack_loss_weight,
                    "semantic_loss_weight": semantic_loss_weight,
                    "semantic_threshold": self.config.attack.semantic_threshold,
                    "objective": self.config.attack.objective,
                    "attack_batch_size": self.config.attack.batch_size,
                    "generator_height": self.config.generator.height,
                    "generator_width": self.config.generator.width,
                    "generator_batch_size": self.config.generator.batch_size,
                    "num_inference_steps": self.config.generator.num_inference_steps,
                    "clean_pred": clean_eval.pred,
                    "clean_pred_label": components.victim.categories[clean_eval.pred],
                    "clean_top1_conf": clean_eval.pred_conf,
                    "clean_correct": clean_correct,
                    "adv_pred": adv_eval.pred,
                    "adv_pred_label": components.victim.categories[adv_eval.pred],
                    "adv_top1_conf": adv_eval.pred_conf,
                    "success": success,
                    "semantic_constrained_success": semantic_constrained_success,
                    "clean_true_conf": clean_eval.true_conf,
                    "adv_true_conf": adv_eval.true_conf,
                    "confidence_drop": clean_eval.true_conf - adv_eval.true_conf,
                    "clean_margin": clean_eval.margin,
                    "adv_margin": adv_eval.margin,
                    "margin_drop": clean_eval.margin - adv_eval.margin,
                    "dino_similarity": dino_value,
                    "ssim": ssim,
                    **pixel_metrics,
                    **nriqa_metrics,
                    "best_step": final_step,
                    "best_attack_step": final_step,
                    "best_semantic_success_step": final_step if semantic_constrained_success else -1,
                    "first_success_step": final_step if success else -1,
                    "first_semantic_success_step": final_step if semantic_constrained_success else -1,
                    "min_adv_true_conf": adv_eval.true_conf,
                    "min_adv_true_conf_step": final_step,
                    "min_adv_margin": adv_eval.margin,
                    "min_adv_margin_step": final_step,
                    "best_attack_loss": float(attack_losses[index].detach().cpu().item()),
                    "best_dino_loss": float(dino_losses[index].detach().cpu().item()),
                    "best_total_loss": float(total_losses[index].detach().cpu().item()),
                    "runtime_seconds": runtime_seconds,
                    "output_image_path": str(output_dir / "adv.png"),
                    "original_image_path": str(output_dir / "original.png"),
                    "grid_image_path": str(grid_path),
                }
                if metrics_path is not None:
                    append_csv_row(metrics_path, row)
                if logger is not None:
                    logger.log_image_result(row=row, original=images[index], adversarial=adv_image)
                rows.append(row)
                seen += 1
                clean_correct_count += int(clean_correct)
                success_count += int(success)
                clean_success_count += int(clean_correct and success)
                semantic_success_count += int(semantic_constrained_success)
            if show_progress:
                elapsed, eta = _progress_eta(
                    started_at=eval_started_at,
                    completed=batch_index + 1,
                    total=len(batches),
                )
                asr = success_count / max(seen, 1)
                clean_asr = clean_success_count / max(clean_correct_count, 1)
                set_process_title(
                    f"prompt_attack {stage}-eval {self.config.output.root.name} "
                    f"{seen}/{len(records)} asr={asr:.3f} eta={_format_duration(eta)}"
                )
                progress.set_postfix(
                    {
                        "images": f"{seen}/{len(records)}",
                        "asr": f"{asr:.3f}",
                        "clean_asr": f"{clean_asr:.3f}",
                        "eta": _format_duration(eta),
                    },
                    refresh=True,
                )
                progress.write(
                    f"[{stage}] "
                    f"batch {batch_index + 1}/{len(batches)} | "
                    f"images={seen}/{len(records)} ({seen / max(len(records), 1):.1%}) | "
                    f"clean_correct={clean_correct_count} | "
                    f"success={success_count} | "
                    f"semantic_success={semantic_success_count} | "
                    f"asr={asr:.3f} | "
                    f"clean_asr={clean_asr:.3f} | "
                    f"elapsed={_format_duration(elapsed)} | "
                    f"eta={_format_duration(eta)}",
                )
        return rows

    def save_universal_prompt(
        self,
        prompt_state: LearnablePrompt,
        *,
        metadata: dict[str, Any],
    ) -> None:
        """Persist learned universal prompt weights and metadata."""
        import torch

        prompt_dir = self.config.output.root / "prompt"
        ensure_dir(prompt_dir)
        learnable_embeddings = prompt_state.learnable_embeddings
        if not isinstance(learnable_embeddings, torch.Tensor):
            raise TypeError("Universal prompt state must expose torch.Tensor learnable embeddings.")
        torch.save(
            {
                "token_texts": prompt_state.token_texts,
                "token_ids": prompt_state.token_ids,
                "learnable_embeddings": learnable_embeddings.detach().cpu(),
                "metadata": metadata,
            },
            prompt_dir / "learned_prompt.pt",
        )
        write_json(
            prompt_dir / "metadata.json",
            {
                **metadata,
                "token_texts": list(prompt_state.token_texts),
                "token_ids": list(prompt_state.token_ids),
                "embedding_shape": list(learnable_embeddings.shape),
            },
        )

    def run_universal(self, *, max_images: int | None = None) -> list[dict[str, Any]]:
        """Train one universal prompt and evaluate it on the same selected split."""
        ensure_dir(self.config.output.root)
        metrics_dir = self.config.output.root / "metrics"
        metrics_path = metrics_dir / "results.csv"
        history_path = metrics_dir / "train_history.csv"
        summary_path = metrics_dir / "summary.json"
        for path in (metrics_path, history_path, summary_path):
            if path.exists():
                path.unlink()

        logger = WandbLogger(self.config)
        logger.start()
        try:
            components = self.build_components()
            records = self.prepare_records(components.victim, max_records=max_images)
            if max_images is not None:
                records = records[:max_images]
            prompt_state, history = self.train_universal_prompt(
                records,
                components,
                logger=logger,
                history_path=history_path,
            )
            rows = self.evaluate_universal_prompt(
                records,
                components,
                prompt_state,
                stage=self.config.data.split or "data",
                metrics_path=metrics_path,
                logger=logger,
            )
            fid_value = compute_fid_for_rows(rows, self.config.quality.fid)
            summary = asdict(summarize_rows(rows, fid=fid_value))
            summary["training_mode"] = "universal"
            summary["train_updates"] = len(history)
            write_json(summary_path, summary)
            self.save_universal_prompt(
                prompt_state,
                metadata={
                    "training_mode": "universal",
                    "split": self.config.data.split,
                    "record_count": len(records),
                    "steps": self.config.attack.steps,
                    "attack_batch_size": self.config.attack.batch_size,
                    "generator_batch_size": self.config.generator.batch_size,
                    "objective": self.config.attack.objective,
                    "lambda_sem": self.config.attack.lambda_sem,
                    "lr": self.config.attack.lr,
                },
            )
            logger.log_summary(summary)
            return rows
        finally:
            logger.finish()

    def attack_one(
        self,
        record: ImageRecord,
        components: AttackComponents,
        *,
        logger: WandbLogger | None = None,
        image_index: int = 0,
    ) -> dict[str, Any]:
        """Run the configured attack on one image record."""
        import torch

        victim = components.victim
        semantic = components.semantic
        generator = components.generator
        quality_evaluator = components.quality_evaluator
        started_at = time.perf_counter()
        attack_loss_weight, semantic_loss_weight = attack_semantic_loss_weights(
            self.config.attack.objective,
            self.config.attack.lambda_sem,
        )
        image = load_image(record.path)
        original_tensor = pil_to_tensor(image, device=self.device)
        clean_logits = victim.logits_from_tensor(original_tensor)
        clean_eval = victim.evaluate_logits(clean_logits, record.class_index)
        prompt_state = generator.create_learnable_prompt(
            class_label=record.class_label,
            num_tokens=self.config.attack.num_learnable_tokens,
            initializer=self.config.attack.learnable_token_initializer,
            init_std=self.config.attack.learnable_token_init_std,
        )
        learnable_embeddings = prompt_state.learnable_embeddings
        if not isinstance(learnable_embeddings, torch.Tensor):
            raise TypeError("Generator prompt state must expose torch.Tensor learnable embeddings.")
        if not learnable_embeddings.requires_grad:
            raise RuntimeError("Learnable-token parameter must require gradients.")
        optimizer = torch.optim.Adam([learnable_embeddings], lr=self.config.attack.lr)
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
                prompt_state=prompt_state,
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
            if semantic_loss_weight > 0:
                dino_sim = semantic.similarity(original_tensor, generated.image_tensor)
                sem_loss = dino_loss(dino_sim)
                total_loss = weighted_attack_semantic_loss(
                    attack_loss,
                    sem_loss,
                    self.config.attack.lambda_sem,
                )
            else:
                total_loss = attack_loss
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"Non-finite loss for {record.image_id} at step {step}")
            total_loss.backward()
            optimizer.step()
            generator.sync_learnable_prompt(prompt_state)
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
                        "attack_loss_weight": attack_loss_weight,
                        "semantic_loss_weight": semantic_loss_weight,
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
            "stage": self.config.data.split or "data",
            "training_mode": self.config.attack.training_mode,
            "image_id": record.image_id,
            "class_id": record.class_index,
            "class_label": record.class_label,
            "run_name": self.config.generator.name,
            "seed": seed,
            "prompt_text": prompt_state.prompt_text,
            "num_learnable_tokens": self.config.attack.num_learnable_tokens,
            "learnable_token_initializer": self.config.attack.learnable_token_initializer,
            "learnable_token_init_std": self.config.attack.learnable_token_init_std,
            "learnable_token_texts": " ".join(prompt_state.token_texts),
            "lr": self.config.attack.lr,
            "lr_scheduler": self.config.attack.lr_scheduler.name,
            "lr_warmup_steps": self.config.attack.lr_scheduler.warmup_steps,
            "lr_min": self.config.attack.lr_scheduler.min_lr,
            "steps": self.config.attack.steps,
            "lambda_sem": self.config.attack.lambda_sem,
            "attack_loss_weight": attack_loss_weight,
            "semantic_loss_weight": semantic_loss_weight,
            "semantic_threshold": self.config.attack.semantic_threshold,
            "objective": self.config.attack.objective,
            "attack_batch_size": self.config.attack.batch_size,
            "generator_height": self.config.generator.height,
            "generator_width": self.config.generator.width,
            "generator_batch_size": self.config.generator.batch_size,
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

    def attack_batch(
        self,
        records: list[ImageRecord],
        components: AttackComponents,
        *,
        logger: WandbLogger | None = None,
        image_start_index: int = 0,
    ) -> list[dict[str, Any]]:
        """Run the configured attack on one image batch with one generator forward per step."""
        import torch

        if not records:
            return []
        victim = components.victim
        semantic = components.semantic
        generator = components.generator
        quality_evaluator = components.quality_evaluator
        started_at = time.perf_counter()
        attack_loss_weight, semantic_loss_weight = attack_semantic_loss_weights(
            self.config.attack.objective,
            self.config.attack.lambda_sem,
        )
        images = [load_image(record.path) for record in records]
        reference_images = [
            image.resize((self.config.generator.width, self.config.generator.height))
            for image in images
        ]
        original_tensor = torch.cat(
            [pil_to_tensor(image, device=self.device) for image in reference_images],
            dim=0,
        )
        true_labels = torch.tensor(
            [record.class_index for record in records],
            device=self.device,
            dtype=torch.long,
        )
        clean_logits = victim.logits_from_tensor(original_tensor)
        clean_evals = victim.evaluate_logits_batch(clean_logits.detach(), true_labels.detach().cpu().tolist())
        prompt_state = generator.create_learnable_prompt_batch(
            class_labels=[record.class_label for record in records],
            num_tokens=self.config.attack.num_learnable_tokens,
            initializer=self.config.attack.learnable_token_initializer,
            init_std=self.config.attack.learnable_token_init_std,
        )
        learnable_embeddings = prompt_state.learnable_embeddings
        if not isinstance(learnable_embeddings, torch.Tensor):
            raise TypeError("Generator batch prompt state must expose torch.Tensor learnable embeddings.")
        if learnable_embeddings.ndim != 3:
            raise ValueError("Batch learnable-token parameter must have shape [B, N, D].")
        if learnable_embeddings.shape[0] != len(records):
            raise ValueError("Batch learnable-token parameter batch dimension does not match records.")
        if not learnable_embeddings.requires_grad:
            raise RuntimeError("Batch learnable-token parameter must require gradients.")
        optimizer = torch.optim.Adam([learnable_embeddings], lr=self.config.attack.lr)
        lr_scheduler = self._build_lr_scheduler(optimizer)
        seeds = [stable_image_seed(0, record.image_id) for record in records]
        best: list[dict[str, Any] | None] = [None] * len(records)
        best_attack: list[dict[str, Any] | None] = [None] * len(records)
        best_semantic_success: list[dict[str, Any] | None] = [None] * len(records)
        min_true_conf: list[dict[str, Any] | None] = [None] * len(records)
        first_success_step = [-1] * len(records)
        first_semantic_success_step = [-1] * len(records)

        for step in range(self.config.attack.steps):
            optimizer.zero_grad(set_to_none=True)
            current_lr = float(optimizer.param_groups[0]["lr"])
            generated = generator.generate_batch(
                input_images=images,
                input_tensor=original_tensor,
                prompt_state=prompt_state,
                seeds=seeds,
                require_grad=True,
            )
            image_tensor = generated.image_tensor
            if not isinstance(image_tensor, torch.Tensor):
                raise TypeError("Generator batch result must expose torch.Tensor image_tensor.")
            if image_tensor.shape[0] != len(records):
                raise RuntimeError(
                    f"Generator returned batch size {image_tensor.shape[0]} for {len(records)} records."
                )
            logits = victim.logits_from_tensor(image_tensor)
            attack_losses = attack_loss_from_objective(
                logits,
                true_labels,
                self.config.attack.objective,
                reduction="none",
            )
            dino_sim = None
            sem_losses = None
            if semantic_loss_weight > 0:
                dino_sim = semantic.similarity(original_tensor, image_tensor)
                sem_losses = 1.0 - dino_sim
                total_losses = (1.0 - self.config.attack.lambda_sem) * attack_losses
                total_losses = total_losses + self.config.attack.lambda_sem * sem_losses
            else:
                total_losses = attack_losses
            total_loss = total_losses.mean()
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"Non-finite batch loss at step {step}")
            total_loss.backward()
            optimizer.step()
            generator.sync_learnable_prompt_batch(prompt_state)
            if lr_scheduler is not None:
                lr_scheduler.step()

            eval_results = victim.evaluate_logits_batch(logits.detach(), true_labels.detach().cpu().tolist())
            if dino_sim is None or sem_losses is None:
                with torch.no_grad():
                    dino_sim = semantic.similarity(original_tensor, image_tensor.detach())
                    sem_losses = 1.0 - dino_sim
            for index, record in enumerate(records):
                eval_result = eval_results[index]
                success = eval_result.pred != record.class_index
                semantic_value = float(dino_sim[index].detach().cpu().item())
                semantic_constrained_success = (
                    success and semantic_value >= self.config.attack.semantic_threshold
                )
                current = {
                    "step": step,
                    "image_tensor": image_tensor[index : index + 1].detach(),
                    "attack_loss": float(attack_losses[index].detach().cpu().item()),
                    "dino_loss": float(sem_losses[index].detach().cpu().item()),
                    "total_loss": float(total_losses[index].detach().cpu().item()),
                    "adv_pred": eval_result.pred,
                    "adv_top1_conf": eval_result.pred_conf,
                    "adv_true_conf": eval_result.true_conf,
                    "adv_margin": eval_result.margin,
                    "dino_similarity": semantic_value,
                    "success": success,
                    "semantic_constrained_success": semantic_constrained_success,
                }
                if success and first_success_step[index] < 0:
                    first_success_step[index] = step
                if semantic_constrained_success and first_semantic_success_step[index] < 0:
                    first_semantic_success_step[index] = step
                if logger is not None:
                    clean_eval = clean_evals[index]
                    logger.log_step(
                        image_index=image_start_index + index,
                        image_id=record.image_id,
                        class_label=record.class_label,
                        attack_step=step,
                        values={
                            "attack_loss": current["attack_loss"],
                            "attack_objective_loss": current["attack_loss"],
                            "semantic_loss": current["dino_loss"],
                            "attack_loss_weight": attack_loss_weight,
                            "semantic_loss_weight": semantic_loss_weight,
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
                previous_best = best[index]
                if previous_best is None or current["total_loss"] < previous_best["total_loss"]:
                    best[index] = current
                previous_best_attack = best_attack[index]
                if (
                    previous_best_attack is None
                    or current["adv_margin"] < previous_best_attack["adv_margin"]
                ):
                    best_attack[index] = current
                previous_min_true_conf = min_true_conf[index]
                if (
                    previous_min_true_conf is None
                    or current["adv_true_conf"] < previous_min_true_conf["adv_true_conf"]
                ):
                    min_true_conf[index] = current
                previous_semantic_success = best_semantic_success[index]
                if semantic_constrained_success and (
                    previous_semantic_success is None
                    or current["adv_margin"] < previous_semantic_success["adv_margin"]
                ):
                    best_semantic_success[index] = current

        rows: list[dict[str, Any]] = []
        runtime_seconds = time.perf_counter() - started_at
        for index, record in enumerate(records):
            best_row = best[index]
            best_attack_row = best_attack[index]
            min_true_conf_row = min_true_conf[index]
            if best_row is None or best_attack_row is None or min_true_conf_row is None:
                raise RuntimeError(f"No optimization step ran for {record.image_id}")
            adv_image = tensor_to_pil(best_row["image_tensor"])
            output_dir = self.config.output.root / "images" / record.class_label / record.image_id
            if self.config.output.save_images:
                save_image(images[index], output_dir / "original.png")
                save_image(adv_image, output_dir / "adv.png")
            grid_path = self.config.output.root / "grids" / f"{record.class_label}_{record.image_id}.png"
            if self.config.output.save_grids:
                grid = make_side_by_side(images[index], adv_image, "original", "adv")
                save_image(grid, grid_path)

            clean_eval = clean_evals[index]
            success = int(best_row["adv_pred"]) != record.class_index
            confidence_drop = clean_eval.true_conf - float(best_row["adv_true_conf"])
            semantic_constrained_success = (
                success and float(best_row["dino_similarity"]) >= self.config.attack.semantic_threshold
            )
            original_one = original_tensor[index : index + 1]
            pixel_metrics = pixel_distance_metrics(original_one, best_row["image_tensor"])
            ssim = global_ssim(original_one, best_row["image_tensor"])
            nriqa_metrics = (
                quality_evaluator.score_tensor(best_row["image_tensor"])
                if quality_evaluator is not None
                else {}
            )
            semantic_success_step = best_semantic_success[index]
            row = {
                "stage": self.config.data.split or "data",
                "training_mode": self.config.attack.training_mode,
                "image_id": record.image_id,
                "class_id": record.class_index,
                "class_label": record.class_label,
                "run_name": self.config.generator.name,
                "seed": seeds[index],
                "prompt_text": prompt_state.prompt_texts[index],
                "num_learnable_tokens": self.config.attack.num_learnable_tokens,
                "learnable_token_initializer": self.config.attack.learnable_token_initializer,
                "learnable_token_init_std": self.config.attack.learnable_token_init_std,
                "learnable_token_texts": " ".join(prompt_state.token_texts),
                "lr": self.config.attack.lr,
                "lr_scheduler": self.config.attack.lr_scheduler.name,
                "lr_warmup_steps": self.config.attack.lr_scheduler.warmup_steps,
                "lr_min": self.config.attack.lr_scheduler.min_lr,
                "steps": self.config.attack.steps,
                "lambda_sem": self.config.attack.lambda_sem,
                "attack_loss_weight": attack_loss_weight,
                "semantic_loss_weight": semantic_loss_weight,
                "semantic_threshold": self.config.attack.semantic_threshold,
                "objective": self.config.attack.objective,
                "attack_batch_size": self.config.attack.batch_size,
                "generator_height": self.config.generator.height,
                "generator_width": self.config.generator.width,
                "generator_batch_size": self.config.generator.batch_size,
                "num_inference_steps": self.config.generator.num_inference_steps,
                "clean_pred": clean_eval.pred,
                "clean_pred_label": victim.categories[clean_eval.pred],
                "clean_top1_conf": clean_eval.pred_conf,
                "adv_pred": int(best_row["adv_pred"]),
                "adv_pred_label": victim.categories[int(best_row["adv_pred"])],
                "adv_top1_conf": float(best_row["adv_top1_conf"]),
                "success": success,
                "semantic_constrained_success": semantic_constrained_success,
                "clean_true_conf": clean_eval.true_conf,
                "adv_true_conf": float(best_row["adv_true_conf"]),
                "confidence_drop": confidence_drop,
                "clean_margin": clean_eval.margin,
                "adv_margin": float(best_row["adv_margin"]),
                "margin_drop": clean_eval.margin - float(best_row["adv_margin"]),
                "dino_similarity": float(best_row["dino_similarity"]),
                "ssim": ssim,
                **pixel_metrics,
                **nriqa_metrics,
                "best_step": int(best_row["step"]),
                "best_attack_step": int(best_attack_row["step"]),
                "best_semantic_success_step": (
                    -1 if semantic_success_step is None else int(semantic_success_step["step"])
                ),
                "first_success_step": first_success_step[index],
                "first_semantic_success_step": first_semantic_success_step[index],
                "min_adv_true_conf": float(min_true_conf_row["adv_true_conf"]),
                "min_adv_true_conf_step": int(min_true_conf_row["step"]),
                "min_adv_margin": float(best_attack_row["adv_margin"]),
                "min_adv_margin_step": int(best_attack_row["step"]),
                "best_attack_loss": float(best_row["attack_loss"]),
                "best_dino_loss": float(best_row["dino_loss"]),
                "best_total_loss": float(best_row["total_loss"]),
                "runtime_seconds": runtime_seconds,
                "output_image_path": str(output_dir / "adv.png"),
                "original_image_path": str(output_dir / "original.png"),
                "grid_image_path": str(grid_path),
            }
            if logger is not None:
                logger.log_image_result(row=row, original=images[index], adversarial=adv_image)
            rows.append(row)
        return rows
