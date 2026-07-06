"""End-to-end textual-inversion token attack runner."""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from math import cos, pi
from pathlib import Path
from typing import Any

from tqdm import tqdm

from prompt_attack.attacks.axis_tokens import (
    AxisPromptState,
    build_axis_prompt_batch,
    embeddings_at,
    rank_weighted_strength_weights,
    split_anchor_axis_tokens,
)
from prompt_attack.attacks.learnable_tokens import build_prompt
from prompt_attack.attacks.losses import (
    attack_semantic_loss_weights,
    is_margin_dino_constraint_objective,
    is_saturating_ce_constraint_objective,
    is_semantic_only_objective,
    objective_loss_components,
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
from prompt_attack.utils.image import (
    AsyncImageWriter,
    image_extension,
    make_side_by_side,
    pil_to_tensor,
    tensor_to_pil,
)
from prompt_attack.utils.io import append_csv_row, ensure_dir, write_csv_rows, write_json
from prompt_attack.utils.process_title import set_process_title
from prompt_attack.utils.seed import stable_image_seed
from prompt_attack.utils.wandb_logger import WandbLogger


CLEAN_FILTER_BATCH_SIZE = 512
CLEAN_FILTER_CACHE_DIR = Path("outputs/cache/clean_correct")
CLEAN_FILTER_CACHE_VERSION = 3
CLEAN_FILTER_PREPROCESS_SIGNATURE = "generator_reference_resize_tensor_v1"
TRAIN_SHUFFLE_SEED = 0


@dataclass(frozen=True)
class LoadedBatchInputs:
    """CPU-side loaded image batch ready to move to the training device."""

    images: list[Any]
    original_tensor: Any
    labels: list[int]


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


def _short_objective_label(objective: str) -> str:
    """Return a compact objective label for process titles."""
    normalized = objective.lower().replace("-", "_")
    labels = {
        "negative_cross_entropy": "ce",
        "neg_cross_entropy": "ce",
        "negce": "ce",
        "untargeted_negative_cross_entropy": "ce",
        "untargeted_margin": "margin",
        "clip_img2img": "clip",
        "clip_image": "clip",
        "clip_image_to_image": "clip",
        "clip_only": "clip",
        "dino_img2img": "dino",
        "dino_image": "dino",
        "dino_image_to_image": "dino",
        "dino_only": "dino",
        "margin_clip_img2img": "mclip",
        "margin_dino": "mdino",
        "margin_lpips": "mlpips",
        "margin_lpips_img2img": "mlpips",
        "margin_oracle": "moracle",
        "margin_oracle_clip": "moracle",
        "sat_ce_oracle": "satce",
        "saturating_ce_oracle": "satce",
        "cr": "cr",
    }
    return labels.get(normalized, normalized[:10])


def _process_title(config: ExperimentConfig, stage: str, **fields: str | int | float) -> str:
    """Return a readable process title for ps/nvidia-smi visibility."""
    stage_label = {
        "tr-eval": "train-eval",
    }.get(stage, stage)
    parts = [
        "prompt-attack",
        "uap",
        stage_label,
        f"obj={_short_objective_label(config.attack.objective)}",
        f"tokens={config.attack.num_learnable_tokens}",
        f"global_batch={config.attack.batch_size}",
    ]
    parts.extend(f"{key}={value}" for key, value in fields.items())
    return " ".join(parts)


def _progress_eta(*, started_at: float, completed: int, total: int) -> tuple[float, float]:
    """Return elapsed seconds and ETA seconds for a progress counter."""
    elapsed = time.perf_counter() - started_at
    if completed <= 0 or total <= 0:
        return elapsed, 0.0
    remaining = max(total - completed, 0)
    eta = elapsed / completed * remaining
    return elapsed, eta


def _semantic_metric_fields(
    semantic: Any,
    value: float,
    *,
    dino_value: float | None = None,
) -> dict[str, Any]:
    """Return generic and model-specific semantic similarity CSV fields."""
    metric_name = str(getattr(semantic, "metric_name", "semantic_similarity"))
    fields: dict[str, Any] = {
        "semantic_model": str(getattr(semantic, "name", type(semantic).__name__)),
        "semantic_metric": metric_name,
        "semantic_similarity": value,
        "dino_similarity": "",
        "clip_image_similarity": "",
    }
    if metric_name == "dino_similarity":
        fields["dino_similarity"] = value
    elif metric_name == "clip_image_similarity":
        fields["clip_image_similarity"] = value
    if dino_value is not None:
        fields["dino_similarity"] = dino_value
    return fields


def _logged_semantic_weight(config: ExperimentConfig, objective_weight: float) -> float:
    """Return the semantic loss weight that actually scales the configured objective."""
    if is_semantic_only_objective(config.attack.objective):
        return 1.0
    if is_margin_dino_constraint_objective(config.attack.objective) or (
        is_saturating_ce_constraint_objective(config.attack.objective)
    ):
        return config.attack.semantic_loss_weight
    return objective_weight


@dataclass(frozen=True)
class AttackComponents:
    """Loaded model components shared across attack entry points."""

    victim: Any
    semantic: Any
    dino_metric: Any | None
    generator: Any
    quality_evaluator: Any


class LearnableTokenAttackRunner:
    """Run textual-inversion token attacks."""

    def __init__(self, config: ExperimentConfig, *, device: str) -> None:
        self.config = config
        self.device = device

    def _image_writer(self) -> AsyncImageWriter:
        """Create a background writer using the configured output policy."""
        return AsyncImageWriter(
            max_workers=self.config.output.image_save_workers,
            image_format=self.config.output.image_format,
            quality=self.config.output.image_quality,
        )

    def _image_file_path(self, output_dir: Path, stem: str) -> Path:
        return output_dir / f"{stem}{image_extension(self.config.output.image_format)}"

    def _grid_file_path(self, *, stage: str, class_label: str, image_id: str) -> Path:
        return (
            self.config.output.root
            / "grids"
            / f"{stage}_{class_label}_{image_id}{image_extension(self.config.output.image_format)}"
        )

    def _save_eval_images(
        self,
        *,
        writer: AsyncImageWriter,
        original: Any,
        adversarial: Any,
        output_dir: Path,
    ) -> tuple[Path, Path]:
        """Queue configured original/adv image writes and return their row paths."""
        original_path = self._image_file_path(output_dir, "original")
        adv_path = self._image_file_path(output_dir, "adv")
        if self.config.output.save_images and self.config.output.save_original_images:
            writer.save(original, original_path)
        if self.config.output.save_images and self.config.output.save_adv_images:
            writer.save(adversarial, adv_path)
        return original_path, adv_path

    def _save_grid_if_all_policy(
        self,
        *,
        writer: AsyncImageWriter,
        original: Any,
        adversarial: Any,
        grid_path: Path,
    ) -> bool:
        if not self.config.output.save_grids:
            return False
        policy = self.config.output.grid_save_policy.lower()
        if policy != "all":
            return False
        grid = make_side_by_side(original, adversarial, "original", "adv")
        writer.save(grid, grid_path)
        return True

    def _representative_grid_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Pick a compact mix of successes, semantic drifts, and failures for grids."""
        if not self.config.output.save_grids:
            return []
        if self.config.output.grid_save_policy.lower() != "representative":
            return []
        limit = self.config.output.max_saved_grids
        if limit <= 0:
            return []

        def is_success(row: dict[str, Any]) -> bool:
            return str(row.get("success", "")).lower() in {"1", "true", "yes", "y"}

        def semantic(row: dict[str, Any]) -> float:
            return float(row.get("semantic_similarity", 0.0))

        successes = [row for row in rows if is_success(row)]
        failures = [row for row in rows if not is_success(row)]
        buckets = [
            sorted(successes, key=semantic, reverse=True),
            sorted(successes, key=semantic),
            sorted(failures, key=semantic, reverse=True),
        ]
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        while len(selected) < limit and any(buckets):
            for bucket in buckets:
                while bucket:
                    row = bucket.pop(0)
                    key = f"{row.get('stage')}::{row.get('class_label')}::{row.get('image_id')}"
                    if key in seen:
                        continue
                    seen.add(key)
                    selected.append(row)
                    break
                if len(selected) >= limit:
                    break
        return selected

    def _save_representative_grids(self, rows: list[dict[str, Any]]) -> None:
        selected = self._representative_grid_rows(rows)
        if not selected:
            return
        with self._image_writer() as writer:
            for row in selected:
                original_path = Path(str(row["original_image_path"]))
                adv_path = Path(str(row["output_image_path"]))
                if not original_path.exists() or not adv_path.exists():
                    continue
                original = load_image(original_path)
                adversarial = load_image(adv_path)
                grid = make_side_by_side(original, adversarial, "original", "adv")
                writer.save(grid, Path(str(row["grid_image_path"])))
                row["grid_saved"] = True

    def build_components(self) -> AttackComponents:
        """Load reusable attack models and evaluators."""
        victim = build_victim(
            self.config.victim.name,
            weights=self.config.victim.weights,
            checkpoint_path=self.config.victim.checkpoint_path,
            device=self.device,
        )
        semantic = build_semantic_model(
            self.config.semantic.name,
            device=self.device,
            class_labels=getattr(victim, "categories", None),
        )
        dino_metric = None
        if str(getattr(semantic, "metric_name", "")) != "dino_similarity":
            dino_metric = build_semantic_model("dinov2_vitb14", device=self.device)
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
            dino_metric=dino_metric,
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

        def selection_class_key(record: ImageRecord) -> str:
            return record.superclass_synset or record.synset

        def can_select(record: ImageRecord) -> bool:
            return cap is None or per_class[selection_class_key(record)] < cap

        if not self.config.data.clean_correct_only:
            for record in tqdm(records, desc="clean-correct filter"):
                if max_records is not None and len(selected) >= max_records:
                    break
                if not can_select(record):
                    continue
                selected.append(record)
                per_class[selection_class_key(record)] += 1
            return selected

        cache_path, cache_metadata = self._clean_correct_cache_path()
        checked_keys, clean_keys = self._load_clean_correct_cache(
            cache_path,
            cache_metadata,
        )
        if checked_keys:
            print(
                "clean-correct cache loaded: "
                f"{len(clean_keys)}/{len(checked_keys)} clean from {cache_path}",
                flush=True,
            )

        def record_key(record: ImageRecord) -> str:
            return f"{record.synset}/{record.image_id}"

        def write_cache() -> None:
            write_json(
                cache_path,
                {
                    "metadata": cache_metadata,
                    "checked_keys": sorted(checked_keys),
                    "clean_keys": sorted(clean_keys),
                },
            )

        filter_batch_size = CLEAN_FILTER_BATCH_SIZE
        pending: list[ImageRecord] = []
        progress = tqdm(total=len(records), desc="clean-correct filter")

        def flush_pending() -> None:
            nonlocal pending
            if not pending:
                return
            batch = pending
            pending = []
            labels = [record.class_index for record in batch]
            if hasattr(victim, "logits_from_tensor") and hasattr(victim, "evaluate_logits_batch"):
                import torch

                loaded = self._load_batch_inputs_cpu(batch)
                _, original_tensor, true_labels = self._loaded_batch_to_device(loaded)
                with torch.no_grad():
                    logits = victim.logits_from_tensor(original_tensor)
                results = victim.evaluate_logits_batch(
                    logits,
                    true_labels.detach().cpu().tolist(),
                )
            else:
                images = [load_image(record.path) for record in batch]
                if hasattr(victim, "evaluate_pil_batch"):
                    results = victim.evaluate_pil_batch(images, labels)
                else:
                    results = [
                        victim.evaluate_pil(image, label) for image, label in zip(images, labels)
                    ]
            for record, result in zip(batch, results):
                key = record_key(record)
                checked_keys.add(key)
                if result.pred == record.class_index:
                    clean_keys.add(key)
                if max_records is not None and len(selected) >= max_records:
                    continue
                if not can_select(record):
                    continue
                if key in clean_keys:
                    selected.append(record)
                    per_class[selection_class_key(record)] += 1
            write_cache()
            progress.update(len(batch))

        try:
            for record in records:
                if max_records is not None and len(selected) >= max_records:
                    break
                if not can_select(record):
                    progress.update(1)
                    continue
                key = record_key(record)
                if key in checked_keys:
                    if key in clean_keys:
                        selected.append(record)
                        per_class[selection_class_key(record)] += 1
                    progress.update(1)
                    continue
                pending.append(record)
                if len(pending) >= filter_batch_size:
                    flush_pending()
            flush_pending()
        finally:
            progress.close()
        return selected

    def _clean_correct_cache_path(self) -> tuple[Path, dict[str, Any]]:
        metadata = {
            "cache_version": CLEAN_FILTER_CACHE_VERSION,
            "preprocess_signature": CLEAN_FILTER_PREPROCESS_SIGNATURE,
            "imagenet_root": str(self.config.data.imagenet_root),
            "split": self.config.data.split,
            "class_mode": self.config.data.class_mode,
            "images_per_class": self.config.data.images_per_class,
            "candidate_multiplier": self.config.data.candidate_multiplier,
            "generator_width": self.config.generator.width,
            "generator_height": self.config.generator.height,
            "victim_name": self.config.victim.name,
            "victim_weights": self.config.victim.weights,
            "victim_checkpoint_path": (
                None
                if self.config.victim.checkpoint_path is None
                else str(self.config.victim.checkpoint_path)
            ),
        }
        digest = hashlib.sha1(
            json.dumps(metadata, sort_keys=True).encode("utf-8"),
        ).hexdigest()[:16]
        filename = (
            f"{self.config.data.class_mode}_{self.config.data.split}_"
            f"{self.config.victim.name}_{digest}.json"
        )
        return CLEAN_FILTER_CACHE_DIR / filename, metadata

    def _load_clean_correct_cache(
        self,
        path: Path,
        metadata: dict[str, Any],
    ) -> tuple[set[str], set[str]]:
        if not path.exists():
            return set(), set()
        try:
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError):
            return set(), set()
        if raw.get("metadata") != metadata:
            return set(), set()
        checked = raw.get("checked_keys", [])
        clean = raw.get("clean_keys", [])
        if not isinstance(checked, list) or not isinstance(clean, list):
            return set(), set()
        return set(str(key) for key in checked), set(str(key) for key in clean)

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

    def _shuffled_record_batches(
        self,
        records: list[ImageRecord],
        *,
        batch_size: int,
        epoch: int,
    ) -> list[list[ImageRecord]]:
        """Split records into deterministic shuffled batches for one train epoch."""
        shuffled = list(records)
        random.Random(TRAIN_SHUFFLE_SEED + epoch).shuffle(shuffled)
        return self._record_batches(shuffled, batch_size=batch_size)

    def _shared_prompt_batch(
        self,
        prompt_state: LearnablePrompt,
        records: list[ImageRecord],
    ) -> LearnablePromptBatch:
        """Build per-class prompt texts backed by one shared learnable token tensor."""
        return LearnablePromptBatch(
            prompt_texts=tuple(
                build_prompt(record.class_label, len(prompt_state.token_texts))
                for record in records
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

    def _load_batch_inputs_cpu(self, records: list[ImageRecord]) -> LoadedBatchInputs:
        """Load and resize a batch on CPU for background prefetch."""
        import torch
        import torchvision.transforms.functional as F

        images = [load_image(record.path) for record in records]
        reference_images = [
            image.resize((self.config.generator.width, self.config.generator.height))
            for image in images
        ]
        original_tensor = torch.cat(
            [F.to_tensor(image).unsqueeze(0) for image in reference_images],
            dim=0,
        ).to(dtype=torch.float32)
        labels = [record.class_index for record in records]
        return LoadedBatchInputs(
            images=images,
            original_tensor=original_tensor,
            labels=labels,
        )

    def _loaded_batch_to_device(self, loaded: LoadedBatchInputs):
        """Move a prefetched CPU batch to the configured device."""
        import torch

        original_tensor = loaded.original_tensor
        non_blocking = False
        if str(self.device).startswith("cuda"):
            original_tensor = original_tensor.pin_memory()
            non_blocking = True
        original_tensor = original_tensor.to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=non_blocking,
        )
        true_labels = torch.tensor(
            loaded.labels,
            device=self.device,
            dtype=torch.long,
        )
        return loaded.images, original_tensor, true_labels

    def _prefetched_record_batches(
        self,
        batches: list[list[ImageRecord]],
        *,
        enabled: bool,
    ) -> Iterator[tuple[list[ImageRecord], list[Any], Any, Any]]:
        """Yield batches with CPU image loading overlapped with GPU work."""
        if not enabled or len(batches) <= 1:
            for batch in batches:
                images, original_tensor, true_labels = self._load_batch_inputs(batch)
                yield batch, images, original_tensor, true_labels
            return

        with ThreadPoolExecutor(max_workers=1) as executor:
            pending: Future[LoadedBatchInputs] | None = executor.submit(
                self._load_batch_inputs_cpu,
                batches[0],
            )
            for index, batch in enumerate(batches):
                if pending is None:
                    raise RuntimeError("Missing prefetched batch.")
                loaded = pending.result()
                next_index = index + 1
                pending = (
                    executor.submit(self._load_batch_inputs_cpu, batches[next_index])
                    if next_index < len(batches)
                    else None
                )
                images, original_tensor, true_labels = self._loaded_batch_to_device(loaded)
                yield batch, images, original_tensor, true_labels

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
            init_seed=self.config.attack.learnable_token_init_seed,
        )

    def _apply_initial_prompt_checkpoint(self, prompt_state: LearnablePrompt) -> LearnablePrompt:
        """Replace a fresh universal prompt parameter with a saved learned prompt."""
        init_prompt_path = self.config.attack.init_prompt_path
        if init_prompt_path is None:
            return prompt_state

        import torch

        checkpoint_path = Path(init_prompt_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Initial prompt checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict) or "learnable_embeddings" not in checkpoint:
            raise ValueError(
                "Initial prompt checkpoint must be a dict containing 'learnable_embeddings'."
            )

        expected_tokens = len(prompt_state.token_texts)
        checkpoint_token_texts = tuple(checkpoint.get("token_texts", ()))
        if checkpoint_token_texts and len(checkpoint_token_texts) != expected_tokens:
            raise ValueError(
                "Initial prompt checkpoint token count mismatch: "
                f"checkpoint has {len(checkpoint_token_texts)}, config expects {expected_tokens}."
            )
        if checkpoint_token_texts and checkpoint_token_texts != prompt_state.token_texts:
            raise ValueError(
                "Initial prompt checkpoint token texts do not match the current prompt tokens."
            )

        current_embeddings = prompt_state.learnable_embeddings
        saved_embeddings = checkpoint["learnable_embeddings"]
        if not isinstance(current_embeddings, torch.Tensor):
            raise TypeError("Current prompt embeddings must be a torch.Tensor.")
        if not isinstance(saved_embeddings, torch.Tensor):
            raise TypeError("Initial prompt 'learnable_embeddings' must be a torch.Tensor.")
        if tuple(saved_embeddings.shape) != tuple(current_embeddings.shape):
            raise ValueError(
                "Initial prompt embedding shape mismatch: "
                f"checkpoint has {tuple(saved_embeddings.shape)}, "
                f"current prompt expects {tuple(current_embeddings.shape)}."
            )

        loaded_embeddings = torch.nn.Parameter(
            saved_embeddings.detach()
            .clone()
            .to(device=current_embeddings.device, dtype=current_embeddings.dtype)
        )
        return replace(prompt_state, learnable_embeddings=loaded_embeddings)

    def create_universal_axis_prompt(self, components: AttackComponents) -> AxisPromptState:
        """Create the shared anchor/axis prompt used by strength-scheduled attacks."""
        num_anchor_tokens, _ = split_anchor_axis_tokens(
            self.config.attack.num_learnable_tokens,
            self.config.attack.num_anchor_tokens,
        )
        return components.generator.create_axis_prompt(
            class_label="object",
            num_tokens=self.config.attack.num_learnable_tokens,
            num_anchor_tokens=num_anchor_tokens,
            initializer=self.config.attack.learnable_token_initializer,
            init_std=self.config.attack.learnable_token_init_std,
            init_seed=self.config.attack.learnable_token_init_seed,
        )

    def _apply_initial_axis_prompt_checkpoint(self, axis_state: AxisPromptState) -> AxisPromptState:
        """Replace a fresh axis prompt with a saved checkpoint (new or legacy format)."""
        init_prompt_path = self.config.attack.init_prompt_path
        if init_prompt_path is None:
            return axis_state

        import torch

        checkpoint_path = Path(init_prompt_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Initial prompt checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict) or "learnable_embeddings" not in checkpoint:
            raise ValueError(
                "Initial prompt checkpoint must be a dict containing 'learnable_embeddings'."
            )

        expected_tokens = len(axis_state.token_texts)
        checkpoint_token_texts = tuple(checkpoint.get("token_texts", ()))
        if checkpoint_token_texts and len(checkpoint_token_texts) != expected_tokens:
            raise ValueError(
                "Initial prompt checkpoint token count mismatch: "
                f"checkpoint has {len(checkpoint_token_texts)}, config expects {expected_tokens}."
            )
        if checkpoint_token_texts and checkpoint_token_texts != axis_state.token_texts:
            raise ValueError(
                "Initial prompt checkpoint token texts do not match the current prompt tokens."
            )

        anchor_device = axis_state.anchor_embeddings.device
        anchor_dtype = axis_state.anchor_embeddings.dtype

        if "anchor_embeddings" in checkpoint and "axis_direction" in checkpoint:
            # New-format (format_version >= 2) checkpoint: load the three tensors directly.
            saved_anchor = checkpoint["anchor_embeddings"]
            saved_axis_base = checkpoint["axis_base"]
            saved_axis_direction = checkpoint["axis_direction"]
            for name, saved, current in (
                ("anchor_embeddings", saved_anchor, axis_state.anchor_embeddings),
                ("axis_base", saved_axis_base, axis_state.axis_base),
                ("axis_direction", saved_axis_direction, axis_state.axis_direction),
            ):
                if tuple(saved.shape) != tuple(current.shape):
                    raise ValueError(
                        f"Initial prompt checkpoint '{name}' shape mismatch: "
                        f"checkpoint has {tuple(saved.shape)}, current prompt expects "
                        f"{tuple(current.shape)}."
                    )
            return replace(
                axis_state,
                anchor_embeddings=torch.nn.Parameter(
                    saved_anchor.detach().clone().to(device=anchor_device, dtype=anchor_dtype)
                ),
                axis_base=saved_axis_base.detach().clone().to(device=anchor_device, dtype=anchor_dtype),
                axis_direction=torch.nn.Parameter(
                    saved_axis_direction.detach()
                    .clone()
                    .to(device=anchor_device, dtype=anchor_dtype)
                ),
            )

        # Legacy flat-tensor checkpoint (pre-axis-mode run): warm-start by splitting the
        # learned overlay into an anchor half and an axis-base half, with a zero-initialized
        # axis direction (the old run recorded no direction information).
        saved_embeddings = checkpoint["learnable_embeddings"]
        if tuple(saved_embeddings.shape)[0] != expected_tokens:
            raise ValueError(
                "Legacy initial prompt checkpoint token count mismatch: "
                f"checkpoint has {tuple(saved_embeddings.shape)[0]} rows, "
                f"config expects {expected_tokens}."
            )
        num_anchor = axis_state.num_anchor_tokens
        saved_embeddings = saved_embeddings.detach().clone().to(
            device=anchor_device, dtype=anchor_dtype
        )
        return replace(
            axis_state,
            anchor_embeddings=torch.nn.Parameter(saved_embeddings[:num_anchor]),
            axis_base=saved_embeddings[num_anchor:],
            axis_direction=torch.nn.Parameter(torch.zeros_like(axis_state.axis_direction)),
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
        prompt_state = self._apply_initial_prompt_checkpoint(
            self.create_universal_prompt(components)
        )
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
        num_batches = len(self._record_batches(records, batch_size=effective_batch_size))
        current_epoch = -1
        batches: list[list[ImageRecord]] = []
        history: list[dict[str, Any]] = []
        train_started_at = time.perf_counter()

        progress = tqdm(
            range(self.config.attack.steps),
            desc="uap-train",
            disable=not dist_context.is_rank0,
            dynamic_ncols=True,
        )
        for step in progress:
            epoch = step // num_batches
            if epoch != current_epoch:
                current_epoch = epoch
                batches = self._shuffled_record_batches(
                    records,
                    batch_size=effective_batch_size,
                    epoch=epoch,
                )
            batch_index = step % num_batches
            batch = batches[batch_index]
            local_batch = dist_context.shard(batch)
            optimizer.zero_grad(set_to_none=True)
            current_lr = float(optimizer.param_groups[0]["lr"])
            step_attack_loss = 0.0
            step_semantic_loss = 0.0
            step_weighted_semantic_loss = 0.0
            step_total_loss = 0.0
            step_success = 0
            processed = 0

            micro_batches = self._record_batches(local_batch, batch_size=micro_batch_size)
            for (
                micro_batch,
                images,
                original_tensor,
                true_labels,
            ) in self._prefetched_record_batches(
                micro_batches,
                enabled=True,
            ):
                prompt_batch = self._shared_prompt_batch(prompt_state, micro_batch)
                if self.config.attack.eot_train_seeds:
                    # Expectation over Transformation: resample the diffusion trajectory
                    # each step so the token cannot memorise one deterministic path into a
                    # fixed overlay. A step-dependent base keeps each image's seed distinct
                    # yet reproducible.
                    seeds = [
                        stable_image_seed(step + 1, record.image_id) for record in micro_batch
                    ]
                else:
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
                semantic_sim = (
                    components.semantic.similarity(
                        original_tensor, image_tensor, labels=true_labels
                    )
                    if semantic_loss_weight > 0
                    else None
                )
                attack_losses, semantic_losses, weighted_semantic_losses, total_losses = (
                    objective_loss_components(
                        logits,
                        true_labels,
                        self.config.attack.objective,
                        semantic_similarity=semantic_sim,
                        lambda_sem=self.config.attack.lambda_sem,
                        semantic_loss_weight=self.config.attack.semantic_loss_weight,
                        attack_margin=self.config.attack.attack_margin,
                    )
                )

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
                step_semantic_loss += float(semantic_losses.detach().sum().cpu().item())
                step_weighted_semantic_loss += float(
                    weighted_semantic_losses.detach().sum().cpu().item()
                )
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
                [
                    step_attack_loss,
                    step_semantic_loss,
                    step_weighted_semantic_loss,
                    step_total_loss,
                    step_success,
                    processed,
                ],
                device=learnable_embeddings.device,
                dtype=torch.float64,
            )
            dist_context.all_reduce_sum(stats)
            global_processed = max(1.0, float(stats[5].detach().cpu().item()))
            optimizer.step()
            components.generator.sync_learnable_prompt(prompt_state)
            if lr_scheduler is not None:
                lr_scheduler.step()

            history_row = {
                "step": step,
                "epoch": epoch,
                "batch_index": batch_index,
                "lr": current_lr,
                "attack_loss": float(stats[0].detach().cpu().item()) / global_processed,
                "semantic_loss": float(stats[1].detach().cpu().item()) / global_processed,
                "weighted_semantic_loss": float(stats[2].detach().cpu().item()) / global_processed,
                "total_loss": float(stats[3].detach().cpu().item()) / global_processed,
                "success_rate": float(stats[4].detach().cpu().item()) / global_processed,
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
                    _process_title(
                        self.config,
                        "train",
                        step=f"{completed_steps}/{self.config.attack.steps}",
                        asr=f"{history_row['success_rate']:.2f}",
                        eta=_format_duration(eta),
                    )
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
                        "semantic_loss": history_row["semantic_loss"],
                        "weighted_semantic_loss": history_row["weighted_semantic_loss"],
                        "total_loss": history_row["total_loss"],
                        "lr": current_lr,
                        "success_rate": history_row["success_rate"],
                        "effective_batch_size": len(batch),
                        "micro_batch_size": micro_batch_size,
                    },
                )

        return prompt_state, history

    def train_universal_axis_prompt(
        self,
        records: list[ImageRecord],
        components: AttackComponents,
        *,
        logger: WandbLogger | None = None,
        history_path: Path | None = None,
        dist_context: DistributedContext | None = None,
    ) -> tuple[AxisPromptState, list[dict[str, Any]]]:
        """Optimize an anchor/axis prompt with a multi-strength path-based loss.

        Mirrors ``train_universal_prompt``, but at each micro-batch sweeps
        ``attack.train_strengths`` along the learned axis direction (same seeds across
        strengths within one step, so the strength axis is the only varying factor) and
        weights each strength's loss with ``rank_weighted_strength_weights`` so an early
        (small-``t``) successful crossing is rewarded more than pushing success out to
        ``t=1`` -- discouraging the model from collapsing onto a single fixed-strength
        overlay.
        """
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
        axis_state = self._apply_initial_axis_prompt_checkpoint(
            self.create_universal_axis_prompt(components)
        )
        anchor_embeddings = axis_state.anchor_embeddings
        axis_direction = axis_state.axis_direction
        if not isinstance(anchor_embeddings, torch.Tensor) or not isinstance(
            axis_direction, torch.Tensor
        ):
            raise TypeError("Axis prompt state must expose torch.Tensor parameters.")
        if not anchor_embeddings.requires_grad or not axis_direction.requires_grad:
            raise RuntimeError("Axis prompt anchor/direction parameters must require gradients.")
        dist_context.broadcast_tensor(anchor_embeddings.data)
        dist_context.broadcast_tensor(axis_direction.data)
        dist_context.broadcast_tensor(axis_state.axis_base)
        components.generator.sync_axis_prompt(axis_state, t=1.0)

        axis_lr = self.config.attack.axis_lr
        optimizer = torch.optim.Adam(
            [
                {"params": [anchor_embeddings], "lr": self.config.attack.lr},
                {"params": [axis_direction], "lr": axis_lr if axis_lr is not None else self.config.attack.lr},
            ]
        )
        lr_scheduler = self._build_lr_scheduler(optimizer)
        effective_batch_size = max(1, self.config.attack.batch_size)
        micro_batch_size = max(1, self.config.generator.batch_size)
        num_batches = len(self._record_batches(records, batch_size=effective_batch_size))
        current_epoch = -1
        batches: list[list[ImageRecord]] = []
        history: list[dict[str, Any]] = []
        train_started_at = time.perf_counter()
        sorted_strengths = sorted(self.config.attack.train_strengths)
        strength_weights = rank_weighted_strength_weights(sorted_strengths)

        progress = tqdm(
            range(self.config.attack.steps),
            desc="uap-train",
            disable=not dist_context.is_rank0,
            dynamic_ncols=True,
        )
        for step in progress:
            epoch = step // num_batches
            if epoch != current_epoch:
                current_epoch = epoch
                batches = self._shuffled_record_batches(
                    records,
                    batch_size=effective_batch_size,
                    epoch=epoch,
                )
            batch_index = step % num_batches
            batch = batches[batch_index]
            local_batch = dist_context.shard(batch)
            optimizer.zero_grad(set_to_none=True)
            current_lr = float(optimizer.param_groups[0]["lr"])
            step_attack_loss = 0.0
            step_semantic_loss = 0.0
            step_weighted_semantic_loss = 0.0
            step_total_loss = 0.0
            step_success = 0
            step_success_min_t = 0
            processed = 0

            micro_batches = self._record_batches(local_batch, batch_size=micro_batch_size)
            for (
                micro_batch,
                images,
                original_tensor,
                true_labels,
            ) in self._prefetched_record_batches(
                micro_batches,
                enabled=True,
            ):
                if self.config.attack.eot_train_seeds:
                    # Same rationale as the legacy loop's EOT handling: resample per step,
                    # but keep it fixed across the strengths sampled *within* this step so
                    # strength is the only thing varying in the sweep below.
                    seeds = [
                        stable_image_seed(step + 1, record.image_id) for record in micro_batch
                    ]
                else:
                    seeds = [stable_image_seed(0, record.image_id) for record in micro_batch]

                micro_attack_loss = 0.0
                micro_semantic_loss = 0.0
                micro_weighted_semantic_loss = 0.0
                micro_total_loss = 0.0
                micro_success_at_min_t = None
                micro_success_at_max_t = 0
                for weight_index, (t, weight) in enumerate(zip(sorted_strengths, strength_weights)):
                    prompt_batch = build_axis_prompt_batch(axis_state, micro_batch, t=t)
                    generated = components.generator.generate_batch(
                        input_images=images,
                        input_tensor=original_tensor,
                        prompt_state=prompt_batch,
                        seeds=seeds,
                        require_grad=True,
                    )
                    image_tensor = generated.image_tensor
                    if not isinstance(image_tensor, torch.Tensor):
                        raise TypeError(
                            "Generator batch result must expose torch.Tensor image_tensor."
                        )
                    logits = components.victim.logits_from_tensor(image_tensor)
                    semantic_sim = (
                        components.semantic.similarity(
                            original_tensor, image_tensor, labels=true_labels
                        )
                        if semantic_loss_weight > 0
                        else None
                    )
                    attack_losses, semantic_losses, weighted_semantic_losses, total_losses = (
                        objective_loss_components(
                            logits,
                            true_labels,
                            self.config.attack.objective,
                            semantic_similarity=semantic_sim,
                            lambda_sem=self.config.attack.lambda_sem,
                            semantic_loss_weight=self.config.attack.semantic_loss_weight,
                            attack_margin=self.config.attack.attack_margin,
                        )
                    )

                    strength_loss = weight * total_losses.sum() / len(batch)
                    if not torch.isfinite(strength_loss):
                        raise FloatingPointError(
                            f"Non-finite universal loss at step {step}, strength {t}"
                        )
                    strength_loss.backward()

                    with torch.no_grad():
                        eval_results = components.victim.evaluate_logits_batch(
                            logits.detach(),
                            true_labels.detach().cpu().tolist(),
                        )
                    success_count = sum(
                        int(result.pred != record.class_index)
                        for result, record in zip(eval_results, micro_batch)
                    )
                    if weight_index == 0:
                        micro_success_at_min_t = success_count
                    micro_success_at_max_t = success_count
                    micro_attack_loss += float(attack_losses.detach().sum().cpu().item())
                    micro_semantic_loss += float(semantic_losses.detach().sum().cpu().item())
                    micro_weighted_semantic_loss += float(
                        weighted_semantic_losses.detach().sum().cpu().item()
                    )
                    micro_total_loss += float(total_losses.detach().sum().cpu().item())

                step_attack_loss += micro_attack_loss / len(sorted_strengths)
                step_semantic_loss += micro_semantic_loss / len(sorted_strengths)
                step_weighted_semantic_loss += micro_weighted_semantic_loss / len(sorted_strengths)
                step_total_loss += micro_total_loss / len(sorted_strengths)
                step_success += micro_success_at_max_t
                step_success_min_t += micro_success_at_min_t or 0
                processed += len(micro_batch)

            for param in (anchor_embeddings, axis_direction):
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                dist_context.all_reduce_sum(param.grad)
            stats = torch.tensor(
                [
                    step_attack_loss,
                    step_semantic_loss,
                    step_weighted_semantic_loss,
                    step_total_loss,
                    step_success,
                    step_success_min_t,
                    processed,
                ],
                device=anchor_embeddings.device,
                dtype=torch.float64,
            )
            dist_context.all_reduce_sum(stats)
            global_processed = max(1.0, float(stats[6].detach().cpu().item()))
            optimizer.step()
            components.generator.sync_axis_prompt(axis_state, t=1.0)
            if lr_scheduler is not None:
                lr_scheduler.step()

            history_row = {
                "step": step,
                "epoch": epoch,
                "batch_index": batch_index,
                "lr": current_lr,
                "attack_loss": float(stats[0].detach().cpu().item()) / global_processed,
                "semantic_loss": float(stats[1].detach().cpu().item()) / global_processed,
                "weighted_semantic_loss": float(stats[2].detach().cpu().item()) / global_processed,
                "total_loss": float(stats[3].detach().cpu().item()) / global_processed,
                "success_rate": float(stats[4].detach().cpu().item()) / global_processed,
                "success_rate_at_min_t": float(stats[5].detach().cpu().item()) / global_processed,
                "train_strengths": ",".join(f"{value:g}" for value in sorted_strengths),
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
                    _process_title(
                        self.config,
                        "train",
                        step=f"{completed_steps}/{self.config.attack.steps}",
                        asr=f"{history_row['success_rate']:.2f}",
                        eta=_format_duration(eta),
                    )
                )
                progress.set_postfix(
                    {
                        "loss": f"{history_row['total_loss']:.4f}",
                        "asr": f"{history_row['success_rate']:.3f}",
                        "asr_min_t": f"{history_row['success_rate_at_min_t']:.3f}",
                        "lr": f"{current_lr:.2e}",
                        "eta": _format_duration(eta),
                    },
                    refresh=True,
                )
                progress.write(
                    "[train-axis] "
                    f"step {completed_steps}/{self.config.attack.steps} "
                    f"({completed_steps / self.config.attack.steps:.1%}) | "
                    f"batch={history_row['batch_index'] + 1}/{len(batches)} | "
                    f"loss={history_row['total_loss']:.4f} | "
                    f"attack_loss={history_row['attack_loss']:.4f} | "
                    f"asr@t1={history_row['success_rate']:.3f} | "
                    f"asr@min_t={history_row['success_rate_at_min_t']:.3f} | "
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
                        "semantic_loss": history_row["semantic_loss"],
                        "weighted_semantic_loss": history_row["weighted_semantic_loss"],
                        "total_loss": history_row["total_loss"],
                        "lr": current_lr,
                        "success_rate": history_row["success_rate"],
                        "success_rate_at_min_t": history_row["success_rate_at_min_t"],
                        "effective_batch_size": len(batch),
                        "micro_batch_size": micro_batch_size,
                    },
                )

        return axis_state, history

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
        progress = tqdm(
            batches,
            desc=f"{stage}-eval",
            disable=not show_progress,
            dynamic_ncols=True,
        )
        with self._image_writer() as image_writer:
            prefetched_batches = self._prefetched_record_batches(
                batches,
                enabled=True,
            )
            for batch_index, (batch, images, original_tensor, true_labels) in enumerate(
                prefetched_batches
            ):
                started_at = time.perf_counter()
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
                    semantic_sim = components.semantic.similarity(
                        original_tensor, image_tensor, labels=true_labels
                    )
                    dino_metric_sim = (
                        components.dino_metric.similarity(original_tensor, image_tensor)
                        if components.dino_metric is not None
                        else None
                    )
                    attack_losses, semantic_losses, weighted_semantic_losses, total_losses = (
                        objective_loss_components(
                            adv_logits,
                            true_labels,
                            self.config.attack.objective,
                            semantic_similarity=semantic_sim,
                            lambda_sem=self.config.attack.lambda_sem,
                            semantic_loss_weight=self.config.attack.semantic_loss_weight,
                            attack_margin=self.config.attack.attack_margin,
                        )
                    )

                runtime_seconds = time.perf_counter() - started_at
                for index, record in enumerate(batch):
                    clean_eval = clean_evals[index]
                    adv_eval = adv_evals[index]
                    adv_image = tensor_to_pil(image_tensor[index : index + 1])
                    output_dir = (
                        self.config.output.root
                        / "images"
                        / stage
                        / record.class_label
                        / record.image_id
                    )
                    original_output_path, adv_output_path = self._save_eval_images(
                        writer=image_writer,
                        original=images[index],
                        adversarial=adv_image,
                        output_dir=output_dir,
                    )
                    original_row_path = (
                        original_output_path
                        if self.config.output.save_images
                        and self.config.output.save_original_images
                        else record.path
                    )
                    grid_path = self._grid_file_path(
                        stage=stage,
                        class_label=record.class_label,
                        image_id=record.image_id,
                    )
                    grid_saved = self._save_grid_if_all_policy(
                        writer=image_writer,
                        original=images[index],
                        adversarial=adv_image,
                        grid_path=grid_path,
                    )

                    original_one = original_tensor[index : index + 1]
                    adv_one = image_tensor[index : index + 1]
                    success = adv_eval.pred != record.class_index
                    clean_correct = clean_eval.pred == record.class_index
                    semantic_value = float(semantic_sim[index].detach().cpu().item())
                    dino_metric_value = (
                        float(dino_metric_sim[index].detach().cpu().item())
                        if dino_metric_sim is not None
                        else None
                    )
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
                        "learnable_token_init_seed": self.config.attack.learnable_token_init_seed,
                        "learnable_token_texts": " ".join(prompt_state.token_texts),
                        "lr": self.config.attack.lr,
                        "lr_scheduler": self.config.attack.lr_scheduler.name,
                        "lr_warmup_steps": self.config.attack.lr_scheduler.warmup_steps,
                        "lr_min": self.config.attack.lr_scheduler.min_lr,
                        "steps": self.config.attack.steps,
                        "lambda_sem": self.config.attack.lambda_sem,
                        "attack_loss_weight": attack_loss_weight,
                        "semantic_loss_weight": _logged_semantic_weight(
                            self.config,
                            semantic_loss_weight,
                        ),
                        "attack_margin": self.config.attack.attack_margin,
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
                        "clean_true_conf": clean_eval.true_conf,
                        "adv_true_conf": adv_eval.true_conf,
                        "confidence_drop": clean_eval.true_conf - adv_eval.true_conf,
                        "clean_margin": clean_eval.margin,
                        "adv_margin": adv_eval.margin,
                        "margin_drop": clean_eval.margin - adv_eval.margin,
                        **_semantic_metric_fields(
                            components.semantic,
                            semantic_value,
                            dino_value=dino_metric_value,
                        ),
                        "semantic_loss": float(semantic_losses[index].detach().cpu().item()),
                        "weighted_semantic_loss": float(
                            weighted_semantic_losses[index].detach().cpu().item()
                        ),
                        "ssim": ssim,
                        **pixel_metrics,
                        **nriqa_metrics,
                        "best_step": final_step,
                        "best_attack_step": final_step,
                        "first_success_step": final_step if success else -1,
                        "min_adv_true_conf": adv_eval.true_conf,
                        "min_adv_true_conf_step": final_step,
                        "min_adv_margin": adv_eval.margin,
                        "min_adv_margin_step": final_step,
                        "best_attack_loss": float(attack_losses[index].detach().cpu().item()),
                        "best_semantic_loss": float(semantic_losses[index].detach().cpu().item()),
                        "best_weighted_semantic_loss": float(
                            weighted_semantic_losses[index].detach().cpu().item()
                        ),
                        "best_total_loss": float(total_losses[index].detach().cpu().item()),
                        "runtime_seconds": runtime_seconds,
                        "output_image_path": str(adv_output_path),
                        "original_image_path": str(original_row_path),
                        "grid_image_path": str(grid_path),
                        "grid_saved": grid_saved,
                    }
                    if metrics_path is not None:
                        append_csv_row(metrics_path, row)
                    if logger is not None:
                        logger.log_image_result(
                            row=row, original=images[index], adversarial=adv_image
                        )
                    rows.append(row)
                    seen += 1
                    clean_correct_count += int(clean_correct)
                    success_count += int(success)
                    clean_success_count += int(clean_correct and success)
                if show_progress:
                    elapsed, eta = _progress_eta(
                        started_at=eval_started_at,
                        completed=batch_index + 1,
                        total=len(batches),
                    )
                    asr = success_count / max(seen, 1)
                    clean_asr = clean_success_count / max(clean_correct_count, 1)
                    set_process_title(
                        _process_title(
                            self.config,
                            f"{stage}-eval",
                            images=f"{seen}/{len(records)}",
                            asr=f"{asr:.2f}",
                            eta=_format_duration(eta),
                        )
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
                        f"asr={asr:.3f} | "
                        f"clean_asr={clean_asr:.3f} | "
                        f"elapsed={_format_duration(elapsed)} | "
                        f"eta={_format_duration(eta)}",
                    )
        self._save_representative_grids(rows)
        if (
            metrics_path is not None
            and self.config.output.grid_save_policy.lower() == "representative"
        ):
            write_csv_rows(metrics_path, rows)
        return rows

    def evaluate_universal_axis_prompt(
        self,
        records: list[ImageRecord],
        components: AttackComponents,
        axis_state: AxisPromptState,
        *,
        stage: str,
        metrics_path: Path | None = None,
        logger: WandbLogger | None = None,
        show_progress: bool = True,
    ) -> list[dict[str, Any]]:
        """Evaluate a frozen axis prompt by sweeping ``attack.eval_strengths``.

        For each test image, generates once per configured strength (same seed across
        strengths, deterministic) and records the earliest strength at which the attack
        both succeeds and stays within the SSIM (and optional semantic) legitimacy gate --
        the sweep-based analogue of MAELS' "earliest valid manifold crossing". The primary
        row fields (``success``, ``ssim``, image paths, ...) are taken from the ``t=1.0``
        sweep point specifically, so top-level ASR stays directly comparable to every prior
        single-strength run.
        """
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
        sorted_strengths = sorted(self.config.attack.eval_strengths)
        primary_index = (
            sorted_strengths.index(1.0)
            if 1.0 in sorted_strengths
            else len(sorted_strengths) - 1
        )
        progress = tqdm(
            batches,
            desc=f"{stage}-eval",
            disable=not show_progress,
            dynamic_ncols=True,
        )
        with self._image_writer() as image_writer:
            prefetched_batches = self._prefetched_record_batches(
                batches,
                enabled=True,
            )
            for batch_index, (batch, images, original_tensor, true_labels) in enumerate(
                prefetched_batches
            ):
                started_at = time.perf_counter()
                seeds = [stable_image_seed(0, record.image_id) for record in batch]
                with torch.no_grad():
                    clean_logits = components.victim.logits_from_tensor(original_tensor)
                    clean_evals = components.victim.evaluate_logits_batch(
                        clean_logits,
                        true_labels.detach().cpu().tolist(),
                    )

                sweep_results: list[dict[str, Any]] = []
                for t in sorted_strengths:
                    prompt_batch = build_axis_prompt_batch(axis_state, batch, t=t)
                    generated = components.generator.generate_batch(
                        input_images=images,
                        input_tensor=original_tensor,
                        prompt_state=prompt_batch,
                        seeds=seeds,
                        require_grad=False,
                    )
                    image_tensor = generated.image_tensor
                    if not isinstance(image_tensor, torch.Tensor):
                        raise TypeError(
                            "Generator batch result must expose torch.Tensor image_tensor."
                        )
                    with torch.no_grad():
                        adv_logits = components.victim.logits_from_tensor(image_tensor)
                        adv_evals = components.victim.evaluate_logits_batch(
                            adv_logits,
                            true_labels.detach().cpu().tolist(),
                        )
                        semantic_sim = components.semantic.similarity(
                            original_tensor, image_tensor, labels=true_labels
                        )
                        dino_metric_sim = (
                            components.dino_metric.similarity(original_tensor, image_tensor)
                            if components.dino_metric is not None
                            else None
                        )
                        attack_losses, semantic_losses, weighted_semantic_losses, total_losses = (
                            objective_loss_components(
                                adv_logits,
                                true_labels,
                                self.config.attack.objective,
                                semantic_similarity=semantic_sim,
                                lambda_sem=self.config.attack.lambda_sem,
                                semantic_loss_weight=self.config.attack.semantic_loss_weight,
                                attack_margin=self.config.attack.attack_margin,
                            )
                        )
                    ssim_values = [
                        global_ssim(original_tensor[index : index + 1], image_tensor[index : index + 1])
                        for index in range(len(batch))
                    ]
                    sweep_results.append(
                        {
                            "t": t,
                            "image_tensor": image_tensor,
                            "adv_evals": adv_evals,
                            "semantic_sim": semantic_sim,
                            "dino_metric_sim": dino_metric_sim,
                            "attack_losses": attack_losses,
                            "semantic_losses": semantic_losses,
                            "weighted_semantic_losses": weighted_semantic_losses,
                            "total_losses": total_losses,
                            "ssim_values": ssim_values,
                        }
                    )

                runtime_seconds = time.perf_counter() - started_at
                primary = sweep_results[primary_index]
                for index, record in enumerate(batch):
                    clean_eval = clean_evals[index]
                    legit_sweep_indices = []
                    for sweep_index, sweep in enumerate(sweep_results):
                        adv_eval_i = sweep["adv_evals"][index]
                        success_i = adv_eval_i.pred != record.class_index
                        ssim_i = sweep["ssim_values"][index]
                        semantic_ok = True
                        threshold = self.config.attack.legitimacy_semantic_threshold
                        if threshold is not None:
                            semantic_value_i = float(
                                sweep["semantic_sim"][index].detach().cpu().item()
                            )
                            semantic_ok = semantic_value_i >= threshold
                        if (
                            success_i
                            and ssim_i >= self.config.attack.legitimacy_ssim_threshold
                            and semantic_ok
                        ):
                            legit_sweep_indices.append(sweep_index)
                    sp_success = len(legit_sweep_indices) > 0
                    best_sweep_index = (
                        legit_sweep_indices[0]
                        if sp_success
                        else min(
                            range(len(sweep_results)),
                            key=lambda si: float(
                                sweep_results[si]["attack_losses"][index].detach().cpu().item()
                            ),
                        )
                    )
                    best_sweep = sweep_results[best_sweep_index]
                    sp_first_success_strength = (
                        sorted_strengths[legit_sweep_indices[0]] if sp_success else None
                    )
                    true_confs = [sweep["adv_evals"][index].true_conf for sweep in sweep_results]
                    margins = [sweep["adv_evals"][index].margin for sweep in sweep_results]
                    min_true_conf_index = min(
                        range(len(sweep_results)), key=lambda si: true_confs[si]
                    )
                    min_margin_index = min(range(len(sweep_results)), key=lambda si: margins[si])

                    adv_eval = primary["adv_evals"][index]
                    adv_image = tensor_to_pil(primary["image_tensor"][index : index + 1])
                    output_dir = (
                        self.config.output.root
                        / "images"
                        / stage
                        / record.class_label
                        / record.image_id
                    )
                    original_output_path, adv_output_path = self._save_eval_images(
                        writer=image_writer,
                        original=images[index],
                        adversarial=adv_image,
                        output_dir=output_dir,
                    )
                    original_row_path = (
                        original_output_path
                        if self.config.output.save_images
                        and self.config.output.save_original_images
                        else record.path
                    )
                    grid_path = self._grid_file_path(
                        stage=stage,
                        class_label=record.class_label,
                        image_id=record.image_id,
                    )
                    grid_saved = self._save_grid_if_all_policy(
                        writer=image_writer,
                        original=images[index],
                        adversarial=adv_image,
                        grid_path=grid_path,
                    )

                    original_one = original_tensor[index : index + 1]
                    adv_one = primary["image_tensor"][index : index + 1]
                    success = adv_eval.pred != record.class_index
                    clean_correct = clean_eval.pred == record.class_index
                    semantic_value = float(primary["semantic_sim"][index].detach().cpu().item())
                    dino_metric_value = (
                        float(primary["dino_metric_sim"][index].detach().cpu().item())
                        if primary["dino_metric_sim"] is not None
                        else None
                    )
                    pixel_metrics = pixel_distance_metrics(original_one, adv_one)
                    ssim = primary["ssim_values"][index]
                    nriqa_metrics = (
                        components.quality_evaluator.score_tensor(adv_one)
                        if components.quality_evaluator is not None
                        else {}
                    )
                    row = {
                        "stage": stage,
                        "training_mode": "universal",
                        "image_id": record.image_id,
                        "class_id": record.class_index,
                        "class_label": record.class_label,
                        "run_name": self.config.generator.name,
                        "seed": seeds[index],
                        "prompt_text": build_axis_prompt_batch(
                            axis_state, [record], t=1.0
                        ).prompt_texts[0],
                        "num_learnable_tokens": self.config.attack.num_learnable_tokens,
                        "num_anchor_tokens": axis_state.num_anchor_tokens,
                        "num_axis_tokens": axis_state.num_axis_tokens,
                        "strength_schedule": True,
                        "eval_strengths": ",".join(f"{value:g}" for value in sorted_strengths),
                        "legitimacy_ssim_threshold": self.config.attack.legitimacy_ssim_threshold,
                        "legitimacy_semantic_threshold": (
                            self.config.attack.legitimacy_semantic_threshold
                            if self.config.attack.legitimacy_semantic_threshold is not None
                            else ""
                        ),
                        "sp_success": sp_success,
                        "sp_first_success_strength": (
                            sp_first_success_strength if sp_first_success_strength is not None else ""
                        ),
                        "learnable_token_initializer": self.config.attack.learnable_token_initializer,
                        "learnable_token_init_std": self.config.attack.learnable_token_init_std,
                        "learnable_token_init_seed": self.config.attack.learnable_token_init_seed,
                        "learnable_token_texts": " ".join(axis_state.token_texts),
                        "lr": self.config.attack.lr,
                        "axis_lr": self.config.attack.axis_lr or self.config.attack.lr,
                        "lr_scheduler": self.config.attack.lr_scheduler.name,
                        "lr_warmup_steps": self.config.attack.lr_scheduler.warmup_steps,
                        "lr_min": self.config.attack.lr_scheduler.min_lr,
                        "steps": self.config.attack.steps,
                        "lambda_sem": self.config.attack.lambda_sem,
                        "attack_loss_weight": attack_loss_weight,
                        "semantic_loss_weight": _logged_semantic_weight(
                            self.config,
                            semantic_loss_weight,
                        ),
                        "attack_margin": self.config.attack.attack_margin,
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
                        "clean_true_conf": clean_eval.true_conf,
                        "adv_true_conf": adv_eval.true_conf,
                        "confidence_drop": clean_eval.true_conf - adv_eval.true_conf,
                        "clean_margin": clean_eval.margin,
                        "adv_margin": adv_eval.margin,
                        "margin_drop": clean_eval.margin - adv_eval.margin,
                        **_semantic_metric_fields(
                            components.semantic,
                            semantic_value,
                            dino_value=dino_metric_value,
                        ),
                        "semantic_loss": float(primary["semantic_losses"][index].detach().cpu().item()),
                        "weighted_semantic_loss": float(
                            primary["weighted_semantic_losses"][index].detach().cpu().item()
                        ),
                        "ssim": ssim,
                        **pixel_metrics,
                        **nriqa_metrics,
                        "best_step": best_sweep_index,
                        "best_attack_step": best_sweep_index,
                        "first_success_step": legit_sweep_indices[0] if sp_success else -1,
                        "min_adv_true_conf": true_confs[min_true_conf_index],
                        "min_adv_true_conf_step": min_true_conf_index,
                        "min_adv_margin": margins[min_margin_index],
                        "min_adv_margin_step": min_margin_index,
                        "best_attack_loss": float(
                            best_sweep["attack_losses"][index].detach().cpu().item()
                        ),
                        "best_semantic_loss": float(
                            best_sweep["semantic_losses"][index].detach().cpu().item()
                        ),
                        "best_weighted_semantic_loss": float(
                            best_sweep["weighted_semantic_losses"][index].detach().cpu().item()
                        ),
                        "best_total_loss": float(
                            best_sweep["total_losses"][index].detach().cpu().item()
                        ),
                        "runtime_seconds": runtime_seconds,
                        "output_image_path": str(adv_output_path),
                        "original_image_path": str(original_row_path),
                        "grid_image_path": str(grid_path),
                        "grid_saved": grid_saved,
                    }
                    if metrics_path is not None:
                        append_csv_row(metrics_path, row)
                    if logger is not None:
                        logger.log_image_result(
                            row=row, original=images[index], adversarial=adv_image
                        )
                    rows.append(row)
                    seen += 1
                    clean_correct_count += int(clean_correct)
                    success_count += int(success)
                    clean_success_count += int(clean_correct and success)
                if show_progress:
                    elapsed, eta = _progress_eta(
                        started_at=eval_started_at,
                        completed=batch_index + 1,
                        total=len(batches),
                    )
                    asr = success_count / max(seen, 1)
                    clean_asr = clean_success_count / max(clean_correct_count, 1)
                    set_process_title(
                        _process_title(
                            self.config,
                            f"{stage}-eval",
                            images=f"{seen}/{len(records)}",
                            asr=f"{asr:.2f}",
                            eta=_format_duration(eta),
                        )
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
                        f"asr={asr:.3f} | "
                        f"clean_asr={clean_asr:.3f} | "
                        f"elapsed={_format_duration(elapsed)} | "
                        f"eta={_format_duration(eta)}",
                    )
        self._save_representative_grids(rows)
        if (
            metrics_path is not None
            and self.config.output.grid_save_policy.lower() == "representative"
        ):
            write_csv_rows(metrics_path, rows)
        return rows

    def save_universal_prompt(
        self,
        prompt_state: LearnablePrompt | AxisPromptState,
        *,
        metadata: dict[str, Any],
    ) -> None:
        """Persist learned universal prompt weights and metadata."""
        import torch

        prompt_dir = self.config.output.root / "prompt"
        ensure_dir(prompt_dir)

        if isinstance(prompt_state, AxisPromptState):
            flat_embeddings = embeddings_at(prompt_state, t=1.0)
            torch.save(
                {
                    "format_version": 2,
                    "token_texts": prompt_state.token_texts,
                    "token_ids": prompt_state.token_ids,
                    "num_anchor_tokens": prompt_state.num_anchor_tokens,
                    "num_axis_tokens": prompt_state.num_axis_tokens,
                    "anchor_embeddings": prompt_state.anchor_embeddings.detach().cpu(),
                    "axis_base": prompt_state.axis_base.detach().cpu(),
                    "axis_direction": prompt_state.axis_direction.detach().cpu(),
                    "learnable_embeddings": flat_embeddings.detach().cpu(),
                    "metadata": metadata,
                },
                prompt_dir / "learned_prompt.pt",
            )
            write_json(
                prompt_dir / "metadata.json",
                {
                    **metadata,
                    "format_version": 2,
                    "token_texts": list(prompt_state.token_texts),
                    "token_ids": list(prompt_state.token_ids),
                    "num_anchor_tokens": prompt_state.num_anchor_tokens,
                    "num_axis_tokens": prompt_state.num_axis_tokens,
                    "embedding_shape": list(flat_embeddings.shape),
                },
            )
            return

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
        if self.config.attack.strength_schedule:
            return self._run_universal_axis(max_images=max_images)

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
                    "learnable_token_initializer": self.config.attack.learnable_token_initializer,
                    "learnable_token_init_seed": self.config.attack.learnable_token_init_seed,
                    "semantic_model": self.config.semantic.name,
                    "semantic_loss_weight": self.config.attack.semantic_loss_weight,
                    "attack_margin": self.config.attack.attack_margin,
                    "lr": self.config.attack.lr,
                },
            )
            logger.log_summary(summary)
            return rows
        finally:
            logger.finish()

    def _run_universal_axis(self, *, max_images: int | None = None) -> list[dict[str, Any]]:
        """Train an anchor/axis prompt and evaluate it with a strength sweep."""
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
            axis_state, history = self.train_universal_axis_prompt(
                records,
                components,
                logger=logger,
                history_path=history_path,
            )
            rows = self.evaluate_universal_axis_prompt(
                records,
                components,
                axis_state,
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
                axis_state,
                metadata={
                    "training_mode": "universal",
                    "split": self.config.data.split,
                    "record_count": len(records),
                    "steps": self.config.attack.steps,
                    "attack_batch_size": self.config.attack.batch_size,
                    "generator_batch_size": self.config.generator.batch_size,
                    "objective": self.config.attack.objective,
                    "lambda_sem": self.config.attack.lambda_sem,
                    "learnable_token_initializer": self.config.attack.learnable_token_initializer,
                    "learnable_token_init_seed": self.config.attack.learnable_token_init_seed,
                    "semantic_model": self.config.semantic.name,
                    "semantic_loss_weight": self.config.attack.semantic_loss_weight,
                    "attack_margin": self.config.attack.attack_margin,
                    "lr": self.config.attack.lr,
                    "axis_lr": self.config.attack.axis_lr or self.config.attack.lr,
                    "train_strengths": list(self.config.attack.train_strengths),
                    "eval_strengths": list(self.config.attack.eval_strengths),
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
            init_seed=self.config.attack.learnable_token_init_seed,
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
        min_true_conf: dict[str, Any] | None = None
        first_success_step = -1

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
            semantic_sim = None
            semantic_loss_value = None
            weighted_semantic_loss = None
            if semantic_loss_weight > 0:
                semantic_sim = semantic.similarity(original_tensor, generated.image_tensor)
            attack_losses, semantic_losses, weighted_semantic_losses, total_losses = (
                objective_loss_components(
                    logits,
                    record.class_index,
                    self.config.attack.objective,
                    semantic_similarity=semantic_sim,
                    lambda_sem=self.config.attack.lambda_sem,
                    semantic_loss_weight=self.config.attack.semantic_loss_weight,
                    attack_margin=self.config.attack.attack_margin,
                )
            )
            attack_loss = attack_losses.mean()
            semantic_loss_value = semantic_losses.mean()
            weighted_semantic_loss = weighted_semantic_losses.mean()
            total_loss = total_losses.mean()
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"Non-finite loss for {record.image_id} at step {step}")
            total_loss.backward()
            optimizer.step()
            generator.sync_learnable_prompt(prompt_state)
            if lr_scheduler is not None:
                lr_scheduler.step()

            eval_result = victim.evaluate_logits(logits.detach(), record.class_index)
            success = eval_result.pred != record.class_index
            if semantic_sim is None:
                with torch.no_grad():
                    semantic_sim = semantic.similarity(
                        original_tensor,
                        generated.image_tensor.detach(),
                    )
                    semantic_loss_value = 1.0 - semantic_sim.mean()
                    weighted_semantic_loss = semantic_loss_weight * semantic_loss_value
            semantic_value = float(semantic_sim.detach().cpu().item())
            current = {
                "step": step,
                "image_tensor": generated.image_tensor.detach(),
                "attack_loss": float(attack_loss.detach().cpu().item()),
                "semantic_loss": float(semantic_loss_value.detach().cpu().item()),
                "weighted_semantic_loss": float(weighted_semantic_loss.detach().cpu().item()),
                "total_loss": float(total_loss.detach().cpu().item()),
                "adv_pred": eval_result.pred,
                "adv_top1_conf": eval_result.pred_conf,
                "adv_true_conf": eval_result.true_conf,
                "adv_margin": eval_result.margin,
                "semantic_similarity": semantic_value,
                "success": success,
            }
            if success and first_success_step < 0:
                first_success_step = step
            if logger is not None:
                logger.log_step(
                    image_index=image_index,
                    image_id=record.image_id,
                    class_label=record.class_label,
                    attack_step=step,
                    values={
                        "attack_loss": current["attack_loss"],
                        "attack_objective_loss": current["attack_loss"],
                        "semantic_loss": current["semantic_loss"],
                        "attack_loss_weight": attack_loss_weight,
                        "semantic_loss_weight": _logged_semantic_weight(
                            self.config,
                            semantic_loss_weight,
                        ),
                        "total_loss": current["total_loss"],
                        "lr": current_lr,
                        "true_conf": current["adv_true_conf"],
                        "confidence_drop": clean_eval.true_conf - current["adv_true_conf"],
                        "top1_conf": current["adv_top1_conf"],
                        "logit_gap_true_vs_other": current["adv_margin"],
                        "semantic_similarity": current["semantic_similarity"],
                        "success": int(success),
                    },
                )
            if best is None or current["total_loss"] < best["total_loss"]:
                best = current
            if best_attack is None or current["adv_margin"] < best_attack["adv_margin"]:
                best_attack = current
            if min_true_conf is None or current["adv_true_conf"] < min_true_conf["adv_true_conf"]:
                min_true_conf = current

        if best is None or best_attack is None or min_true_conf is None:
            raise RuntimeError(f"No optimization step ran for {record.image_id}")

        adv_image = tensor_to_pil(best["image_tensor"])
        output_dir = self.config.output.root / "images" / record.class_label / record.image_id
        with self._image_writer() as image_writer:
            original_output_path, adv_output_path = self._save_eval_images(
                writer=image_writer,
                original=image,
                adversarial=adv_image,
                output_dir=output_dir,
            )
            grid_path = self._grid_file_path(
                stage=self.config.data.split or "data",
                class_label=record.class_label,
                image_id=record.image_id,
            )
            grid_saved = self._save_grid_if_all_policy(
                writer=image_writer,
                original=image,
                adversarial=adv_image,
                grid_path=grid_path,
            )
        original_row_path = (
            original_output_path
            if self.config.output.save_images and self.config.output.save_original_images
            else record.path
        )

        success = int(best["adv_pred"]) != record.class_index
        confidence_drop = clean_eval.true_conf - float(best["adv_true_conf"])
        dino_metric_value = None
        if components.dino_metric is not None:
            with torch.no_grad():
                dino_metric_sim = components.dino_metric.similarity(
                    original_tensor,
                    best["image_tensor"],
                )
                dino_metric_value = float(dino_metric_sim.detach().cpu().item())
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
            "learnable_token_init_seed": self.config.attack.learnable_token_init_seed,
            "learnable_token_texts": " ".join(prompt_state.token_texts),
            "lr": self.config.attack.lr,
            "lr_scheduler": self.config.attack.lr_scheduler.name,
            "lr_warmup_steps": self.config.attack.lr_scheduler.warmup_steps,
            "lr_min": self.config.attack.lr_scheduler.min_lr,
            "steps": self.config.attack.steps,
            "lambda_sem": self.config.attack.lambda_sem,
            "attack_loss_weight": attack_loss_weight,
            "semantic_loss_weight": _logged_semantic_weight(self.config, semantic_loss_weight),
            "attack_margin": self.config.attack.attack_margin,
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
            "clean_true_conf": clean_eval.true_conf,
            "adv_true_conf": float(best["adv_true_conf"]),
            "confidence_drop": confidence_drop,
            "clean_margin": clean_eval.margin,
            "adv_margin": float(best["adv_margin"]),
            "margin_drop": clean_eval.margin - float(best["adv_margin"]),
            **_semantic_metric_fields(
                semantic,
                float(best["semantic_similarity"]),
                dino_value=dino_metric_value,
            ),
            "semantic_loss": float(best["semantic_loss"]),
            "weighted_semantic_loss": float(best["weighted_semantic_loss"]),
            "ssim": ssim,
            **pixel_metrics,
            **nriqa_metrics,
            "best_step": int(best["step"]),
            "best_attack_step": int(best_attack["step"]),
            "first_success_step": first_success_step,
            "min_adv_true_conf": float(min_true_conf["adv_true_conf"]),
            "min_adv_true_conf_step": int(min_true_conf["step"]),
            "min_adv_margin": float(best_attack["adv_margin"]),
            "min_adv_margin_step": int(best_attack["step"]),
            "best_attack_loss": float(best["attack_loss"]),
            "best_semantic_loss": float(best["semantic_loss"]),
            "best_weighted_semantic_loss": float(best["weighted_semantic_loss"]),
            "best_total_loss": float(best["total_loss"]),
            "runtime_seconds": runtime_seconds,
            "output_image_path": str(adv_output_path),
            "original_image_path": str(original_row_path),
            "grid_image_path": str(grid_path),
            "grid_saved": grid_saved,
        }
        if self.config.output.grid_save_policy.lower() == "representative":
            self._save_representative_grids([row])
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
        clean_evals = victim.evaluate_logits_batch(
            clean_logits.detach(), true_labels.detach().cpu().tolist()
        )
        prompt_state = generator.create_learnable_prompt_batch(
            class_labels=[record.class_label for record in records],
            num_tokens=self.config.attack.num_learnable_tokens,
            initializer=self.config.attack.learnable_token_initializer,
            init_std=self.config.attack.learnable_token_init_std,
            init_seed=self.config.attack.learnable_token_init_seed,
        )
        learnable_embeddings = prompt_state.learnable_embeddings
        if not isinstance(learnable_embeddings, torch.Tensor):
            raise TypeError(
                "Generator batch prompt state must expose torch.Tensor learnable embeddings."
            )
        if learnable_embeddings.ndim != 3:
            raise ValueError("Batch learnable-token parameter must have shape [B, N, D].")
        if learnable_embeddings.shape[0] != len(records):
            raise ValueError(
                "Batch learnable-token parameter batch dimension does not match records."
            )
        if not learnable_embeddings.requires_grad:
            raise RuntimeError("Batch learnable-token parameter must require gradients.")
        optimizer = torch.optim.Adam([learnable_embeddings], lr=self.config.attack.lr)
        lr_scheduler = self._build_lr_scheduler(optimizer)
        seeds = [stable_image_seed(0, record.image_id) for record in records]
        best: list[dict[str, Any] | None] = [None] * len(records)
        best_attack: list[dict[str, Any] | None] = [None] * len(records)
        min_true_conf: list[dict[str, Any] | None] = [None] * len(records)
        first_success_step = [-1] * len(records)

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
            semantic_sim = None
            if semantic_loss_weight > 0:
                semantic_sim = semantic.similarity(original_tensor, image_tensor)
            attack_losses, semantic_losses, weighted_semantic_losses, total_losses = (
                objective_loss_components(
                    logits,
                    true_labels,
                    self.config.attack.objective,
                    semantic_similarity=semantic_sim,
                    lambda_sem=self.config.attack.lambda_sem,
                    semantic_loss_weight=self.config.attack.semantic_loss_weight,
                    attack_margin=self.config.attack.attack_margin,
                )
            )
            total_loss = total_losses.mean()
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"Non-finite batch loss at step {step}")
            total_loss.backward()
            optimizer.step()
            generator.sync_learnable_prompt_batch(prompt_state)
            if lr_scheduler is not None:
                lr_scheduler.step()

            eval_results = victim.evaluate_logits_batch(
                logits.detach(), true_labels.detach().cpu().tolist()
            )
            if semantic_sim is None:
                with torch.no_grad():
                    semantic_sim = semantic.similarity(original_tensor, image_tensor.detach())
                    semantic_losses = 1.0 - semantic_sim
                    weighted_semantic_losses = semantic_loss_weight * semantic_losses
            for index, record in enumerate(records):
                eval_result = eval_results[index]
                success = eval_result.pred != record.class_index
                semantic_value = float(semantic_sim[index].detach().cpu().item())
                current = {
                    "step": step,
                    "image_tensor": image_tensor[index : index + 1].detach(),
                    "attack_loss": float(attack_losses[index].detach().cpu().item()),
                    "semantic_loss": float(semantic_losses[index].detach().cpu().item()),
                    "weighted_semantic_loss": float(
                        weighted_semantic_losses[index].detach().cpu().item()
                    ),
                    "total_loss": float(total_losses[index].detach().cpu().item()),
                    "adv_pred": eval_result.pred,
                    "adv_top1_conf": eval_result.pred_conf,
                    "adv_true_conf": eval_result.true_conf,
                    "adv_margin": eval_result.margin,
                    "semantic_similarity": semantic_value,
                    "success": success,
                }
                if success and first_success_step[index] < 0:
                    first_success_step[index] = step
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
                            "semantic_loss": current["semantic_loss"],
                            "attack_loss_weight": attack_loss_weight,
                            "semantic_loss_weight": _logged_semantic_weight(
                                self.config,
                                semantic_loss_weight,
                            ),
                            "total_loss": current["total_loss"],
                            "lr": current_lr,
                            "true_conf": current["adv_true_conf"],
                            "confidence_drop": clean_eval.true_conf - current["adv_true_conf"],
                            "top1_conf": current["adv_top1_conf"],
                            "logit_gap_true_vs_other": current["adv_margin"],
                            "semantic_similarity": current["semantic_similarity"],
                            "success": int(success),
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

        rows: list[dict[str, Any]] = []
        runtime_seconds = time.perf_counter() - started_at
        dino_metric_sim = None
        if components.dino_metric is not None:
            with torch.no_grad():
                best_tensors = torch.cat(
                    [best_row["image_tensor"] for best_row in best if best_row is not None],
                    dim=0,
                )
                dino_metric_sim = components.dino_metric.similarity(original_tensor, best_tensors)
        with self._image_writer() as image_writer:
            for index, record in enumerate(records):
                best_row = best[index]
                best_attack_row = best_attack[index]
                min_true_conf_row = min_true_conf[index]
                if best_row is None or best_attack_row is None or min_true_conf_row is None:
                    raise RuntimeError(f"No optimization step ran for {record.image_id}")
                adv_image = tensor_to_pil(best_row["image_tensor"])
                output_dir = (
                    self.config.output.root / "images" / record.class_label / record.image_id
                )
                original_output_path, adv_output_path = self._save_eval_images(
                    writer=image_writer,
                    original=images[index],
                    adversarial=adv_image,
                    output_dir=output_dir,
                )
                original_row_path = (
                    original_output_path
                    if self.config.output.save_images and self.config.output.save_original_images
                    else record.path
                )
                grid_path = self._grid_file_path(
                    stage=self.config.data.split or "data",
                    class_label=record.class_label,
                    image_id=record.image_id,
                )
                grid_saved = self._save_grid_if_all_policy(
                    writer=image_writer,
                    original=images[index],
                    adversarial=adv_image,
                    grid_path=grid_path,
                )

                clean_eval = clean_evals[index]
                success = int(best_row["adv_pred"]) != record.class_index
                confidence_drop = clean_eval.true_conf - float(best_row["adv_true_conf"])
                dino_metric_value = (
                    float(dino_metric_sim[index].detach().cpu().item())
                    if dino_metric_sim is not None
                    else None
                )
                original_one = original_tensor[index : index + 1]
                pixel_metrics = pixel_distance_metrics(original_one, best_row["image_tensor"])
                ssim = global_ssim(original_one, best_row["image_tensor"])
                nriqa_metrics = (
                    quality_evaluator.score_tensor(best_row["image_tensor"])
                    if quality_evaluator is not None
                    else {}
                )
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
                    "learnable_token_init_seed": self.config.attack.learnable_token_init_seed,
                    "learnable_token_texts": " ".join(prompt_state.token_texts),
                    "lr": self.config.attack.lr,
                    "lr_scheduler": self.config.attack.lr_scheduler.name,
                    "lr_warmup_steps": self.config.attack.lr_scheduler.warmup_steps,
                    "lr_min": self.config.attack.lr_scheduler.min_lr,
                    "steps": self.config.attack.steps,
                    "lambda_sem": self.config.attack.lambda_sem,
                    "attack_loss_weight": attack_loss_weight,
                    "semantic_loss_weight": _logged_semantic_weight(
                        self.config, semantic_loss_weight
                    ),
                    "attack_margin": self.config.attack.attack_margin,
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
                    "clean_true_conf": clean_eval.true_conf,
                    "adv_true_conf": float(best_row["adv_true_conf"]),
                    "confidence_drop": confidence_drop,
                    "clean_margin": clean_eval.margin,
                    "adv_margin": float(best_row["adv_margin"]),
                    "margin_drop": clean_eval.margin - float(best_row["adv_margin"]),
                    **_semantic_metric_fields(
                        semantic,
                        float(best_row["semantic_similarity"]),
                        dino_value=dino_metric_value,
                    ),
                    "semantic_loss": float(best_row["semantic_loss"]),
                    "weighted_semantic_loss": float(best_row["weighted_semantic_loss"]),
                    "ssim": ssim,
                    **pixel_metrics,
                    **nriqa_metrics,
                    "best_step": int(best_row["step"]),
                    "best_attack_step": int(best_attack_row["step"]),
                    "first_success_step": first_success_step[index],
                    "min_adv_true_conf": float(min_true_conf_row["adv_true_conf"]),
                    "min_adv_true_conf_step": int(min_true_conf_row["step"]),
                    "min_adv_margin": float(best_attack_row["adv_margin"]),
                    "min_adv_margin_step": int(best_attack_row["step"]),
                    "best_attack_loss": float(best_row["attack_loss"]),
                    "best_semantic_loss": float(best_row["semantic_loss"]),
                    "best_weighted_semantic_loss": float(best_row["weighted_semantic_loss"]),
                    "best_total_loss": float(best_row["total_loss"]),
                    "runtime_seconds": runtime_seconds,
                    "output_image_path": str(adv_output_path),
                    "original_image_path": str(original_row_path),
                    "grid_image_path": str(grid_path),
                    "grid_saved": grid_saved,
                }
                if logger is not None:
                    logger.log_image_result(row=row, original=images[index], adversarial=adv_image)
                rows.append(row)
        if self.config.output.grid_save_policy.lower() == "representative":
            self._save_representative_grids(rows)
        return rows
