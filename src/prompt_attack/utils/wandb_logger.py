"""Weights & Biases logging helpers."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Literal, cast

from PIL import Image

from prompt_attack.config import ExperimentConfig

WandbMode = Literal["online", "offline", "disabled", "shared"]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class WandbLogger:
    """Small wrapper so the attack loop does not depend directly on wandb."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._run: Any | None = None
        self._wandb: Any | None = None
        self._global_step = 0
        self._image_result_rows: list[list[Any]] = []
        self._defined_image_metrics: set[str] = set()

    @property
    def enabled(self) -> bool:
        return self.config.logging.wandb.enabled

    def start(self) -> None:
        if not self.enabled:
            return

        import wandb

        wandb_config = self.config.logging.wandb
        run_dir = wandb_config.dir
        run_dir.mkdir(parents=True, exist_ok=True)
        mode_raw = os.environ.get("WANDB_MODE") or wandb_config.mode
        if mode_raw not in {"online", "offline", "disabled", "shared"}:
            raise ValueError(
                "logging.wandb.mode or WANDB_MODE must be one of "
                "online, offline, disabled, shared."
            )
        mode = cast(WandbMode, mode_raw)
        project = os.environ.get("WANDB_PROJECT") or wandb_config.project
        entity = os.environ.get("WANDB_ENTITY") or wandb_config.entity
        name = wandb_config.name or os.environ.get("WANDB_NAME")

        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            mode=mode,
            dir=str(run_dir),
            tags=list(wandb_config.tags),
            config=_jsonable(self.config),
        )
        self._run.define_metric("progress/global_step")
        self._run.define_metric("image_result/*", step_metric="image_result/index")
        self._run.define_metric("summary/*")

    def _image_metric_key(self, *, image_index: int, image_id: str, class_label: str) -> str:
        label = re.sub(r"[^A-Za-z0-9]+", "_", class_label).strip("_").lower()
        return f"image{image_index:03d}_{label}_{image_id[:8]}"

    def _define_image_train_metrics(self, image_key: str) -> None:
        if self._run is None or image_key in self._defined_image_metrics:
            return
        step_metric = f"image_train/{image_key}/step"
        self._run.define_metric(step_metric)
        self._run.define_metric(f"image_train/{image_key}/*", step_metric=step_metric)
        self._defined_image_metrics.add(image_key)

    def log_step(
        self,
        *,
        image_index: int,
        image_id: str,
        class_label: str,
        attack_step: int,
        values: dict[str, float | int | bool],
    ) -> None:
        if self._run is None:
            self._global_step += 1
            return
        if attack_step % self.config.logging.wandb.log_every_steps != 0:
            self._global_step += 1
            return
        image_key = self._image_metric_key(
            image_index=image_index,
            image_id=image_id,
            class_label=class_label,
        )
        self._define_image_train_metrics(image_key)
        payload: dict[str, Any] = {
            "progress/global_step": self._global_step,
            "progress/image_index": image_index,
            "image/id": image_id,
            "image/class_label": class_label,
            f"image_train/{image_key}/step": attack_step,
        }
        payload.update({f"image_train/{image_key}/{key}": value for key, value in values.items()})
        self._run.log(payload, step=self._global_step)
        self._global_step += 1

    def _optional_float(self, row: dict[str, Any], key: str) -> float | None:
        value = row.get(key)
        if value is None or value == "":
            return None
        return float(value)

    def log_image_result(
        self,
        *,
        row: dict[str, Any],
        original: Image.Image,
        adversarial: Image.Image,
    ) -> None:
        if self._run is None:
            return
        image_result_index = len(self._image_result_rows)
        payload: dict[str, Any] = {
            "image_result/index": image_result_index,
            "image_result/success": int(bool(row["success"])),
            "image_result/semantic_constrained_success": int(
                bool(row["semantic_constrained_success"])
            ),
            "image_result/clean_true_conf": float(row["clean_true_conf"]),
            "image_result/adv_true_conf": float(row["adv_true_conf"]),
            "image_result/confidence_drop": float(row["confidence_drop"]),
            "image_result/clean_top1_conf": float(row["clean_top1_conf"]),
            "image_result/adv_top1_conf": float(row["adv_top1_conf"]),
            "image_result/clean_logit_gap_true_vs_other": float(row["clean_margin"]),
            "image_result/adv_logit_gap_true_vs_other": float(row["adv_margin"]),
            "image_result/logit_gap_drop": float(row["margin_drop"]),
            "image_result/dino_similarity": float(row["dino_similarity"]),
            "image_result/ssim": float(row["ssim"]),
            "image_result/pixel_l1_mean": float(row["pixel_l1_mean"]),
            "image_result/pixel_l2": float(row["pixel_l2"]),
            "image_result/pixel_l2_mean": float(row["pixel_l2_mean"]),
            "image_result/pixel_linf": float(row["pixel_linf"]),
            "image_result/best_step": int(row["best_step"]),
            "image_result/best_attack_step": int(row["best_attack_step"]),
            "image_result/best_semantic_success_step": int(row["best_semantic_success_step"]),
            "image_result/first_success_step": int(row["first_success_step"]),
            "image_result/first_semantic_success_step": int(row["first_semantic_success_step"]),
            "image_result/min_adv_true_conf": float(row["min_adv_true_conf"]),
            "image_result/min_adv_logit_gap_true_vs_other": float(row["min_adv_margin"]),
            "image_result/runtime_seconds": float(row["runtime_seconds"]),
            "image_result/class_label": row["class_label"],
            "image_result/clean_pred_label": row["clean_pred_label"],
            "image_result/adv_pred_label": row["adv_pred_label"],
            "image_result/image_id": row["image_id"],
            "image_result/objective": row["objective"],
        }
        for key in ("nima_ava", "hyperiqa", "musiq_ava", "musiq_koniq", "tres"):
            value = self._optional_float(row, f"iqa_{key}")
            if value is not None:
                payload[f"image_result/iqa_{key}"] = value
        if self.config.logging.wandb.log_images and self._wandb is not None:
            original_caption = (
                f"{row['class_label']} | clean={row['clean_pred_label']} "
                f"p={float(row['clean_true_conf']):.3f}"
            )
            adversarial_caption = (
                f"adv={row['adv_pred_label']} | success={row['success']} | "
                f"sem_success={row['semantic_constrained_success']} | "
                f"dino={float(row['dino_similarity']):.3f}"
            )
            payload["image_result/original"] = self._wandb.Image(
                original,
                caption=original_caption,
            )
            payload["image_result/adversarial"] = self._wandb.Image(
                adversarial,
                caption=adversarial_caption,
            )
            table_original = self._wandb.Image(original, caption=original_caption)
            table_adversarial = self._wandb.Image(adversarial, caption=adversarial_caption)
        else:
            table_original = str(row["original_image_path"])
            table_adversarial = str(row["output_image_path"])
        self._image_result_rows.append(
            [
                row["image_id"],
                row["class_label"],
                row["clean_pred_label"],
                row["adv_pred_label"],
                bool(row["success"]),
                bool(row["semantic_constrained_success"]),
                float(row["clean_true_conf"]),
                float(row["adv_true_conf"]),
                float(row["confidence_drop"]),
                float(row["margin_drop"]),
                float(row["dino_similarity"]),
                float(row["ssim"]),
                self._optional_float(row, "iqa_nima_ava"),
                self._optional_float(row, "iqa_hyperiqa"),
                self._optional_float(row, "iqa_musiq_ava"),
                self._optional_float(row, "iqa_musiq_koniq"),
                self._optional_float(row, "iqa_tres"),
                int(row["best_step"]),
                int(row["first_success_step"]),
                float(row["runtime_seconds"]),
                table_original,
                table_adversarial,
            ]
        )
        if self._wandb is not None:
            payload["image_results/table"] = self._wandb.Table(
                columns=[
                    "image_id",
                    "true_label",
                    "clean_pred",
                    "adv_pred",
                    "success",
                    "semantic_success",
                    "clean_true_conf",
                    "adv_true_conf",
                    "confidence_drop",
                    "logit_gap_drop",
                    "dino_similarity",
                    "ssim",
                    "NIMA-AVA",
                    "HyperIQA",
                    "MUSIQ-AVA",
                    "MUSIQ-KonIQ",
                    "TReS",
                    "best_step",
                    "first_success_step",
                    "runtime_seconds",
                    "original",
                    "adversarial",
                ],
                data=self._image_result_rows,
            )
        self._run.log(payload, step=self._global_step)

    def log_summary(self, summary: dict[str, Any]) -> None:
        if self._run is None:
            return
        for key, value in summary.items():
            self._run.summary[key] = value
        self._run.log(
            {f"summary/{key}": value for key, value in summary.items() if value is not None}
        )

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
        self._run = None
        self._wandb = None
        self._image_result_rows = []
        self._defined_image_metrics = set()
