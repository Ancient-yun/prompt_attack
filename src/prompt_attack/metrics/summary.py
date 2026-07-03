"""Result summarization."""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True)
class MetricSummary:
    count: int
    success_count: int
    asr: float
    mean_semantic_similarity: float
    mean_dino_similarity: float
    mean_clip_image_similarity: float | None
    mean_ssim: float
    mean_decision_logit_gap_drop: float
    mean_confidence_drop: float
    mean_pixel_l1_mean: float
    mean_pixel_l2: float
    mean_pixel_l2_mean: float
    mean_pixel_linf: float
    mean_iqa_nima_ava: float | None
    mean_iqa_hyperiqa: float | None
    mean_iqa_musiq_ava: float | None
    mean_iqa_musiq_koniq: float | None
    mean_iqa_tres: float | None
    fid: float | None
    mean_runtime_seconds: float
    sp_asr: float | None = None
    mean_first_success_strength: float | None = None
    median_first_success_strength: float | None = None


def _mean_optional(rows: Sequence[Mapping[str, object]], key: str) -> float | None:
    values = [row.get(key) for row in rows]
    numeric = [float(cast(Any, value)) for value in values if value not in {None, ""}]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def summarize_rows(rows: Sequence[Mapping[str, object]], *, fid: float | None = None) -> MetricSummary:
    """Summarize attack rows."""
    if not rows:
        return MetricSummary(
            count=0,
            success_count=0,
            asr=0.0,
            mean_semantic_similarity=0.0,
            mean_dino_similarity=0.0,
            mean_clip_image_similarity=None,
            mean_ssim=0.0,
            mean_decision_logit_gap_drop=0.0,
            mean_confidence_drop=0.0,
            mean_pixel_l1_mean=0.0,
            mean_pixel_l2=0.0,
            mean_pixel_l2_mean=0.0,
            mean_pixel_linf=0.0,
            mean_iqa_nima_ava=None,
            mean_iqa_hyperiqa=None,
            mean_iqa_musiq_ava=None,
            mean_iqa_musiq_koniq=None,
            mean_iqa_tres=None,
            fid=fid,
            mean_runtime_seconds=0.0,
        )
    success_count = sum(1 for row in rows if _as_bool(row["success"]))
    semantic = [float(cast(Any, row["semantic_similarity"])) for row in rows]
    dino_values = [row.get("dino_similarity") for row in rows]
    dino = [float(cast(Any, value)) for value in dino_values if value not in {None, ""}]
    clip = _mean_optional(rows, "clip_image_similarity")
    ssim = [float(cast(Any, row["ssim"])) for row in rows]
    margin = [float(cast(Any, row["margin_drop"])) for row in rows]
    confidence = [float(cast(Any, row["confidence_drop"])) for row in rows]
    pixel_l1 = [float(cast(Any, row["pixel_l1_mean"])) for row in rows]
    pixel_l2 = [float(cast(Any, row["pixel_l2"])) for row in rows]
    pixel_l2_mean = [float(cast(Any, row["pixel_l2_mean"])) for row in rows]
    pixel_linf = [float(cast(Any, row["pixel_linf"])) for row in rows]
    runtime = [float(cast(Any, row["runtime_seconds"])) for row in rows]
    sp_rows = [row for row in rows if "sp_success" in row]
    sp_asr = (
        sum(1 for row in sp_rows if _as_bool(row["sp_success"])) / len(sp_rows)
        if sp_rows
        else None
    )
    first_success_strengths = sorted(
        float(cast(Any, row["sp_first_success_strength"]))
        for row in sp_rows
        if _as_bool(row.get("sp_success", False))
        and row.get("sp_first_success_strength") not in {None, ""}
    )
    mean_first_success_strength = (
        sum(first_success_strengths) / len(first_success_strengths)
        if first_success_strengths
        else None
    )
    median_first_success_strength = (
        statistics.median(first_success_strengths) if first_success_strengths else None
    )
    return MetricSummary(
        count=len(rows),
        success_count=success_count,
        asr=success_count / len(rows),
        mean_semantic_similarity=sum(semantic) / len(semantic),
        mean_dino_similarity=sum(dino) / len(dino) if dino else 0.0,
        mean_clip_image_similarity=clip,
        mean_ssim=sum(ssim) / len(ssim),
        mean_decision_logit_gap_drop=sum(margin) / len(margin),
        mean_confidence_drop=sum(confidence) / len(confidence),
        mean_pixel_l1_mean=sum(pixel_l1) / len(pixel_l1),
        mean_pixel_l2=sum(pixel_l2) / len(pixel_l2),
        mean_pixel_l2_mean=sum(pixel_l2_mean) / len(pixel_l2_mean),
        mean_pixel_linf=sum(pixel_linf) / len(pixel_linf),
        mean_iqa_nima_ava=_mean_optional(rows, "iqa_nima_ava"),
        mean_iqa_hyperiqa=_mean_optional(rows, "iqa_hyperiqa"),
        mean_iqa_musiq_ava=_mean_optional(rows, "iqa_musiq_ava"),
        mean_iqa_musiq_koniq=_mean_optional(rows, "iqa_musiq_koniq"),
        mean_iqa_tres=_mean_optional(rows, "iqa_tres"),
        fid=fid,
        mean_runtime_seconds=sum(runtime) / len(runtime),
        sp_asr=sp_asr,
        mean_first_success_strength=mean_first_success_strength,
        median_first_success_strength=median_first_success_strength,
    )
