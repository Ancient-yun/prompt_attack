"""Classifier-independent image-collapse evaluation utilities."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class CollapseThresholds:
    """Semantic-preservation thresholds calibrated from benign edits."""

    dino_similarity: float | None = None
    clip_image_similarity: float | None = None
    dreamsim_distance: float | None = None
    source: str = "calibrated"

    def active_metrics(self) -> tuple[str, ...]:
        """Return metrics with configured thresholds."""
        metrics: list[str] = []
        if self.dino_similarity is not None:
            metrics.append("dino")
        if self.clip_image_similarity is not None:
            metrics.append("clip")
        if self.dreamsim_distance is not None:
            metrics.append("dreamsim")
        return tuple(metrics)

    def with_overrides(
        self,
        *,
        dino_similarity: float | None = None,
        clip_image_similarity: float | None = None,
        dreamsim_distance: float | None = None,
        source: str | None = None,
    ) -> "CollapseThresholds":
        """Return a copy with explicit threshold overrides applied."""
        return replace(
            self,
            dino_similarity=(
                self.dino_similarity if dino_similarity is None else float(dino_similarity)
            ),
            clip_image_similarity=(
                self.clip_image_similarity
                if clip_image_similarity is None
                else float(clip_image_similarity)
            ),
            dreamsim_distance=(
                self.dreamsim_distance if dreamsim_distance is None else float(dreamsim_distance)
            ),
            source=self.source if source is None else source,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return JSON/CSV-friendly threshold fields."""
        return {
            "dino_similarity_threshold": self.dino_similarity,
            "clip_image_similarity_threshold": self.clip_image_similarity,
            "dreamsim_distance_threshold": self.dreamsim_distance,
            "threshold_source": self.source,
            "active_metrics": ",".join(self.active_metrics()),
        }


def _is_missing(value: object) -> bool:
    return value is None or value == ""


def _as_float(value: object) -> float | None:
    if _is_missing(value):
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not isfinite(result):
        return None
    return result


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _semantic_metric(row: Mapping[str, object]) -> str:
    return str(row.get("semantic_metric", "")).strip().lower()


def metric_value(row: Mapping[str, object], metric: str) -> float | None:
    """Return a row metric value using current and legacy column names."""
    normalized = metric.lower().replace("-", "_")
    if normalized == "dino":
        direct = _as_float(row.get("dino_similarity"))
        if direct is not None:
            return direct
        if _semantic_metric(row) == "dino_similarity":
            return _as_float(row.get("semantic_similarity"))
        return None
    if normalized == "clip":
        direct = _as_float(row.get("clip_image_similarity"))
        if direct is not None:
            return direct
        if _semantic_metric(row) == "clip_image_similarity":
            return _as_float(row.get("semantic_similarity"))
        return None
    if normalized == "dreamsim":
        return _as_float(row.get("dreamsim_distance", row.get("dreamsim")))
    if normalized == "ssim":
        return _as_float(row.get("ssim"))
    if normalized == "lpips":
        return _as_float(row.get("lpips_distance", row.get("lpips")))
    raise ValueError(f"Unsupported collapse metric: {metric}")


def attack_success(row: Mapping[str, object]) -> bool:
    """Return source-victim attack success for a metrics row."""
    if "success" in row and not _is_missing(row.get("success")):
        return _as_bool(row["success"])
    adv_pred = row.get("adv_pred")
    true_label = row.get("true_label")
    if _is_missing(adv_pred) or _is_missing(true_label):
        raise KeyError("Row must contain either success or adv_pred/true_label.")
    return str(adv_pred) != str(true_label)


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    if q < 0.0 or q > 1.0:
        raise ValueError(f"Quantile must be in [0, 1], got {q}.")
    import numpy as np

    return float(np.quantile(np.asarray(values, dtype=float), q))


def _finite_metric_values(rows: Iterable[Mapping[str, object]], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = metric_value(row, metric)
        if value is not None:
            values.append(value)
    return values


def calibrate_thresholds(
    benign_rows: Sequence[Mapping[str, object]],
    *,
    dino_quantile: float = 0.05,
    clip_quantile: float = 0.05,
    dreamsim_quantile: float = 0.95,
    source: str = "benign_edit_quantile",
) -> CollapseThresholds:
    """Calibrate preservation thresholds from benign-edit metric rows."""
    return CollapseThresholds(
        dino_similarity=_quantile(_finite_metric_values(benign_rows, "dino"), dino_quantile),
        clip_image_similarity=_quantile(_finite_metric_values(benign_rows, "clip"), clip_quantile),
        dreamsim_distance=_quantile(
            _finite_metric_values(benign_rows, "dreamsim"),
            dreamsim_quantile,
        ),
        source=source,
    )


def preservation_failures(
    row: Mapping[str, object],
    thresholds: CollapseThresholds,
) -> tuple[str, ...]:
    """Return semantic-preservation failures for a row."""
    if not thresholds.active_metrics():
        raise ValueError("At least one collapse threshold must be configured.")

    failures: list[str] = []
    if thresholds.dino_similarity is not None:
        value = metric_value(row, "dino")
        if value is None:
            failures.append("dino_missing")
        elif value < thresholds.dino_similarity:
            failures.append("dino_below_threshold")

    if thresholds.clip_image_similarity is not None:
        value = metric_value(row, "clip")
        if value is None:
            failures.append("clip_missing")
        elif value < thresholds.clip_image_similarity:
            failures.append("clip_below_threshold")

    if thresholds.dreamsim_distance is not None:
        value = metric_value(row, "dreamsim")
        if value is None:
            failures.append("dreamsim_missing")
        elif value > thresholds.dreamsim_distance:
            failures.append("dreamsim_above_threshold")

    return tuple(failures)


def semantic_preserved(row: Mapping[str, object], thresholds: CollapseThresholds) -> bool:
    """Return whether a row satisfies all calibrated preservation constraints."""
    return not preservation_failures(row, thresholds)


def annotate_collapse_row(
    row: Mapping[str, object],
    thresholds: CollapseThresholds,
) -> dict[str, Any]:
    """Return a metrics row with collapse/SP-ASR fields appended."""
    failures = preservation_failures(row, thresholds)
    success = attack_success(row)
    preserved = not failures
    collapsed = success and not preserved
    annotated = dict(row)
    annotated.update(
        {
            "collapse_attack_success": success,
            "semantic_preserved": preserved,
            "sp_success": success and preserved,
            "image_collapse": collapsed,
            "collapse_reason": ";".join(failures) if collapsed else "",
            **thresholds.as_dict(),
        }
    )
    return annotated


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def summarize_collapse_rows(
    rows: Sequence[Mapping[str, object]],
    thresholds: CollapseThresholds,
) -> dict[str, Any]:
    """Return collapse summary metrics for rows."""
    annotated = [annotate_collapse_row(row, thresholds) for row in rows]
    count = len(annotated)
    success_count = sum(1 for row in annotated if _as_bool(row["collapse_attack_success"]))
    preserved_count = sum(1 for row in annotated if _as_bool(row["semantic_preserved"]))
    sp_success_count = sum(1 for row in annotated if _as_bool(row["sp_success"]))
    collapse_count = sum(1 for row in annotated if _as_bool(row["image_collapse"]))

    summary = {
        "count": count,
        "success_count": success_count,
        "raw_asr": success_count / count if count else 0.0,
        "semantic_preserved_count": preserved_count,
        "semantic_preservation_rate": preserved_count / count if count else 0.0,
        "sp_success_count": sp_success_count,
        "sp_asr": sp_success_count / count if count else 0.0,
        "collapse_count": collapse_count,
        "collapse_rate": collapse_count / count if count else 0.0,
        "conditional_collapse_rate": collapse_count / success_count if success_count else 0.0,
        "mean_dino_similarity": _mean(_finite_metric_values(rows, "dino")),
        "mean_clip_image_similarity": _mean(_finite_metric_values(rows, "clip")),
        "mean_dreamsim_distance": _mean(_finite_metric_values(rows, "dreamsim")),
        "mean_ssim": _mean(_finite_metric_values(rows, "ssim")),
        "mean_lpips_distance": _mean(_finite_metric_values(rows, "lpips")),
    }
    summary.update(thresholds.as_dict())
    return summary
