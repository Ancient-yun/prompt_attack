"""No-reference image quality assessment with optional pyiqa backends."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any

from prompt_attack.config import NRIQAConfig


@dataclass(frozen=True)
class NRIQAMetricSpec:
    key: str
    candidates: tuple[str, ...]


NR_IQA_SPECS: dict[str, NRIQAMetricSpec] = {
    "nima_ava": NRIQAMetricSpec("nima_ava", ("nima-vgg16-ava", "nima")),
    "hyperiqa": NRIQAMetricSpec("hyperiqa", ("hyperiqa",)),
    "musiq_ava": NRIQAMetricSpec("musiq_ava", ("musiq-ava",)),
    "musiq_koniq": NRIQAMetricSpec(
        "musiq_koniq",
        ("musiq-koniq", "musiq", "musiq-paq2piq", "musiq-spaq", "nima-koniq"),
    ),
    "tres": NRIQAMetricSpec("tres", ("tres",)),
}


def nriqa_result_keys(metric_keys: tuple[str, ...]) -> tuple[str, ...]:
    """Return stable row keys for configured NR-IQA metrics."""
    return tuple(f"iqa_{key}" for key in metric_keys)


def empty_nriqa_scores(metric_keys: tuple[str, ...]) -> dict[str, float | None]:
    """Return empty NR-IQA score fields for disabled or unavailable evaluators."""
    return {key: None for key in nriqa_result_keys(metric_keys)}


def _normalize_metric_name(metric_name: str) -> str:
    return re.sub(r"\s+", "", metric_name).lower().replace("_", "-")


def _available_pyiqa_metrics(pyiqa_module: Any) -> set[str]:
    if hasattr(pyiqa_module, "list_models"):
        return set(pyiqa_module.list_models())
    if hasattr(pyiqa_module, "list_metrics"):
        return set(pyiqa_module.list_metrics())
    return set()


def _resolve_metric_name(
    spec: NRIQAMetricSpec,
    available_metrics: set[str],
) -> tuple[str, str | None]:
    for candidate in spec.candidates:
        if candidate in available_metrics:
            return candidate, None
        normalized_candidate = _normalize_metric_name(candidate)
        for available in available_metrics:
            if _normalize_metric_name(available) == normalized_candidate:
                return available, None

    normalized = _normalize_metric_name(spec.key)
    close_matches = difflib.get_close_matches(normalized, sorted(available_metrics), n=5)
    return "", (
        f"NR-IQA metric '{spec.key}' is unavailable in this pyiqa version. "
        f"Closest matches: {close_matches if close_matches else sorted(available_metrics)[:10]}"
    )


class NoReferenceIQAEvaluator:
    """Compute NIMA/HyperIQA/MUSIQ/TReS style no-reference quality scores."""

    def __init__(self, config: NRIQAConfig, *, device: str) -> None:
        self.config = config
        self.metric_keys = tuple(metric for metric in config.metrics if metric in NR_IQA_SPECS)
        self.device = device
        self.available = False
        self.error: str | None = None
        self._metrics: dict[str, Any] = {}
        if not config.enabled:
            return
        try:
            import pyiqa
        except ImportError as exc:
            self.error = (
                "pyiqa is not installed; NR-IQA metrics will be empty. "
                "Rebuild the Docker image after installing the updated dependencies."
            )
            if config.fail_on_error:
                raise RuntimeError(self.error) from exc
            return

        available_metrics = _available_pyiqa_metrics(pyiqa)
        for key in self.metric_keys:
            spec = NR_IQA_SPECS[key]
            resolved, note = _resolve_metric_name(spec, available_metrics)
            if not resolved:
                self.error = note or f"NR-IQA metric '{spec.key}' is unavailable."
                if config.fail_on_error:
                    raise RuntimeError(self.error)
                continue
            try:
                self._metrics[key] = pyiqa.create_metric(resolved, device=device)
            except Exception as exc:  # pragma: no cover - backend/model dependent
                self.error = f"Failed to create pyiqa metric '{resolved}': {exc}"
                if config.fail_on_error:
                    raise RuntimeError(self.error) from exc
        self.available = bool(self._metrics)

    def empty_scores(self) -> dict[str, float | None]:
        """Return empty fields for every configured metric."""
        return empty_nriqa_scores(self.metric_keys)

    def score_tensor(self, image_tensor) -> dict[str, float | None]:
        """Score a [1, C, H, W] or [C, H, W] tensor in [0, 1]."""
        import torch

        scores = self.empty_scores()
        if not self.available:
            return scores
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.detach().clamp(0, 1).to(self.device)
        for key, metric in self._metrics.items():
            with torch.no_grad():
                value = metric(image_tensor)
            scores[f"iqa_{key}"] = float(value.detach().mean().cpu().item())
        return scores
