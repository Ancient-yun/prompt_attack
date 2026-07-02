import pytest

from prompt_attack.metrics.collapse import (
    CollapseThresholds,
    annotate_collapse_row,
    attack_success,
    calibrate_thresholds,
    semantic_preserved,
    summarize_collapse_rows,
)


def test_calibrate_thresholds_uses_benign_edit_quantiles() -> None:
    benign_rows = [
        {"dino_similarity": "0.80", "clip_image_similarity": "0.70", "dreamsim_distance": "0.10"},
        {"dino_similarity": "0.90", "clip_image_similarity": "0.80", "dreamsim_distance": "0.20"},
        {"dino_similarity": "1.00", "clip_image_similarity": "0.90", "dreamsim_distance": "0.30"},
    ]

    thresholds = calibrate_thresholds(
        benign_rows,
        dino_quantile=0.0,
        clip_quantile=0.0,
        dreamsim_quantile=1.0,
    )

    assert thresholds.dino_similarity == 0.80
    assert thresholds.clip_image_similarity == 0.70
    assert thresholds.dreamsim_distance == 0.30


def test_collapse_requires_attack_success_and_semantic_failure() -> None:
    thresholds = CollapseThresholds(
        dino_similarity=0.70,
        clip_image_similarity=0.70,
        dreamsim_distance=0.30,
    )
    collapsed = {
        "success": "True",
        "dino_similarity": "0.60",
        "clip_image_similarity": "0.80",
        "dreamsim_distance": "0.20",
    }
    failed_attack = {
        "success": "False",
        "dino_similarity": "0.60",
        "clip_image_similarity": "0.80",
        "dreamsim_distance": "0.20",
    }

    collapsed_row = annotate_collapse_row(collapsed, thresholds)
    failed_row = annotate_collapse_row(failed_attack, thresholds)

    assert collapsed_row["semantic_preserved"] is False
    assert collapsed_row["sp_success"] is False
    assert collapsed_row["image_collapse"] is True
    assert collapsed_row["collapse_reason"] == "dino_below_threshold"
    assert failed_row["semantic_preserved"] is False
    assert failed_row["image_collapse"] is False


def test_semantic_preservation_uses_all_available_thresholds() -> None:
    thresholds = CollapseThresholds(dino_similarity=0.70, clip_image_similarity=0.70)

    assert semantic_preserved(
        {"success": True, "dino_similarity": 0.71, "clip_image_similarity": 0.71},
        thresholds,
    )
    assert not semantic_preserved(
        {"success": True, "dino_similarity": 0.71, "clip_image_similarity": 0.69},
        thresholds,
    )


def test_attack_success_can_be_derived_from_predictions() -> None:
    assert attack_success({"adv_pred": "12", "true_label": "11"})
    assert not attack_success({"adv_pred": "11", "true_label": "11"})


def test_summarize_collapse_rows_reports_sp_asr_and_conditional_collapse() -> None:
    thresholds = CollapseThresholds(dino_similarity=0.70, clip_image_similarity=0.70)
    rows = [
        {"success": True, "dino_similarity": 0.90, "clip_image_similarity": 0.80, "ssim": 0.7},
        {"success": True, "dino_similarity": 0.50, "clip_image_similarity": 0.80, "ssim": 0.2},
        {"success": False, "dino_similarity": 0.95, "clip_image_similarity": 0.90, "ssim": 0.9},
    ]

    summary = summarize_collapse_rows(rows, thresholds)

    assert summary["raw_asr"] == pytest.approx(2 / 3)
    assert summary["sp_asr"] == pytest.approx(1 / 3)
    assert summary["collapse_rate"] == pytest.approx(1 / 3)
    assert summary["conditional_collapse_rate"] == pytest.approx(1 / 2)
    assert summary["mean_dino_similarity"] == pytest.approx((0.90 + 0.50 + 0.95) / 3)
