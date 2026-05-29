from prompt_attack.metrics.summary import summarize_rows


def test_summary_includes_optional_quality_metrics() -> None:
    rows = [
        {
            "success": True,
            "semantic_similarity": 0.9,
            "dino_similarity": 0.9,
            "clip_image_similarity": "",
            "ssim": 0.8,
            "margin_drop": 1.0,
            "confidence_drop": 0.2,
            "pixel_l1_mean": 0.1,
            "pixel_l2": 2.0,
            "pixel_l2_mean": 0.2,
            "pixel_linf": 0.3,
            "iqa_nima_ava": 4.0,
            "iqa_hyperiqa": None,
            "iqa_musiq_ava": 5.0,
            "iqa_musiq_koniq": 6.0,
            "iqa_tres": 7.0,
            "runtime_seconds": 10.0,
        }
    ]

    summary = summarize_rows(rows, fid=12.5)

    assert summary.asr == 1.0
    assert summary.mean_semantic_similarity == 0.9
    assert summary.mean_dino_similarity == 0.9
    assert summary.mean_clip_image_similarity is None
    assert summary.mean_decision_logit_gap_drop == 1.0
    assert summary.mean_iqa_nima_ava == 4.0
    assert summary.mean_iqa_hyperiqa is None
    assert summary.mean_iqa_musiq_ava == 5.0
    assert summary.mean_iqa_musiq_koniq == 6.0
    assert summary.mean_iqa_tres == 7.0
    assert summary.fid == 12.5


def test_summary_parses_csv_boolean_strings() -> None:
    base = {
        "semantic_similarity": 0.9,
        "dino_similarity": 0.9,
        "clip_image_similarity": 0.8,
        "ssim": 0.8,
        "margin_drop": 1.0,
        "confidence_drop": 0.2,
        "pixel_l1_mean": 0.1,
        "pixel_l2": 2.0,
        "pixel_l2_mean": 0.2,
        "pixel_linf": 0.3,
        "runtime_seconds": 10.0,
    }
    rows = [
        {
            **base,
            "success": "True",
        },
        {
            **base,
            "success": "False",
        },
    ]

    summary = summarize_rows(rows)

    assert summary.success_count == 1
    assert summary.asr == 0.5
    assert summary.mean_semantic_similarity == 0.9
    assert summary.mean_clip_image_similarity == 0.8
