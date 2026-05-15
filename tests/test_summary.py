from prompt_attack.metrics.summary import summarize_rows


def test_summary_includes_optional_quality_metrics() -> None:
    rows = [
        {
            "success": True,
            "semantic_constrained_success": True,
            "dino_similarity": 0.9,
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
    assert summary.mean_decision_logit_gap_drop == 1.0
    assert summary.mean_iqa_nima_ava == 4.0
    assert summary.mean_iqa_hyperiqa is None
    assert summary.mean_iqa_musiq_ava == 5.0
    assert summary.mean_iqa_musiq_koniq == 6.0
    assert summary.mean_iqa_tres == 7.0
    assert summary.fid == 12.5
