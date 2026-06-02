import pytest
import torch
import torch.nn.functional as F

from prompt_attack.attacks.losses import (
    attack_semantic_loss_weights,
    attack_loss_from_objective,
    semantic_image_loss,
    margin_hinge_loss,
    objective_loss_components,
    cr_loss,
    is_cr_dino_objective,
    is_cr_objective,
    is_margin_dino_constraint_objective,
    is_semantic_only_objective,
    negative_cross_entropy_loss,
    untargeted_margin_loss,
    validate_lambda_sem,
    weighted_attack_semantic_loss,
)


def test_negative_cross_entropy_loss() -> None:
    logits = torch.tensor([[1.0, 2.0, -1.0]])
    expected = -F.cross_entropy(logits, torch.tensor([1]))
    actual = negative_cross_entropy_loss(logits, true_label=1)
    assert torch.allclose(actual, expected)


def test_batch_negative_cross_entropy_loss_matches_scalar_loop() -> None:
    logits = torch.tensor([[1.0, 2.0, -1.0], [3.0, -2.0, 0.5]])
    labels = torch.tensor([1, 2])

    per_sample = negative_cross_entropy_loss(logits, labels, reduction="none")
    expected_per_sample = torch.stack(
        [
            negative_cross_entropy_loss(logits[0:1], 1),
            negative_cross_entropy_loss(logits[1:2], 2),
        ]
    )

    assert torch.allclose(per_sample, expected_per_sample)
    assert torch.allclose(negative_cross_entropy_loss(logits, labels), expected_per_sample.mean())


def test_batch_margin_loss_matches_scalar_loop() -> None:
    logits = torch.tensor([[1.0, 2.0, -1.0], [3.0, -2.0, 0.5]])
    labels = torch.tensor([1, 2])

    per_sample = untargeted_margin_loss(logits, labels, reduction="none")
    expected_per_sample = torch.stack(
        [
            untargeted_margin_loss(logits[0:1], 1),
            untargeted_margin_loss(logits[1:2], 2),
        ]
    )

    assert torch.allclose(per_sample, expected_per_sample)
    assert torch.allclose(untargeted_margin_loss(logits, labels), expected_per_sample.mean())


def test_attack_loss_dispatch() -> None:
    logits = torch.tensor([[1.0, 2.0, -1.0]])
    assert torch.allclose(
        attack_loss_from_objective(logits, 1, "untargeted_margin"),
        untargeted_margin_loss(logits, 1),
    )
    assert torch.allclose(
        attack_loss_from_objective(logits, 1, "negative_cross_entropy"),
        negative_cross_entropy_loss(logits, 1),
    )
    assert torch.allclose(
        attack_loss_from_objective(logits, 1, "cr"),
        cr_loss(logits, 1),
    )
    assert torch.allclose(attack_loss_from_objective(logits, 1, "cr_dino"), cr_loss(logits, 1))


def test_cr_loss_matches_classification_rejection_term() -> None:
    logits = torch.tensor([[1.0, 2.0, -1.0]])

    assert torch.allclose(cr_loss(logits, true_label=1), negative_cross_entropy_loss(logits, 1))
    assert is_cr_objective("cr")
    assert is_cr_objective("classification-rejection")
    assert is_cr_dino_objective("cr-dino")
    assert is_margin_dino_constraint_objective("margin-dino-constraint")
    assert is_margin_dino_constraint_objective("margin-clip-img2img")
    assert is_semantic_only_objective("clip-img2img")
    assert is_semantic_only_objective("dino-img2img")


def test_cr_objective_disables_semantic_loss_weight() -> None:
    assert attack_semantic_loss_weights("cr", lambda_sem=0.9) == (1.0, 0.0)
    assert attack_semantic_loss_weights("cr_dino", lambda_sem=0.9) == pytest.approx((0.1, 0.9))
    assert attack_semantic_loss_weights("clip_img2img", lambda_sem=0.9) == (0.0, 1.0)


def test_weighted_attack_semantic_loss_uses_complementary_weights() -> None:
    attack_loss = torch.tensor(-8.0)
    semantic_loss = torch.tensor(0.5)

    actual = weighted_attack_semantic_loss(attack_loss, semantic_loss, lambda_sem=0.25)

    assert torch.allclose(actual, torch.tensor(-5.875))


def test_margin_hinge_loss_is_zero_after_attack_success() -> None:
    logits = torch.tensor([[1.0, 3.0, 0.0], [4.0, 1.0, 2.0]])
    labels = torch.tensor([0, 0])

    losses = margin_hinge_loss(logits, labels, margin=0.0, reduction="none")

    assert torch.allclose(losses, torch.tensor([0.0, 2.0]))


def test_semantic_image_loss_is_continuous() -> None:
    similarity = torch.tensor([0.90, 0.70])

    losses = semantic_image_loss(similarity, reduction="none")

    assert torch.allclose(losses, torch.tensor([0.10, 0.30]))


def test_margin_dino_loss_components_are_batched() -> None:
    logits = torch.tensor([[1.0, 3.0, 0.0], [4.0, 1.0, 2.0]])
    labels = torch.tensor([0, 0])
    semantic_similarity = torch.tensor([0.90, 0.70])

    attack, semantic, weighted_semantic, total = objective_loss_components(
        logits,
        labels,
        "margin_dino",
        semantic_similarity=semantic_similarity,
        lambda_sem=0.5,
        semantic_loss_weight=10.0,
        attack_margin=0.0,
    )

    assert torch.allclose(attack, torch.tensor([0.0, 2.0]))
    assert torch.allclose(semantic, torch.tensor([0.10, 0.30]))
    assert torch.allclose(weighted_semantic, 10.0 * semantic)
    assert torch.allclose(total, attack + weighted_semantic)


def test_margin_clip_img2img_loss_uses_same_continuous_similarity_form() -> None:
    logits = torch.tensor([[1.0, 3.0, 0.0]])
    labels = torch.tensor([0])
    clip_similarity = torch.tensor([0.75])

    attack, semantic, weighted_semantic, total = objective_loss_components(
        logits,
        labels,
        "margin_clip_img2img",
        semantic_similarity=clip_similarity,
        lambda_sem=0.0,
        semantic_loss_weight=3.0,
        attack_margin=0.0,
    )

    assert torch.allclose(attack, torch.tensor([0.0]))
    assert torch.allclose(semantic, torch.tensor([0.25]))
    assert torch.allclose(weighted_semantic, torch.tensor([0.75]))
    assert torch.allclose(total, torch.tensor([0.75]))


def test_clip_img2img_loss_has_no_margin_term() -> None:
    logits = torch.tensor([[10.0, -2.0, 0.0]])
    labels = torch.tensor([0])
    clip_similarity = torch.tensor([0.75])

    attack, semantic, weighted_semantic, total = objective_loss_components(
        logits,
        labels,
        "clip_img2img",
        semantic_similarity=clip_similarity,
        lambda_sem=0.0,
        semantic_loss_weight=3.0,
        attack_margin=0.0,
    )

    assert torch.allclose(attack, torch.tensor([0.0]))
    assert torch.allclose(semantic, torch.tensor([0.25]))
    assert torch.allclose(weighted_semantic, torch.tensor([0.25]))
    assert torch.allclose(total, weighted_semantic)


def test_dino_img2img_loss_has_no_margin_term() -> None:
    logits = torch.tensor([[10.0, -2.0, 0.0]])
    labels = torch.tensor([0])
    dino_similarity = torch.tensor([0.60])

    attack, semantic, weighted_semantic, total = objective_loss_components(
        logits,
        labels,
        "dino_img2img",
        semantic_similarity=dino_similarity,
        lambda_sem=0.0,
        semantic_loss_weight=3.0,
        attack_margin=0.0,
    )

    assert torch.allclose(attack, torch.tensor([0.0]))
    assert torch.allclose(semantic, torch.tensor([0.40]))
    assert torch.allclose(weighted_semantic, torch.tensor([0.40]))
    assert torch.allclose(total, weighted_semantic)


def test_micro_batch_scaled_loss_matches_global_batch_mean() -> None:
    logits = torch.tensor(
        [
            [1.0, 3.0, 0.0],
            [4.0, 1.0, 2.0],
            [0.0, 2.0, 1.0],
            [3.0, 0.0, 1.0],
        ]
    )
    labels = torch.tensor([0, 0, 1, 2])
    semantic_similarity = torch.tensor([0.90, 0.70, 0.80, 0.95])
    _, _, _, full_total = objective_loss_components(
        logits,
        labels,
        "margin_dino",
        semantic_similarity=semantic_similarity,
        lambda_sem=0.0,
        semantic_loss_weight=10.0,
        attack_margin=0.0,
    )
    accumulated = torch.tensor(0.0)
    for start in (0, 2):
        _, _, _, micro_total = objective_loss_components(
            logits[start : start + 2],
            labels[start : start + 2],
            "margin_dino",
            semantic_similarity=semantic_similarity[start : start + 2],
            lambda_sem=0.0,
            semantic_loss_weight=10.0,
            attack_margin=0.0,
        )
        accumulated = accumulated + micro_total.sum() / len(labels)

    assert torch.allclose(accumulated, full_total.mean())


def test_lambda_sem_must_be_unit_interval() -> None:
    for value in (-0.1, 1.1):
        try:
            validate_lambda_sem(value)
        except ValueError as exc:
            assert "1 - lambda_sem" in str(exc)
        else:
            raise AssertionError(f"validate_lambda_sem accepted {value}")
