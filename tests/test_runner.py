from dataclasses import replace
from pathlib import Path

from prompt_attack.attacks.runner import LearnableTokenAttackRunner
from prompt_attack.config import load_config
from prompt_attack.data.imagenet import ImageRecord


class RaisingVictim:
    def evaluate_pil(self, image: object, true_label: int) -> object:
        del image, true_label
        raise AssertionError("victim should not be evaluated when clean_correct_only is false")


def test_clean_correct_filter_skips_victim_when_disabled() -> None:
    config = load_config(Path("configs/flux2_resnet18.yaml"))
    config = replace(
        config,
        data=replace(config.data, clean_correct_only=False, images_per_class=1),
    )
    records = [
        ImageRecord(
            path=Path("does-not-exist.png"),
            synset="class_0000",
            class_label="dummy",
            class_index=0,
            image_id="dummy",
        )
    ]

    selected = LearnableTokenAttackRunner(config, device="cpu")._clean_correct_records(
        records,
        RaisingVictim(),
    )

    assert selected == records
