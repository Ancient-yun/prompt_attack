from pathlib import Path

import torch

from prompt_attack.models.victim import build_victim


def test_resnet18_mixed13_checkpoint_victim_loads(tmp_path: Path) -> None:
    from torch import nn
    from torchvision.models import resnet18

    categories = [f"class_{index}" for index in range(13)]
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(categories))
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "architecture": "resnet18",
            "victim_name": "resnet18_mixed13",
            "model_state_dict": model.state_dict(),
            "categories": categories,
            "preprocess": {"weights": "IMAGENET1K_V1"},
        },
        checkpoint_path,
    )

    victim = build_victim(
        "resnet18_mixed13",
        checkpoint_path=checkpoint_path,
        device="cpu",
    )
    logits = victim.logits_from_tensor(torch.zeros((1, 3, 224, 224)))

    assert victim.categories == categories
    assert tuple(logits.shape) == (1, 13)
