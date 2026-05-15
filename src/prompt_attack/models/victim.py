"""Victim classifier wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from PIL import Image


@dataclass(frozen=True)
class ClassificationResult:
    pred: int
    pred_conf: float
    true_conf: float
    margin: float


class TorchvisionImageNetVictim:
    """ImageNet pretrained torchvision classifier wrapper."""

    def __init__(self, *, name: str, weights: str, device: str) -> None:
        import torch
        from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

        self.device = device
        self.name = name
        builders: dict[str, tuple[Callable[..., Any], Any]] = {
            "resnet18": (resnet18, ResNet18_Weights),
            "resnet50": (resnet50, ResNet50_Weights),
        }
        if name not in builders:
            raise ValueError(f"Unsupported victim model: {name}")
        builder, weights_enum = builders[name]
        self.weights = weights_enum.DEFAULT if weights == "DEFAULT" else getattr(weights_enum, weights)
        self.categories = list(self.weights.meta["categories"])
        self.model = builder(weights=self.weights).to(device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.preprocess_pil = self.weights.transforms()
        self.resize_size = int(_first_value(self.preprocess_pil.resize_size))
        self.crop_size = int(_first_value(self.preprocess_pil.crop_size))
        mean = getattr(self.preprocess_pil, "mean", [0.485, 0.456, 0.406])
        std = getattr(self.preprocess_pil, "std", [0.229, 0.224, 0.225])
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)

    def preprocess_tensor(self, image_tensor):
        """Differentiably resize, center crop, and normalize an image tensor in [0, 1]."""
        import torch.nn.functional as F

        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        _, _, height, width = image_tensor.shape
        if height <= width:
            resize_height = self.resize_size
            resize_width = round(width * self.resize_size / height)
        else:
            resize_height = round(height * self.resize_size / width)
            resize_width = self.resize_size
        resized = F.interpolate(
            image_tensor,
            size=(resize_height, resize_width),
            mode="bilinear",
            align_corners=False,
        )
        top = max((resize_height - self.crop_size) // 2, 0)
        left = max((resize_width - self.crop_size) // 2, 0)
        cropped = resized[:, :, top : top + self.crop_size, left : left + self.crop_size]
        return (cropped - self.mean) / self.std

    def logits_from_tensor(self, image_tensor):
        """Return logits for a differentiable image tensor."""
        return self.model(self.preprocess_tensor(image_tensor))

    def evaluate_logits(self, logits, true_label: int) -> ClassificationResult:
        """Compute prediction, true confidence, and margin from logits."""
        import torch

        probs = torch.softmax(logits, dim=-1)
        pred = int(torch.argmax(logits, dim=-1).item())
        pred_conf = float(probs[0, pred].detach().cpu().item())
        true_conf = float(probs[0, true_label].detach().cpu().item())
        true_logit = logits[0, true_label]
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[0, true_label] = False
        other_max = logits.masked_select(mask).view(1, -1).max(dim=-1).values[0]
        margin = float((true_logit - other_max).detach().cpu().item())
        return ClassificationResult(pred=pred, pred_conf=pred_conf, true_conf=true_conf, margin=margin)

    def evaluate_pil(self, image: Image.Image, true_label: int) -> ClassificationResult:
        """Evaluate one PIL image without gradient tracking."""
        import torch

        tensor = self.preprocess_pil(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        return self.evaluate_logits(logits, true_label)


def _first_value(value: int | list[int] | tuple[int, ...]) -> int:
    if isinstance(value, int):
        return value
    return int(value[0])


def build_victim(name: str, *, weights: str = "DEFAULT", device: str) -> TorchvisionImageNetVictim:
    """Build a victim model by name."""
    return TorchvisionImageNetVictim(name=name, weights=weights, device=device)
