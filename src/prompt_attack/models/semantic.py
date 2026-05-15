"""Semantic feature model wrappers."""

from __future__ import annotations


class DINOv2Encoder:
    """DINOv2 CLS feature extractor."""

    def __init__(self, *, device: str, model_name: str = "dinov2_vitb14") -> None:
        import torch

        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2", model_name).to(device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def preprocess_tensor(self, image_tensor):
        """Differentiably resize and normalize an image tensor."""
        import torch.nn.functional as F

        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        resized = F.interpolate(image_tensor, size=(224, 224), mode="bilinear", align_corners=False)
        return (resized - self.mean) / self.std

    def features(self, image_tensor):
        """Return normalized DINO CLS features."""
        import torch.nn.functional as F

        output = self.model(self.preprocess_tensor(image_tensor))
        return F.normalize(output, dim=-1)

    def similarity(self, left, right):
        """Return cosine similarity between two image tensors."""
        return (self.features(left) * self.features(right)).sum(dim=-1)


def build_semantic_model(name: str, *, device: str) -> DINOv2Encoder:
    """Build a semantic model by name."""
    if name != "dinov2_vitb14":
        raise ValueError(f"Unsupported semantic model: {name}")
    return DINOv2Encoder(device=device, model_name=name)

