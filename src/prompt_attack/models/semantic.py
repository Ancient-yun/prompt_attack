"""Semantic feature model wrappers."""

from __future__ import annotations


class DINOv2Encoder:
    """DINOv2 CLS feature extractor."""

    metric_name = "dino_similarity"

    def __init__(self, *, device: str, model_name: str = "dinov2_vitb14") -> None:
        import torch

        self.name = model_name
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


class CLIPImageEncoder:
    """CLIP image encoder for image-to-image semantic similarity."""

    metric_name = "clip_image_similarity"

    def __init__(
        self,
        *,
        device: str,
        model_id: str = "openai/clip-vit-base-patch32",
    ) -> None:
        import torch
        from transformers import CLIPModel

        self.name = model_id
        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()  # type: ignore[arg-type]
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073],
            device=device,
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711],
            device=device,
        ).view(1, 3, 1, 1)
        self.image_size = int(self.model.config.vision_config.image_size)

    def preprocess_tensor(self, image_tensor):
        """Differentiably resize and normalize an image tensor."""
        import torch.nn.functional as F

        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        resized = F.interpolate(
            image_tensor,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
        )
        return (resized - self.mean) / self.std

    def features(self, image_tensor):
        """Return normalized CLIP image features."""
        import torch.nn.functional as F

        vision_output = self.model.vision_model(pixel_values=self.preprocess_tensor(image_tensor))
        output = self.model.visual_projection(vision_output.pooler_output)
        return F.normalize(output, dim=-1)

    def similarity(self, left, right):
        """Return cosine similarity between two image tensors."""
        return (self.features(left) * self.features(right)).sum(dim=-1)


def build_semantic_model(name: str, *, device: str) -> DINOv2Encoder | CLIPImageEncoder:
    """Build a semantic model by name."""
    normalized = name.lower().replace("-", "_").replace("/", "_")
    if normalized == "dinov2_vitb14":
        return DINOv2Encoder(device=device, model_name="dinov2_vitb14")
    if normalized in {"clip_vit_b32", "clip_vit_base_patch32", "openai_clip_vit_base_patch32"}:
        return CLIPImageEncoder(device=device, model_id="openai/clip-vit-base-patch32")
    if normalized in {"clip_vit_l14", "clip_vit_large_patch14", "openai_clip_vit_large_patch14"}:
        return CLIPImageEncoder(device=device, model_id="openai/clip-vit-large-patch14")
    if name.startswith("openai/clip-"):
        return CLIPImageEncoder(device=device, model_id=name)
    raise ValueError(f"Unsupported semantic model: {name}")
