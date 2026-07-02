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

    def similarity(self, left, right, labels=None):
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

    def similarity(self, left, right, labels=None):
        """Return cosine similarity between two image tensors."""
        return (self.features(left) * self.features(right)).sum(dim=-1)


class LPIPSSimilarity:
    """Differentiable LPIPS-based preservation signal.

    Exposes ``similarity(left, right) = 1 - LPIPS(left, right)`` per sample so the shared
    ``1 - similarity`` semantic loss becomes the LPIPS distance itself. Unlike DINO/CLIP
    cosine similarity, LPIPS is sensitive to pixel-level structure, so it penalises the
    fixed adversarial overlay that DINO/CLIP are blind to.
    """

    metric_name = "lpips_similarity"

    def __init__(self, *, device: str) -> None:
        import pyiqa

        self.name = "lpips"
        self.device = device
        # as_loss=True builds a metric whose forward keeps gradients; we call the wrapped
        # net directly to get a per-sample distance (the top-level __call__ mean-reduces).
        self.metric = pyiqa.create_metric("lpips", device=device, as_loss=True)
        self.net = self.metric.net
        for param in self.net.parameters():
            param.requires_grad_(False)

    def similarity(self, left, right, labels=None):
        """Return 1 - LPIPS distance between two [0, 1] image tensors, per sample."""
        if left.ndim == 3:
            left = left.unsqueeze(0)
        if right.ndim == 3:
            right = right.unsqueeze(0)
        distance = self.net(left, right).reshape(-1)
        return 1.0 - distance


class ClipZeroShotOracle:
    """CLIP zero-shot classifier used as an identity oracle.

    ``similarity(original, adv, labels)`` returns the CLIP softmax probability of the
    true class for the *adversarial* image, differentiable in the image. Used as a
    semantic-preservation signal that (unlike pixel/LPIPS or DINO cosine) permits large
    but on-manifold edits — pose/viewpoint changes are fine as long as CLIP still reads
    the original class, while collapse drops the true-class probability.
    """

    metric_name = "oracle_true_prob"

    def __init__(
        self,
        *,
        device: str,
        class_labels,
        model_id: str = "openai/clip-vit-base-patch32",
        prompt_template: str = "a photo of a {label}",
    ) -> None:
        import torch
        import torch.nn.functional as F
        from transformers import CLIPModel, CLIPTokenizer

        self.name = f"clip_oracle:{model_id}"
        self.device = device
        self.class_labels = list(class_labels)
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()  # type: ignore[arg-type]
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        self.image_size = int(self.model.config.vision_config.image_size)
        tokenizer = CLIPTokenizer.from_pretrained(model_id)
        texts = [prompt_template.format(label=label) for label in self.class_labels]
        tokens = tokenizer(texts, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            text_output = self.model.text_model(**tokens)
            text_features = self.model.text_projection(text_output.pooler_output)
            self.text_features = F.normalize(text_features, dim=-1).detach()  # [C, D]
        self.logit_scale = self.model.logit_scale.exp().detach()

    def _preprocess(self, image_tensor):
        import torch.nn.functional as F

        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        resized = F.interpolate(
            image_tensor, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False
        )
        return (resized - self.mean) / self.std

    def logits(self, image_tensor):
        """Return per-sample CLIP zero-shot logits over the configured class labels."""
        import torch.nn.functional as F

        vision_output = self.model.vision_model(pixel_values=self._preprocess(image_tensor))
        features = F.normalize(self.model.visual_projection(vision_output.pooler_output), dim=-1)
        return self.logit_scale * features @ self.text_features.t()  # [B, C]

    def similarity(self, left, right, labels=None):
        """Return the true-class CLIP probability for the adversarial image ``right``."""
        import torch

        if labels is None:
            raise ValueError("ClipZeroShotOracle.similarity requires labels.")
        probs = self.logits(right).softmax(dim=-1)
        index = labels.to(device=probs.device, dtype=torch.long).view(-1, 1)
        return probs.gather(1, index).squeeze(1)


def build_semantic_model(
    name: str, *, device: str, class_labels=None
) -> DINOv2Encoder | CLIPImageEncoder | LPIPSSimilarity | ClipZeroShotOracle:
    """Build a semantic model by name."""
    normalized = name.lower().replace("-", "_").replace("/", "_")
    if normalized == "dinov2_vitb14":
        return DINOv2Encoder(device=device, model_name="dinov2_vitb14")
    if normalized in {"lpips", "lpips_alex"}:
        return LPIPSSimilarity(device=device)
    if normalized in {"clip_oracle", "oracle", "clip_zero_shot", "clip_zeroshot"}:
        if class_labels is None:
            raise ValueError("clip_oracle semantic model requires class_labels.")
        return ClipZeroShotOracle(device=device, class_labels=class_labels)
    if normalized in {"clip_vit_b32", "clip_vit_base_patch32", "openai_clip_vit_base_patch32"}:
        return CLIPImageEncoder(device=device, model_id="openai/clip-vit-base-patch32")
    if normalized in {"clip_vit_l14", "clip_vit_large_patch14", "openai_clip_vit_large_patch14"}:
        return CLIPImageEncoder(device=device, model_id="openai/clip-vit-large-patch14")
    if name.startswith("openai/clip-"):
        return CLIPImageEncoder(device=device, model_id=name)
    raise ValueError(f"Unsupported semantic model: {name}")
