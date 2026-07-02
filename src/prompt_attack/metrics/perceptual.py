"""Perceptual pair metrics used for post-hoc collapse evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


def _auto_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _pil_batch(images: Sequence[Image.Image], *, size: tuple[int, int] | None = None):
    import torch
    from torchvision.transforms import functional as TF

    tensors = []
    for image in images:
        if size is not None and image.size != size:
            image = image.resize(size, Image.Resampling.BICUBIC)
        tensors.append(TF.to_tensor(image))
    return torch.stack(tensors, dim=0)


@dataclass(frozen=True)
class ImagePair:
    """A resolved clean/generated image pair."""

    original: Path
    generated: Path


class LPIPSDistance:
    """LPIPS distance wrapper backed by the existing pyiqa dependency."""

    def __init__(self, *, device: str | None = None) -> None:
        import pyiqa

        self.device = device or _auto_device()
        self.metric = pyiqa.create_metric("lpips", device=self.device, as_loss=False)

    def distances(self, pairs: Sequence[ImagePair], *, batch_size: int = 8) -> list[float]:
        import torch

        output: list[float] = []
        with torch.no_grad():
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start : start + batch_size]
                generated = [_load_rgb(pair.generated) for pair in batch]
                # LPIPS expects pair tensors with the same spatial size. Align originals to
                # generated images, matching the existing image-quality metric policy.
                originals = [
                    _load_rgb(pair.original).resize(image.size, Image.Resampling.BICUBIC)
                    for pair, image in zip(batch, generated, strict=True)
                ]
                left = _pil_batch(originals).to(self.device)
                right = _pil_batch(generated).to(self.device)
                values = self.metric(left, right).flatten().detach().cpu().tolist()
                output.extend(float(value) for value in values)
        return output


class DreamSimDistance:
    """DreamSim distance wrapper.

    DreamSim returns a distance, not a similarity: lower means more perceptually similar.
    """

    def __init__(
        self,
        *,
        device: str | None = None,
        cache_dir: Path | str = ".cache/dreamsim",
        dreamsim_type: str = "ensemble",
    ) -> None:
        from dreamsim import dreamsim

        self.device = device or _auto_device()
        self.model, self.preprocess = dreamsim(
            pretrained=True,
            device=self.device,
            cache_dir=str(cache_dir),
            dreamsim_type=dreamsim_type,
        )

    def distances(self, pairs: Sequence[ImagePair], *, batch_size: int = 8) -> list[float]:
        import torch

        output: list[float] = []
        with torch.no_grad():
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start : start + batch_size]
                left = torch.cat(
                    [self.preprocess(_load_rgb(pair.original)) for pair in batch],
                    dim=0,
                ).to(self.device)
                right = torch.cat(
                    [self.preprocess(_load_rgb(pair.generated)) for pair in batch],
                    dim=0,
                ).to(self.device)
                values = self.model(left, right).flatten().detach().cpu().tolist()
                output.extend(float(value) for value in values)
        return output


class SemanticImageSimilarity:
    """DINO/CLIP image-to-image cosine similarity wrapper."""

    def __init__(self, model_name: str, *, device: str | None = None) -> None:
        from prompt_attack.models.semantic import build_semantic_model

        self.device = device or _auto_device()
        self.model = build_semantic_model(model_name, device=self.device)

    def similarities(self, pairs: Sequence[ImagePair], *, batch_size: int = 8) -> list[float]:
        import torch

        output: list[float] = []
        with torch.no_grad():
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start : start + batch_size]
                originals = [_load_rgb(pair.original) for pair in batch]
                generated = [_load_rgb(pair.generated) for pair in batch]
                # Stackable batch input is needed before model-specific preprocessing; the
                # semantic encoders resize again to their own canonical input size.
                left = _pil_batch(originals, size=(512, 512)).to(self.device)
                right = _pil_batch(generated, size=(512, 512)).to(self.device)
                values = self.model.similarity(left, right).flatten().detach().cpu().tolist()
                output.extend(float(value) for value in values)
        return output
