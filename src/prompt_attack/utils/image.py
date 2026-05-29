"""Image tensor conversion and grid helpers."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def pil_to_tensor(image: Image.Image, *, device: str):
    """Convert PIL RGB image to a float tensor in [0, 1]."""
    import torch
    import torchvision.transforms.functional as F

    return F.to_tensor(image).unsqueeze(0).to(device=device, dtype=torch.float32)


def tensor_to_pil(tensor) -> Image.Image:
    """Convert a float image tensor in [0, 1] to PIL."""
    import torchvision.transforms.functional as F

    if tensor.ndim == 4:
        tensor = tensor[0]
    return F.to_pil_image(tensor.detach().clamp(0, 1).cpu())


def _normalize_image_format(image_format: str | None, path: Path) -> str:
    if image_format:
        normalized = image_format.lower().strip()
    else:
        normalized = path.suffix.lower().lstrip(".")
    if normalized in {"jpg", "jpeg"}:
        return "JPEG"
    if normalized == "webp":
        return "WEBP"
    return "PNG"


def image_extension(image_format: str) -> str:
    """Return a file extension for a configured image format."""
    normalized = image_format.lower().strip()
    if normalized in {"jpg", "jpeg"}:
        return ".jpg"
    if normalized == "webp":
        return ".webp"
    return ".png"


def save_image(
    image: Image.Image,
    path: Path,
    *,
    image_format: str | None = None,
    quality: int = 95,
) -> None:
    """Save an image and create its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_format = _normalize_image_format(image_format, path)
    if pil_format == "JPEG":
        image = image.convert("RGB")
        image.save(path, format=pil_format, quality=quality)
    elif pil_format == "WEBP":
        image = image.convert("RGB")
        image.save(path, format=pil_format, quality=quality)
    else:
        image.save(path, format=pil_format, compress_level=1)


class AsyncImageWriter:
    """Bounded background image writer for CPU-bound encode/disk work."""

    def __init__(
        self,
        *,
        max_workers: int = 4,
        max_pending: int | None = None,
        image_format: str = "png",
        quality: int = 95,
    ) -> None:
        self.max_workers = max(1, max_workers)
        self.max_pending = max_pending or self.max_workers * 8
        self.image_format = image_format
        self.quality = quality
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._pending: set[Future[None]] = set()

    def save(self, image: Image.Image, path: Path) -> None:
        """Queue an image save and apply backpressure when the queue is full."""
        future = self._executor.submit(
            save_image,
            image.copy(),
            path,
            image_format=self.image_format,
            quality=self.quality,
        )
        self._pending.add(future)
        if len(self._pending) >= self.max_pending:
            self._drain(completed_only=True)

    def _drain(self, *, completed_only: bool) -> None:
        if not self._pending:
            return
        if completed_only:
            done, pending = wait(self._pending, return_when=FIRST_COMPLETED)
        else:
            done, pending = wait(self._pending)
        self._pending = set(pending)
        for future in done:
            future.result()

    def close(self) -> None:
        """Wait for all queued writes and release worker threads."""
        try:
            self._drain(completed_only=False)
        finally:
            self._executor.shutdown(wait=True)

    def __enter__(self) -> "AsyncImageWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def make_side_by_side(
    left: Image.Image, right: Image.Image, left_label: str, right_label: str
) -> Image.Image:
    """Create a simple two-panel qualitative grid."""
    width = max(left.width, right.width)
    height = max(left.height, right.height)
    label_h = 28
    canvas = Image.new("RGB", (width * 2, height + label_h), "white")
    canvas.paste(left.resize((width, height)), (0, label_h))
    canvas.paste(right.resize((width, height)), (width, label_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    draw.text((8, 8), left_label, fill="black", font=font)
    draw.text((width + 8, 8), right_label, fill="black", font=font)
    return canvas
