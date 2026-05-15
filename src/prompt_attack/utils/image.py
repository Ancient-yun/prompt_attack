"""Image tensor conversion and grid helpers."""

from __future__ import annotations

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


def save_image(image: Image.Image, path: Path) -> None:
    """Save an image and create its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def make_side_by_side(left: Image.Image, right: Image.Image, left_label: str, right_label: str) -> Image.Image:
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

