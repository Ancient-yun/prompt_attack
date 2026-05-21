"""ImageNet subset discovery and loading."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image

from prompt_attack.config import DataConfig


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG"}


@dataclass(frozen=True)
class FixedClass:
    synset: str
    label: str


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    synset: str
    class_label: str
    class_index: int
    image_id: str


FIXED_10_CLASSES: tuple[FixedClass, ...] = (
    FixedClass("n01685808", "whiptail lizard"),
    FixedClass("n02113624", "toy poodle"),
    FixedClass("n02640242", "sturgeon fish"),
    FixedClass("n02802426", "basketball"),
    FixedClass("n03028079", "church building"),
    FixedClass("n03141823", "crutch"),
    FixedClass("n04141076", "saxophone"),
    FixedClass("n04461696", "tow truck"),
    FixedClass("n04599235", "wool"),
    FixedClass("n12267677", "acorn"),
)


def split_dir(config: DataConfig) -> Path:
    """Return the ImageNet split directory."""
    path = config.imagenet_root if not config.split else config.imagenet_root / config.split
    if not path.exists():
        raise FileNotFoundError(f"ImageNet split directory not found: {path}")
    return path


def class_index_map(split_path: Path) -> dict[str, int]:
    """Map ImageNet synset folder names to torchvision-compatible class indices."""
    synsets = sorted(p.name for p in split_path.iterdir() if p.is_dir())
    if len(synsets) != 1000:
        raise ValueError(f"Expected 1000 ImageNet class folders in {split_path}, found {len(synsets)}")
    return {synset: idx for idx, synset in enumerate(synsets)}


def selected_classes(config: DataConfig) -> tuple[FixedClass, ...]:
    """Return the configured ImageNet class subset."""
    if config.class_mode != "fixed_10":
        raise ValueError(f"Unsupported class_mode: {config.class_mode}")
    return FIXED_10_CLASSES


def list_class_images(class_dir: Path) -> list[Path]:
    """List image files for one ImageNet synset directory."""
    images = [p for p in class_dir.iterdir() if p.is_file() and p.suffix in IMAGE_EXTENSIONS]
    return sorted(images)


@lru_cache(maxsize=1)
def imagenet_categories() -> tuple[str, ...]:
    """Return torchvision's ImageNet-1K category names."""
    from torchvision.models import ResNet18_Weights

    return tuple(ResNet18_Weights.IMAGENET1K_V1.meta["categories"])


def build_fixed_10_records(config: DataConfig) -> list[ImageRecord]:
    """Build records from ImageNet synset folders."""
    root = split_dir(config)
    indices = class_index_map(root)
    records: list[ImageRecord] = []
    max_candidates = config.images_per_class * max(config.candidate_multiplier, 1)

    for cls in selected_classes(config):
        class_dir = root / cls.synset
        if not class_dir.exists():
            raise FileNotFoundError(f"Configured synset not found: {class_dir}")
        class_idx = indices[cls.synset]
        for image_path in list_class_images(class_dir)[:max_candidates]:
            records.append(
                ImageRecord(
                    path=image_path,
                    synset=cls.synset,
                    class_label=cls.label,
                    class_index=class_idx,
                    image_id=image_path.stem,
                )
            )
    return records


def build_csv_image_records(config: DataConfig) -> list[ImageRecord]:
    """Build records from an images.csv plus images/{ImageId}.png dataset."""
    root = split_dir(config)
    csv_path = root / "images.csv"
    image_dir = root / "images"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV metadata not found: {csv_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"CSV image directory not found: {image_dir}")

    categories = imagenet_categories()
    records: list[ImageRecord] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = str(row["ImageId"])
            class_idx = int(row["TrueLabel"]) - 1
            if class_idx < 0 or class_idx >= len(categories):
                raise ValueError(f"TrueLabel out of ImageNet-1K range for {image_id}: {class_idx}")
            image_path = image_dir / f"{image_id}.png"
            if not image_path.exists():
                raise FileNotFoundError(f"Image listed in CSV not found: {image_path}")
            records.append(
                ImageRecord(
                    path=image_path,
                    synset=f"class_{class_idx:04d}",
                    class_label=categories[class_idx],
                    class_index=class_idx,
                    image_id=image_id,
                )
            )
    return records


def build_imagenet_folder_records(config: DataConfig) -> list[ImageRecord]:
    """Build records from an ImageNet split laid out as split/{synset}/*.JPEG."""
    root = split_dir(config)
    indices = class_index_map(root)
    categories = imagenet_categories()
    records: list[ImageRecord] = []
    max_candidates = config.images_per_class * max(config.candidate_multiplier, 1)

    for synset, class_idx in sorted(indices.items(), key=lambda item: item[1]):
        class_dir = root / synset
        for image_path in list_class_images(class_dir)[:max_candidates]:
            records.append(
                ImageRecord(
                    path=image_path,
                    synset=synset,
                    class_label=categories[class_idx],
                    class_index=class_idx,
                    image_id=image_path.stem,
                )
            )
    return records


def build_candidate_records(config: DataConfig) -> list[ImageRecord]:
    """Build candidate records before clean-correct filtering."""
    if config.class_mode == "fixed_10":
        return build_fixed_10_records(config)
    if config.class_mode == "csv_images":
        return build_csv_image_records(config)
    if config.class_mode == "imagenet_folder":
        return build_imagenet_folder_records(config)
    raise ValueError(f"Unsupported class_mode: {config.class_mode}")


def load_image(path: Path) -> Image.Image:
    """Load an RGB PIL image."""
    with Image.open(path) as image:
        return image.convert("RGB")
