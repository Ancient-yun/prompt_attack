"""Validate an ImageNet-1K folder dataset layout."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EXPECTED_CLASS_COUNT = 1000
EXPECTED_VAL_IMAGES_PER_CLASS = 50


@dataclass(frozen=True)
class SplitReport:
    split: str
    class_count: int
    image_count: int
    extension_counts: dict[str, int]
    zero_byte_count: int
    non_image_count: int
    min_class: tuple[str, int] | None
    max_class: tuple[str, int] | None
    classes_not_50: int | None = None


@dataclass(frozen=True)
class DecodeReport:
    mode: str
    checked: int
    bad_count: int
    bad_samples: list[dict[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--verify-images",
        choices=("none", "val", "sample", "train-full"),
        default="none",
    )
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260521)
    return parser.parse_args()


def class_dirs(split_dir: Path) -> list[Path]:
    """Return sorted ImageNet synset directories for one split."""
    return sorted(Path(entry.path) for entry in os.scandir(split_dir) if entry.is_dir())


def iter_image_files(split_dir: Path) -> list[Path]:
    """Return all image files under a split."""
    images: list[Path] = []
    for class_dir in class_dirs(split_dir):
        images.extend(
            Path(entry.path)
            for entry in os.scandir(class_dir)
            if entry.is_file() and Path(entry.name).suffix.lower() in IMAGE_EXTENSIONS
        )
    return images


def inspect_split(root: Path, split: str) -> tuple[SplitReport, list[str]]:
    """Inspect split structure and return a report plus validation errors."""
    split_dir = root / split
    errors: list[str] = []
    if not split_dir.exists():
        return (
            SplitReport(
                split=split,
                class_count=0,
                image_count=0,
                extension_counts={},
                zero_byte_count=0,
                non_image_count=0,
                min_class=None,
                max_class=None,
            ),
            [f"Missing split directory: {split_dir}"],
        )

    counts: list[tuple[str, int]] = []
    extension_counts: Counter[str] = Counter()
    zero_byte_count = 0
    non_image_count = 0
    for class_dir in class_dirs(split_dir):
        image_count = 0
        for entry in os.scandir(class_dir):
            if not entry.is_file():
                continue
            suffix = Path(entry.name).suffix.lower()
            if suffix not in IMAGE_EXTENSIONS:
                non_image_count += 1
                continue
            image_count += 1
            extension_counts[suffix] += 1
            if entry.stat().st_size == 0:
                zero_byte_count += 1
        counts.append((class_dir.name, image_count))

    class_count = len(counts)
    image_count = sum(count for _, count in counts)
    min_class = min(counts, key=lambda item: item[1]) if counts else None
    max_class = max(counts, key=lambda item: item[1]) if counts else None
    classes_not_50 = (
        sum(1 for _, count in counts if count != EXPECTED_VAL_IMAGES_PER_CLASS)
        if split == "val"
        else None
    )

    if class_count != EXPECTED_CLASS_COUNT:
        errors.append(f"{split}: expected {EXPECTED_CLASS_COUNT} class dirs, found {class_count}")
    if image_count == 0:
        errors.append(f"{split}: found no images")
    if zero_byte_count:
        errors.append(f"{split}: found {zero_byte_count} zero-byte image files")
    if non_image_count:
        errors.append(f"{split}: found {non_image_count} non-image files")
    if split == "val" and classes_not_50:
        errors.append(
            f"val: expected {EXPECTED_VAL_IMAGES_PER_CLASS} images per class, "
            f"{classes_not_50} classes differ"
        )

    return (
        SplitReport(
            split=split,
            class_count=class_count,
            image_count=image_count,
            extension_counts=dict(extension_counts),
            zero_byte_count=zero_byte_count,
            non_image_count=non_image_count,
            min_class=min_class,
            max_class=max_class,
            classes_not_50=classes_not_50,
        ),
        errors,
    )


def verify_decode(paths: list[Path], *, mode: str) -> DecodeReport:
    """Open image files with Pillow and report decode failures."""
    from PIL import Image

    bad_samples: list[dict[str, str]] = []
    bad_count = 0
    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:  # pragma: no cover - depends on external data
            bad_count += 1
            if len(bad_samples) < 10:
                bad_samples.append({"path": str(path), "error": str(exc)})
    return DecodeReport(
        mode=mode,
        checked=len(paths),
        bad_count=bad_count,
        bad_samples=bad_samples,
    )


def select_sample(root: Path, *, sample_size: int, seed: int) -> list[Path]:
    """Select a deterministic train image sample with class coverage."""
    rng = random.Random(seed)
    train_dir = root / "train"
    selected: list[Path] = []
    per_class = max(1, (sample_size + EXPECTED_CLASS_COUNT - 1) // EXPECTED_CLASS_COUNT)
    for class_dir in class_dirs(train_dir):
        class_images = sorted(
            Path(entry.path)
            for entry in os.scandir(class_dir)
            if entry.is_file() and Path(entry.name).suffix.lower() in IMAGE_EXTENSIONS
        )
        if not class_images:
            continue
        rng.shuffle(class_images)
        selected.extend(class_images[:per_class])
        if len(selected) >= sample_size:
            return selected[:sample_size]
    return selected


def validate_synsets(root: Path) -> tuple[dict[str, int], list[str]]:
    """Check that top-level train tars and train/val directories describe the same classes."""
    tars = {path.stem for path in root.glob("*.tar")}
    train = {path.name for path in class_dirs(root / "train")}
    val = {path.name for path in class_dirs(root / "val")}
    errors: list[str] = []
    if tars and len(tars) != EXPECTED_CLASS_COUNT:
        errors.append(f"expected {EXPECTED_CLASS_COUNT} top-level class tar files, found {len(tars)}")
    if tars and tars != train:
        errors.append("top-level class tar names do not match train class dirs")
    if train != val:
        errors.append("train class dirs do not match val class dirs")
    return (
        {
            "tar_count": len(tars),
            "tar_minus_train": len(tars - train),
            "train_minus_tar": len(train - tars) if tars else 0,
            "train_minus_val": len(train - val),
            "val_minus_train": len(val - train),
        },
        errors,
    )


def main() -> None:
    args = parse_args()
    root = args.root
    errors: list[str] = []
    if not root.exists():
        raise SystemExit(f"ImageNet root not found: {root}")

    train_report, train_errors = inspect_split(root, "train")
    val_report, val_errors = inspect_split(root, "val")
    synset_report, synset_errors = validate_synsets(root)
    errors.extend(train_errors)
    errors.extend(val_errors)
    errors.extend(synset_errors)

    decode_report = None
    if args.verify_images == "val":
        decode_report = verify_decode(iter_image_files(root / "val"), mode="val")
    elif args.verify_images == "sample":
        decode_report = verify_decode(
            select_sample(root, sample_size=args.sample_size, seed=args.seed),
            mode="sample",
        )
    elif args.verify_images == "train-full":
        decode_report = verify_decode(iter_image_files(root / "train"), mode="train-full")
    if decode_report is not None and decode_report.bad_count:
        errors.append(
            f"{decode_report.mode}: found {decode_report.bad_count} image decode failures"
        )

    report = {
        "root": str(root),
        "train": asdict(train_report),
        "val": asdict(val_report),
        "synsets": synset_report,
        "decode": None if decode_report is None else asdict(decode_report),
        "errors": errors,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
