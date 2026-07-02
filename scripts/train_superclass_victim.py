"""Fine-tune a torchvision classifier on a CustomImageNet superclass dataset."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from prompt_attack.data.superclass import (
    build_superclass_image_records,
)
from prompt_attack.utils.io import ensure_dir, write_json


ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_optional_count(value: str) -> int | None:
    if value.lower() == "all":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("count must be positive or 'all'.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mixed_13", choices=("mixed_13", "mixed13"))
    parser.add_argument("--imagenet-root", type=Path, required=True)
    parser.add_argument(
        "--imagenet-info-root",
        type=Path,
        required=True,
        help="Directory containing imagenet_class_index.json, wordnet.is_a.txt, and words.txt.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/victims/mixed13_resnet18"))
    parser.add_argument("--architecture", default="resnet18", choices=("resnet18",))
    parser.add_argument("--weights", default="IMAGENET1K_V1")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--balanced-sampler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-images-per-class", type=parse_optional_count, default=None)
    parser.add_argument("--val-images-per-class", type=parse_optional_count, default=None)
    parser.add_argument("--save-every-epoch", action="store_true")
    return parser.parse_args()


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


class SuperclassImageDataset(Dataset[Any]):
    def __init__(self, records, transform) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.path) as loaded_image:
            image = loaded_image.convert("RGB")
            tensor = self.transform(image)
        return tensor, int(record.class_index)


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_counts(records, num_classes: int) -> list[int]:
    counts = [0 for _ in range(num_classes)]
    for record in records:
        counts[int(record.class_index)] += 1
    return counts


def first_size(value: int | list[int] | tuple[int, ...]) -> int:
    if isinstance(value, int):
        return value
    return int(value[0])


def make_transforms(weights, *, train: bool):
    from torchvision import transforms

    preprocess = weights.transforms()
    mean = getattr(preprocess, "mean", [0.485, 0.456, 0.406])
    std = getattr(preprocess, "std", [0.229, 0.224, 0.225])
    crop_size = first_size(preprocess.crop_size)
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    return preprocess


def preprocess_metadata(weights) -> dict[str, Any]:
    preprocess = weights.transforms()
    return {
        "weights": weights.name,
        "resize_size": first_size(preprocess.resize_size),
        "crop_size": first_size(preprocess.crop_size),
        "mean": list(getattr(preprocess, "mean", [0.485, 0.456, 0.406])),
        "std": list(getattr(preprocess, "std", [0.229, 0.224, 0.225])),
    }


def build_model(args: argparse.Namespace, *, num_classes: int):
    import torch
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18

    weights = None
    weights_enum = getattr(ResNet18_Weights, args.weights)
    if not args.no_pretrained:
        weights = weights_enum
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(torch.device(args.device)), weights_enum


def make_loader(records, transform, args: argparse.Namespace, *, train: bool, num_classes: int):
    from torch.utils.data import DataLoader, WeightedRandomSampler

    dataset = SuperclassImageDataset(records, transform)
    sampler = None
    shuffle = train
    if train and args.balanced_sampler:
        counts = class_counts(records, num_classes)
        weights = [1.0 / max(1, counts[int(record.class_index)]) for record in records]
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(records),
            replacement=True,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=str(args.device).startswith("cuda"),
        persistent_workers=args.workers > 0,
    )


def train_one_epoch(model, loader, optimizer, scaler, args: argparse.Namespace) -> dict[str, float]:
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    device = torch.device(args.device)
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    amp_enabled = bool(args.amp and device.type == "cuda")
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.shape[0]
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu().item())
        total_seen += batch_size
    return {
        "loss": total_loss / max(1, total_seen),
        "accuracy": total_correct / max(1, total_seen),
    }


def evaluate(model, loader, args: argparse.Namespace, *, num_classes: int) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    device = torch.device(args.device)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="val", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            preds = logits.argmax(dim=1)
            batch_size = labels.shape[0]
            total_loss += float(loss.detach().cpu().item()) * batch_size
            total_correct += int((preds == labels).sum().detach().cpu().item())
            total_seen += batch_size
            for label, pred in zip(labels.detach().cpu(), preds.detach().cpu(), strict=True):
                confusion[int(label), int(pred)] += 1

    per_class = []
    for index in range(num_classes):
        row_total = int(confusion[index].sum().item())
        row_correct = int(confusion[index, index].item())
        per_class.append(row_correct / row_total if row_total else 0.0)
    return {
        "loss": total_loss / max(1, total_seen),
        "accuracy": total_correct / max(1, total_seen),
        "per_class_accuracy": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def write_history_row(path: Path, row: dict[str, Any]) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "elapsed_seconds",
        "checkpoint_path",
    ]
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def log_stage(message: str) -> None:
    print(f"[mixed13-train] {message}", flush=True)


def checkpoint_payload(
    *,
    model,
    args: argparse.Namespace,
    categories: list[str],
    mapping,
    train_counts: list[int],
    val_counts: list[int],
    epoch: int,
    val_metrics: dict[str, Any],
    weights,
) -> dict[str, Any]:
    return {
        "architecture": "resnet18",
        "victim_name": "resnet18_mixed13",
        "model_state_dict": model.state_dict(),
        "categories": categories,
        "dataset": mapping.name,
        "superclasses": [
            {
                "index": group.index,
                "synset": group.synset,
                "label": group.label,
                "descendants": list(group.descendants),
            }
            for group in mapping.groups
        ],
        "superclass_mapping_hash": mapping.mapping_hash,
        "preprocess": preprocess_metadata(weights),
        "epoch": epoch,
        "val_accuracy": val_metrics["accuracy"],
        "val_loss": val_metrics["loss"],
        "train_class_counts": train_counts,
        "val_class_counts": val_counts,
        "args": serializable_args(args),
    }


def main() -> None:
    import torch
    from torch import optim

    args = parse_args()
    set_seed(args.seed)
    output_root = args.output_root
    checkpoints_dir = ensure_dir(output_root / "checkpoints")
    metrics_dir = ensure_dir(output_root / "metrics")
    ensure_dir(output_root)

    log_stage(
        "starting "
        f"dataset={args.dataset} output_root={output_root} "
        f"imagenet_root={args.imagenet_root} info_root={args.imagenet_info_root}"
    )
    record_started = time.perf_counter()
    log_stage("building train records")
    train_records, train_mapping = build_superclass_image_records(
        imagenet_root=args.imagenet_root,
        split="train",
        dataset_name=args.dataset,
        info_root=args.imagenet_info_root,
        images_per_superclass=args.train_images_per_class,
    )
    log_stage(f"built {len(train_records):,} train records in {time.perf_counter() - record_started:.1f}s")
    record_started = time.perf_counter()
    log_stage("building val records")
    val_records, val_mapping = build_superclass_image_records(
        imagenet_root=args.imagenet_root,
        split="val",
        dataset_name=args.dataset,
        info_root=args.imagenet_info_root,
        images_per_superclass=args.val_images_per_class,
    )
    log_stage(f"built {len(val_records):,} val records in {time.perf_counter() - record_started:.1f}s")
    if train_mapping.mapping_hash != val_mapping.mapping_hash:
        raise RuntimeError("Train and val mixed_13 mappings differ; check ImageNet metadata.")

    categories = train_mapping.categories
    num_classes = len(categories)
    train_counts = class_counts(train_records, num_classes)
    val_counts = class_counts(val_records, num_classes)
    log_stage(f"class names: {', '.join(categories)}")
    log_stage(f"train class counts: {train_counts}")
    log_stage(f"val class counts: {val_counts}")
    log_stage("building model and dataloaders")
    model, weights_enum = build_model(args, num_classes=num_classes)
    train_loader = make_loader(
        train_records,
        make_transforms(weights_enum, train=True),
        args,
        train=True,
        num_classes=num_classes,
    )
    val_loader = make_loader(
        val_records,
        make_transforms(weights_enum, train=False),
        args,
        train=False,
        num_classes=num_classes,
    )
    log_stage(
        f"loaders ready: train_batches={len(train_loader):,} val_batches={len(val_loader):,} "
        f"batch_size={args.batch_size} workers={args.workers} balanced_sampler={args.balanced_sampler}"
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(
        device="cuda" if str(args.device).startswith("cuda") else "cpu",
        enabled=bool(args.amp and str(args.device).startswith("cuda")),
    )

    write_json(
        metrics_dir / "dataset_summary.json",
        {
            "dataset": train_mapping.name,
            "categories": categories,
            "mapping_hash": train_mapping.mapping_hash,
            "train_record_count": len(train_records),
            "val_record_count": len(val_records),
            "train_class_counts": train_counts,
            "val_class_counts": val_counts,
            "superclasses": [asdict(group) for group in train_mapping.groups],
        },
    )

    started = time.perf_counter()
    best_acc = -1.0
    best_epoch = 0
    best_checkpoint_path = output_root / "best.pt"
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        print(f"\n===== mixed_13 ResNet-18 epoch {epoch}/{args.epochs} =====", flush=True)
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, args)
        val_metrics = evaluate(model, val_loader, args, num_classes=num_classes)
        elapsed = time.perf_counter() - epoch_started
        epoch_checkpoint_path = checkpoints_dir / f"epoch_{epoch:03d}.pt"
        payload = checkpoint_payload(
            model=model,
            args=args,
            categories=categories,
            mapping=train_mapping,
            train_counts=train_counts,
            val_counts=val_counts,
            epoch=epoch,
            val_metrics=val_metrics,
            weights=weights_enum,
        )
        if args.save_every_epoch:
            torch.save(payload, epoch_checkpoint_path)
        if val_metrics["accuracy"] > best_acc:
            best_acc = float(val_metrics["accuracy"])
            best_epoch = epoch
            torch.save(payload, best_checkpoint_path)
            shutil.copy2(best_checkpoint_path, checkpoints_dir / "best.pt")
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "elapsed_seconds": elapsed,
            "checkpoint_path": str(best_checkpoint_path),
        }
        history.append(row)
        write_history_row(metrics_dir / "train_history.csv", row)
        print(
            "epoch "
            f"{epoch}: train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"best_acc={best_acc:.4f}",
            flush=True,
        )

    summary = {
        "dataset": train_mapping.name,
        "architecture": "resnet18",
        "victim_name": "resnet18_mixed13",
        "pretrained": not args.no_pretrained,
        "weights": args.weights,
        "categories": categories,
        "mapping_hash": train_mapping.mapping_hash,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_acc,
        "best_checkpoint_path": str(best_checkpoint_path),
        "elapsed_seconds": time.perf_counter() - started,
        "train_record_count": len(train_records),
        "val_record_count": len(val_records),
        "train_class_counts": train_counts,
        "val_class_counts": val_counts,
        "history": history,
        "args": serializable_args(args),
    }
    write_json(metrics_dir / "summary.json", summary)
    print(f"\nSaved best checkpoint: {best_checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
