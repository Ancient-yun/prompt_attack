"""Inspect the configured ImageNet subset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from prompt_attack.config import load_config
from prompt_attack.data.imagenet import build_candidate_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    records = build_candidate_records(config.data)
    if args.output is None:
        for record in records[:50]:
            print(f"{record.synset},{record.class_index},{record.class_label},{record.path}")
        print(f"records={len(records)}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_id", "synset", "class_index", "class_label", "path"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "image_id": record.image_id,
                    "synset": record.synset,
                    "class_index": record.class_index,
                    "class_label": record.class_label,
                    "path": str(record.path),
                }
            )


if __name__ == "__main__":
    main()

