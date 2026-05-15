"""Output persistence helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, cast


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    """Append one row to a CSV file, writing a header when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_json(path: Path, value: Any) -> None:
    """Write JSON with dataclass support."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(cast(Any, value))
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=str)
