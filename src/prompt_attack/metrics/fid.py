"""FID computation through the vendored pytorch-fid adapter."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from prompt_attack.config import FIDConfig


def _load_image_array(path: Path, *, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, resample=Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array.transpose(2, 0, 1)


def _load_clean_adv_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    clean_images: list[np.ndarray] = []
    adv_images: list[np.ndarray] = []
    for row in rows:
        adv_path = Path(str(row["output_image_path"]))
        original_path = Path(str(row["original_image_path"]))
        adv_image = Image.open(adv_path).convert("RGB")
        adv_size = adv_image.size
        adv_array = np.asarray(adv_image, dtype=np.float32) / 255.0
        adv_images.append(adv_array.transpose(2, 0, 1))
        clean_images.append(_load_image_array(original_path, size=adv_size))
    return (
        np.stack(adv_images, axis=0).astype(np.float32),
        np.stack(clean_images, axis=0).astype(np.float32),
    )


def _resolve_fid_root(fid_root: Path) -> Path:
    if fid_root.is_absolute():
        return fid_root
    candidates = [
        Path.cwd() / fid_root,
        Path(__file__).resolve().parents[3] / fid_root,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def compute_fid_for_rows(rows: list[dict[str, Any]], config: FIDConfig) -> float | None:
    """Compute FID between generated adversarial images and aligned originals."""
    if not config.enabled or not rows:
        return None
    fid_root = _resolve_fid_root(config.fid_root)
    if not fid_root.exists():
        message = f"FID implementation root does not exist: {fid_root}"
        if config.fail_on_error:
            raise RuntimeError(message)
        return None

    import_root = fid_root.parent
    sys.path.insert(0, str(import_root))
    try:
        import torch
        from pytorch_fid.fid_score import calculate_fid_given_data_list

        all_adv_images, all_clean_images = _load_clean_adv_arrays(rows)
        fid = calculate_fid_given_data_list(
            data_list=[all_adv_images, all_clean_images],
            batch_size=config.batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dims=config.dims,
        )
        return float(fid)
    except Exception as exc:  # pragma: no cover - model/backend dependent
        if config.fail_on_error:
            raise RuntimeError(f"Failed to compute FID: {exc}") from exc
        return None
    finally:
        try:
            sys.path.remove(str(import_root))
        except ValueError:
            pass
