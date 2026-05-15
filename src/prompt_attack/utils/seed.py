"""Reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch if available."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        return


def stable_image_seed(base_seed: int, image_id: str) -> int:
    """Create a stable integer seed for an image id."""
    value = base_seed
    for char in image_id:
        value = (value * 131 + ord(char)) % (2**31 - 1)
    return value

