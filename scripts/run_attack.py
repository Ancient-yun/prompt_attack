"""Run the configured soft-token attack."""

from __future__ import annotations

import argparse
from pathlib import Path

from prompt_attack.attacks.runner import SoftTokenAttackRunner
from prompt_attack.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runner = SoftTokenAttackRunner(config, device=args.device)
    runner.run(max_images=args.max_images)


if __name__ == "__main__":
    main()
