"""Run a small architecture smoke test.

By default this uses the differentiable mock generator. Pass --real-generator
to test the FLUX.2 differentiable prompt-embedding path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from prompt_attack.attacks.runner import SoftTokenAttackRunner
from prompt_attack.config import load_config, with_smoke_overrides
from prompt_attack.generators.factory import build_generator
from prompt_attack.models.semantic import build_semantic_model
from prompt_attack.models.victim import build_victim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--real-generator", action="store_true")
    parser.add_argument("--imports-only", action="store_true")
    parser.add_argument("--max-images", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = with_smoke_overrides(load_config(args.config), use_mock_generator=not args.real_generator)

    if args.imports_only:
        build_victim(config.victim.name, weights=config.victim.weights, device=args.device)
        build_semantic_model(config.semantic.name, device=args.device)
        build_generator(config.generator, device=args.device)
        print("imports-ok")
        return

    runner = SoftTokenAttackRunner(config, device=args.device)
    rows = runner.run(max_images=args.max_images)
    print(f"smoke-ok rows={len(rows)}")


if __name__ == "__main__":
    main()
