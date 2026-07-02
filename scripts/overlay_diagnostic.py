"""Overlay diagnostic: is the learned universal token a fixed UAP overlay?

Uses already-saved adversarial PNGs (no FLUX / GPU needed). Two tests:
  (A) residual coherence: resid = adv - original. If the token is a fixed overlay,
      residuals across unrelated images are highly correlated (high mean pairwise
      cosine) and the mean residual carries most of the energy.
  (B) adv self-similarity: adv images from different source classes are unusually
      similar to each other (shared texture) vs how similar the originals are.

Run from repo root:
  python scripts/overlay_diagnostic.py \
    --run outputs/uap_mixed13/mixed13_resnet18_t32_mdino_ipc20_gb16_e10 \
    --imagenet-root E:/ImageNet --max 160
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--imagenet-root", type=Path, action="append", default=[])
    p.add_argument("--size", type=int, default=256, help="analysis resolution")
    p.add_argument("--max", type=int, default=200)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def find_original(op: str, roots: list[Path]) -> Path | None:
    m = re.search(r"/(val|train)/(n\d{8})/([^/]+)$", op.replace("\\", "/"))
    if not m:
        return None
    split, syn, fn = m.groups()
    stem = Path(fn).stem
    for root in roots:
        for suffix in (Path(fn).suffix, ".JPEG", ".jpg", ".jpeg", ".png"):
            cand = root / split / syn / f"{stem}{suffix}"
            if cand.exists():
                return cand
    return None


def load_arr(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0  # H,W,3 in [0,1]


def mean_pairwise_cosine(mat: np.ndarray, cap: int = 120) -> float:
    """Mean off-diagonal cosine similarity of row vectors (subsampled)."""
    if len(mat) > cap:
        idx = np.linspace(0, len(mat) - 1, cap).astype(int)
        mat = mat[idx]
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    unit = mat / norm
    sim = unit @ unit.T
    n = len(sim)
    off = (sim.sum() - np.trace(sim)) / (n * (n - 1))
    return float(off)


def main() -> None:
    args = parse_args()
    roots = [r for r in args.imagenet_root if r.exists()]
    rows = list(csv.DictReader(open(args.run / "metrics/test_results.csv", encoding="utf-8-sig")))

    resids = []          # flattened adv-original residuals
    advs = []            # flattened adv
    origs = []           # flattened original
    labels = []
    success_flags = []
    matched = 0
    for r in rows:
        if len(resids) >= args.max:
            break
        adv_path = args.run / "images/test" / r["class_label"] / r["image_id"] / "adv.png"
        if not adv_path.exists():
            continue
        orig_path = find_original(r.get("original_image_path", ""), roots)
        if orig_path is None:
            continue
        adv = load_arr(adv_path, args.size)
        orig = load_arr(orig_path, args.size)
        advs.append(adv.reshape(-1))
        origs.append(orig.reshape(-1))
        resids.append((adv - orig).reshape(-1))
        labels.append(r["class_label"])
        success_flags.append(str(r.get("success", "")).lower() in ("1", "true", "yes"))
        matched += 1

    if matched < 5:
        print(f"Only matched {matched} originals; check --imagenet-root. Roots tried: {roots}")
        return

    R = np.stack(resids)   # N, D
    A = np.stack(advs)
    O = np.stack(origs)
    N = len(R)

    # (A) residual coherence
    mean_resid = R.mean(axis=0)
    energy_total = float((R ** 2).mean())
    energy_mean = float((mean_resid ** 2).mean())
    explained = energy_mean / energy_total if energy_total > 0 else 0.0
    resid_cos = mean_pairwise_cosine(R)

    # (B) self-similarity of adv vs originals (mean-centered so we measure structure overlap)
    adv_cos = mean_pairwise_cosine(A - A.mean(axis=0))
    orig_cos = mean_pairwise_cosine(O - O.mean(axis=0))

    print("=" * 64)
    print(f"Overlay diagnostic  |  matched pairs: {N}  |  success in set: {sum(success_flags)}")
    print("=" * 64)
    print("\n(A) Residual (adv - original) coherence")
    print(f"    mean pairwise cosine of residuals : {resid_cos:+.3f}")
    print(f"    fraction of residual energy in the")
    print(f"    single shared mean residual       : {explained:6.1%}")
    print("      -> overlay if BOTH are high (residuals point the same way,")
    print("         and one common pattern explains most of the change).")
    print("\n(B) Adv self-similarity vs original self-similarity (mean-centered)")
    print(f"    mean pairwise cosine, ADV images  : {adv_cos:+.3f}")
    print(f"    mean pairwise cosine, ORIG images : {orig_cos:+.3f}")
    print(f"    ratio adv/orig                    : {adv_cos / orig_cos if orig_cos else float('nan'):.2f}x")
    print("      -> overlay if adv images are much more alike than originals.")

    # save the mean residual as a viewable image (amplified)
    out = args.out or (args.run / "overlay_diagnostic")
    out.mkdir(parents=True, exist_ok=True)
    mr = mean_resid.reshape(args.size, args.size, 3)
    amp = np.clip(0.5 + mr * 4.0, 0, 1)  # center at gray, amplify 4x
    Image.fromarray((amp * 255).astype(np.uint8)).save(out / "mean_residual_amp4x.png")
    # also a tiled montage of first 16 raw residuals amplified
    tile = []
    for i in range(min(16, N)):
        ri = R[i].reshape(args.size, args.size, 3)
        tile.append(np.clip(0.5 + ri * 4.0, 0, 1))
    if tile:
        cols = 4
        rows_n = (len(tile) + cols - 1) // cols
        canvas = np.ones((rows_n * args.size, cols * args.size, 3))
        for i, t in enumerate(tile):
            rr, cc = divmod(i, cols)
            canvas[rr * args.size:(rr + 1) * args.size, cc * args.size:(cc + 1) * args.size] = t
        Image.fromarray((canvas * 255).astype(np.uint8)).save(out / "residual_montage_amp4x.png")
    print(f"\nSaved mean residual + montage to {out}")


if __name__ == "__main__":
    main()
