"""Evaluate classifier-independent image collapse from attack CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from prompt_attack.metrics.collapse import (
    CollapseThresholds,
    annotate_collapse_row,
    calibrate_thresholds,
    metric_value,
    summarize_collapse_rows,
)
from prompt_attack.metrics.perceptual import (
    DreamSimDistance,
    ImagePair,
    LPIPSDistance,
    SemanticImageSimilarity,
)

FIXED10_LABEL_TO_SYNSET = {
    "whiptail lizard": "n01685808",
    "toy poodle": "n02113624",
    "sturgeon fish": "n02640242",
    "basketball": "n02802426",
    "church building": "n03028079",
    "crutch": "n03141823",
    "saxophone": "n04141076",
    "tow truck": "n04461696",
    "wool": "n04599235",
    "acorn": "n12267677",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate semantic-preservation thresholds from benign edits and annotate "
            "attack results with SP-ASR / image-collapse metrics."
        )
    )
    parser.add_argument(
        "--attack-csv",
        action="append",
        type=Path,
        required=True,
        help="Attack metrics CSV. May be passed multiple times.",
    )
    parser.add_argument(
        "--benign-csv",
        action="append",
        type=Path,
        default=[],
        help="Benign-edit metrics CSV used for threshold calibration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for collapse_rows.csv, summaries, and thresholds.json.",
    )
    parser.add_argument(
        "--run-label",
        action="append",
        default=[],
        help="Optional run label for each --attack-csv, in the same order.",
    )
    parser.add_argument("--dino-quantile", type=float, default=0.05)
    parser.add_argument("--clip-quantile", type=float, default=0.05)
    parser.add_argument("--dreamsim-quantile", type=float, default=0.95)
    parser.add_argument("--dino-threshold", type=float)
    parser.add_argument("--clip-threshold", type=float)
    parser.add_argument("--dreamsim-threshold", type=float)
    parser.add_argument(
        "--compute-missing-metrics",
        nargs="*",
        choices=["dino", "clip", "dreamsim", "lpips", "all"],
        default=[],
        help="Compute missing image-pair metrics from original/output image paths.",
    )
    parser.add_argument(
        "--metric-device",
        default="auto",
        help="Device for DreamSim/LPIPS metric computation. Use 'auto', 'cuda', or 'cpu'.",
    )
    parser.add_argument("--metric-batch-size", type=int, default=8)
    parser.add_argument(
        "--dreamsim-cache-dir",
        type=Path,
        default=Path(".cache/dreamsim"),
        help="Directory for DreamSim weights.",
    )
    parser.add_argument("--dreamsim-type", default="ensemble")
    parser.add_argument(
        "--max-benign-rows",
        type=int,
        help="Optional row limit for quick calibration/debug runs.",
    )
    parser.add_argument(
        "--max-attack-rows",
        type=int,
        help="Optional row limit for quick attack/debug runs.",
    )
    parser.add_argument(
        "--imagenet-root",
        action="append",
        type=Path,
        default=[Path("E:/ImageNet")],
        help="Local ImageNet roots used to translate remote original image paths.",
    )
    parser.add_argument(
        "--group-by",
        nargs="*",
        default=["class_label"],
        help="Fields for grouped summaries. run_label is always included.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def requested_metrics(values: Sequence[str]) -> set[str]:
    requested = {value.lower() for value in values}
    if "all" in requested:
        return {"dino", "clip", "dreamsim", "lpips"}
    return requested


def default_run_label(path: Path) -> str:
    if path.parent.name == "metrics" and path.parent.parent.name:
        return path.parent.parent.name
    return path.stem


def calibrate_from_args(
    args: argparse.Namespace,
    benign_rows: Sequence[Mapping[str, Any]],
) -> tuple[CollapseThresholds, dict[str, Any]]:
    thresholds = (
        calibrate_thresholds(
            benign_rows,
            dino_quantile=args.dino_quantile,
            clip_quantile=args.clip_quantile,
            dreamsim_quantile=args.dreamsim_quantile,
        )
        if benign_rows
        else CollapseThresholds(source="manual")
    )

    override_used = any(
        value is not None
        for value in (args.dino_threshold, args.clip_threshold, args.dreamsim_threshold)
    )
    thresholds = thresholds.with_overrides(
        dino_similarity=args.dino_threshold,
        clip_image_similarity=args.clip_threshold,
        dreamsim_distance=args.dreamsim_threshold,
        source="benign_edit_quantile+manual_override" if benign_rows and override_used else None,
    )
    if not thresholds.active_metrics():
        raise SystemExit(
            "No collapse thresholds are available. Pass --benign-csv or at least one "
            "manual threshold flag."
        )

    calibration = {
        "benign_csv": [str(path) for path in args.benign_csv],
        "benign_row_count": len(benign_rows),
        "dino_quantile": args.dino_quantile,
        "clip_quantile": args.clip_quantile,
        "dreamsim_quantile": args.dreamsim_quantile,
        **thresholds.as_dict(),
    }
    return thresholds, calibration


def grouped_summary_rows(
    rows: Sequence[Mapping[str, Any]],
    thresholds: CollapseThresholds,
    group_by: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    keys = ["run_label", *[key for key in group_by if key != "run_label"]]
    for row in rows:
        group_key = tuple(str(row.get(key, "")) for key in keys)
        groups[group_key].append(row)

    summaries: list[dict[str, Any]] = []
    for values, group_rows in sorted(groups.items()):
        prefix = dict(zip(keys, values, strict=True))
        summaries.append({**prefix, **summarize_collapse_rows(group_rows, thresholds)})
    return summaries


def label_distribution_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_run: dict[str, Counter[str]] = defaultdict(Counter)
    by_run_success: dict[str, Counter[str]] = defaultdict(Counter)
    run_totals: Counter[str] = Counter()
    run_success_totals: Counter[str] = Counter()
    for row in rows:
        run = str(row.get("run_label", ""))
        label = str(row.get("adv_pred_label") or row.get("adv_pred") or "")
        if not label:
            continue
        by_run[run][label] += 1
        run_totals[run] += 1
        if str(row.get("collapse_attack_success", "")).lower() == "true":
            by_run_success[run][label] += 1
            run_success_totals[run] += 1

    output: list[dict[str, Any]] = []
    for run, counts in sorted(by_run.items()):
        for label, count in counts.most_common():
            success_count = by_run_success[run][label]
            output.append(
                {
                    "run_label": run,
                    "adv_pred_label": label,
                    "count": count,
                    "share": count / run_totals[run] if run_totals[run] else 0.0,
                    "success_count": success_count,
                    "success_share": (
                        success_count / run_success_totals[run] if run_success_totals[run] else 0.0
                    ),
                }
            )
    return output


def _path_candidates(raw: str, csv_path: Path) -> list[Path]:
    path = Path(raw)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend(
            [
                Path.cwd() / path,
                csv_path.parent / path,
                csv_path.parent.parent / path,
                csv_path.parent.parent.parent / path,
            ]
        )
    normalized = raw.replace("\\", "/")
    if "outputs/" in normalized:
        candidates.append(Path.cwd() / normalized[normalized.index("outputs/") :])
    return candidates


def _imagenet_candidates(raw: str, roots: Sequence[Path]) -> list[Path]:
    normalized = raw.replace("\\", "/")
    match = re.search(r"/(train|val)/(n\d{8})/([^/]+)$", normalized)
    if match is None:
        return []
    split, synset, filename = match.groups()
    return [root / split / synset / filename for root in roots]


def _fixed10_imagenet_candidates(
    row: Mapping[str, Any],
    *,
    roots: Sequence[Path],
) -> list[Path]:
    label = str(row.get("class_label", "")).strip()
    image_id = str(row.get("image_id", "")).strip()
    synset = FIXED10_LABEL_TO_SYNSET.get(label)
    if not synset or not image_id:
        return []
    candidates: list[Path] = []
    for root in roots:
        for suffix in (".JPEG", ".jpg", ".jpeg", ".png"):
            candidates.append(root / "val" / synset / f"{image_id}{suffix}")
    return candidates


def resolve_path(raw: object, *, csv_path: Path, imagenet_roots: Sequence[Path]) -> Path | None:
    if raw is None or raw == "":
        return None
    text = str(raw)
    for candidate in [*_path_candidates(text, csv_path), *_imagenet_candidates(text, imagenet_roots)]:
        if candidate.exists():
            return candidate
    return None


def load_summary_original_map(csv_path: Path) -> dict[str, str]:
    summary_path = csv_path.parent / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    selected = data.get("selected_images", [])
    if not isinstance(selected, list):
        return {}
    output: dict[str, str] = {}
    for item in selected:
        if not isinstance(item, dict):
            continue
        image_id = item.get("image_id")
        path = item.get("path")
        if image_id and path:
            output[str(image_id)] = str(path)
    return output


def resolve_pair(
    row: Mapping[str, Any],
    *,
    csv_path: Path,
    imagenet_roots: Sequence[Path],
    summary_originals: Mapping[str, str],
) -> ImagePair | None:
    generated_raw = (
        row.get("output_image_path")
        or row.get("image_path")
        or row.get("adv_image_path")
        or row.get("generated_image_path")
    )
    original_raw = (
        row.get("original_image_path")
        or row.get("source_image_path")
        or row.get("input_image_path")
        or summary_originals.get(str(row.get("image_id", "")))
    )
    original = resolve_path(original_raw, csv_path=csv_path, imagenet_roots=imagenet_roots)
    generated = resolve_path(generated_raw, csv_path=csv_path, imagenet_roots=imagenet_roots)
    if original is None:
        for candidate in _fixed10_imagenet_candidates(row, roots=imagenet_roots):
            if candidate.exists():
                original = candidate
                break
    if original is None or generated is None:
        return None
    return ImagePair(original=original, generated=generated)


def metric_device(value: str) -> str | None:
    normalized = value.strip().lower()
    if normalized == "auto":
        return None
    return normalized


def fill_missing_pair_metrics(
    rows: Sequence[dict[str, Any]],
    *,
    csv_path: Path,
    metrics: set[str],
    device: str | None,
    batch_size: int,
    dreamsim_cache_dir: Path,
    dreamsim_type: str,
    imagenet_roots: Sequence[Path],
) -> dict[str, int]:
    if not metrics:
        return {"resolved_pairs": 0, "missing_pairs": 0, "dreamsim_computed": 0, "lpips_computed": 0}

    summary_originals = load_summary_original_map(csv_path)
    resolved: list[tuple[int, ImagePair]] = []
    missing_pairs = 0
    for index, row in enumerate(rows):
        pair = resolve_pair(
            row,
            csv_path=csv_path,
            imagenet_roots=imagenet_roots,
            summary_originals=summary_originals,
        )
        if pair is None:
            missing_pairs += 1
            continue
        resolved.append((index, pair))

    stats = {
        "resolved_pairs": len(resolved),
        "missing_pairs": missing_pairs,
        "dino_computed": 0,
        "clip_computed": 0,
        "dreamsim_computed": 0,
        "lpips_computed": 0,
    }
    if not resolved:
        return stats

    if "dino" in metrics:
        targets = [
            (index, pair) for index, pair in resolved if metric_value(rows[index], "dino") is None
        ]
        if targets:
            dino_metric = SemanticImageSimilarity("dinov2_vitb14", device=device)
            values = dino_metric.similarities([pair for _, pair in targets], batch_size=batch_size)
            for (index, _), value in zip(targets, values, strict=True):
                rows[index]["dino_similarity"] = value
            stats["dino_computed"] = len(targets)

    if "clip" in metrics:
        targets = [
            (index, pair) for index, pair in resolved if metric_value(rows[index], "clip") is None
        ]
        if targets:
            clip_metric = SemanticImageSimilarity("clip_vit_b32", device=device)
            values = clip_metric.similarities([pair for _, pair in targets], batch_size=batch_size)
            for (index, _), value in zip(targets, values, strict=True):
                rows[index]["clip_image_similarity"] = value
            stats["clip_computed"] = len(targets)

    if "dreamsim" in metrics:
        targets = [
            (index, pair)
            for index, pair in resolved
            if metric_value(rows[index], "dreamsim") is None
        ]
        if targets:
            dreamsim_metric = DreamSimDistance(
                device=device,
                cache_dir=dreamsim_cache_dir,
                dreamsim_type=dreamsim_type,
            )
            values = dreamsim_metric.distances([pair for _, pair in targets], batch_size=batch_size)
            for (index, _), value in zip(targets, values, strict=True):
                rows[index]["dreamsim_distance"] = value
            stats["dreamsim_computed"] = len(targets)

    if "lpips" in metrics:
        targets = [
            (index, pair) for index, pair in resolved if metric_value(rows[index], "lpips") is None
        ]
        if targets:
            lpips_metric = LPIPSDistance(device=device)
            values = lpips_metric.distances([pair for _, pair in targets], batch_size=batch_size)
            for (index, _), value in zip(targets, values, strict=True):
                rows[index]["lpips_distance"] = value
            stats["lpips_computed"] = len(targets)

    return stats


def add_metric_presence(summary: dict[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = dict(summary)
    for metric, key in (
        ("dino", "dino_value_count"),
        ("clip", "clip_value_count"),
        ("dreamsim", "dreamsim_value_count"),
        ("ssim", "ssim_value_count"),
        ("lpips", "lpips_value_count"),
    ):
        result[key] = sum(1 for row in rows if metric_value(row, metric) is not None)
    return result


def main() -> None:
    args = parse_args()
    if args.run_label and len(args.run_label) != len(args.attack_csv):
        raise SystemExit("--run-label must be supplied once for each --attack-csv.")

    metrics_to_compute = requested_metrics(args.compute_missing_metrics)
    device = metric_device(args.metric_device)
    imagenet_roots = [path for path in args.imagenet_root if path.exists()]

    benign_rows: list[dict[str, Any]] = []
    metric_compute_stats: dict[str, Any] = {}
    for path in args.benign_csv:
        rows = read_csv(path)
        if args.max_benign_rows is not None:
            rows = rows[: args.max_benign_rows]
        metric_compute_stats[f"benign:{path}"] = fill_missing_pair_metrics(
            rows,
            csv_path=path,
            metrics=metrics_to_compute,
            device=device,
            batch_size=args.metric_batch_size,
            dreamsim_cache_dir=args.dreamsim_cache_dir,
            dreamsim_type=args.dreamsim_type,
            imagenet_roots=imagenet_roots,
        )
        benign_rows.extend(rows)

    thresholds, calibration = calibrate_from_args(args, benign_rows)
    calibration["metric_compute_stats"] = metric_compute_stats
    calibration["imagenet_roots"] = [str(path) for path in imagenet_roots]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated_rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    for index, path in enumerate(args.attack_csv):
        rows = read_csv(path)
        if args.max_attack_rows is not None:
            rows = rows[: args.max_attack_rows]
        metric_compute_stats[f"attack:{path}"] = fill_missing_pair_metrics(
            rows,
            csv_path=path,
            metrics=metrics_to_compute,
            device=device,
            batch_size=args.metric_batch_size,
            dreamsim_cache_dir=args.dreamsim_cache_dir,
            dreamsim_type=args.dreamsim_type,
            imagenet_roots=imagenet_roots,
        )
        label = args.run_label[index] if args.run_label else default_run_label(path)
        annotated = []
        for row in rows:
            annotated_row = annotate_collapse_row(row, thresholds)
            annotated_row["run_label"] = label
            annotated_row["source_csv"] = str(path)
            annotated.append(annotated_row)
        annotated_rows.extend(annotated)
        summary = summarize_collapse_rows(rows, thresholds)
        run_summaries.append(
            add_metric_presence(
                {
                    "run_label": label,
                    "source_csv": str(path),
                    **summary,
                },
                rows,
            )
        )

    write_csv(output_dir / "collapse_rows.csv", annotated_rows)
    write_csv(output_dir / "collapse_summary.csv", run_summaries)
    write_csv(output_dir / "collapse_by_class.csv", grouped_summary_rows(annotated_rows, thresholds, args.group_by))
    write_csv(output_dir / "adv_label_distribution.csv", label_distribution_rows(annotated_rows))
    write_json(output_dir / "thresholds.json", calibration)
    write_json(
        output_dir / "collapse_summary.json",
        {
            "thresholds": calibration,
            "runs": run_summaries,
            "overall": add_metric_presence(
                summarize_collapse_rows(annotated_rows, thresholds),
                annotated_rows,
            ),
        },
    )

    print(f"Wrote collapse evaluation to {output_dir}")
    print(json.dumps({"thresholds": calibration, "runs": run_summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
