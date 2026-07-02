"""ImageNet superclass dataset helpers.

The mixed_13 definitions follow the CustomImageNet superclass list from the
MadryLab robustness documentation. Descendants are resolved from the official
ImageNet/WordNet metadata files instead of hard-coding the 1K leaf list.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from prompt_attack.data.imagenet import ImageRecord, list_class_images


@dataclass(frozen=True)
class SuperclassDefinition:
    synset: str
    label: str


@dataclass(frozen=True)
class SuperclassGroup:
    index: int
    synset: str
    label: str
    descendants: tuple[str, ...]


@dataclass(frozen=True)
class SuperclassMapping:
    name: str
    groups: tuple[SuperclassGroup, ...]
    imagenet_class_index: dict[str, int]
    imagenet_class_labels: dict[str, str]
    mapping_hash: str

    @property
    def categories(self) -> list[str]:
        return [group.label for group in self.groups]

    @property
    def synset_to_group(self) -> dict[str, SuperclassGroup]:
        return {
            descendant: group
            for group in self.groups
            for descendant in group.descendants
        }


MIXED_13_CLASSES: tuple[SuperclassDefinition, ...] = (
    SuperclassDefinition("n02084071", "dog"),
    SuperclassDefinition("n01503061", "bird"),
    SuperclassDefinition("n02159955", "insect"),
    SuperclassDefinition("n03405725", "furniture"),
    SuperclassDefinition("n02512053", "fish"),
    SuperclassDefinition("n02484322", "monkey"),
    SuperclassDefinition("n02958343", "car"),
    SuperclassDefinition("n02120997", "cat"),
    SuperclassDefinition("n04490091", "truck"),
    SuperclassDefinition("n13134947", "fruit"),
    SuperclassDefinition("n12992868", "fungus"),
    SuperclassDefinition("n02858304", "boat"),
    SuperclassDefinition("n03082979", "computer"),
)

SUPERCLASS_DATASETS: dict[str, tuple[SuperclassDefinition, ...]] = {
    "mixed_13": MIXED_13_CLASSES,
    "mixed13": MIXED_13_CLASSES,
}

REQUIRED_IMAGENET_INFO_FILES = (
    "imagenet_class_index.json",
    "wordnet.is_a.txt",
    "words.txt",
)

BREEDS_IMAGENET_INFO_FILES = (
    "dataset_class_info.json",
    "class_hierarchy.txt",
    "node_names.txt",
)


def normalize_superclass_dataset_name(name: str) -> str:
    normalized = name.lower().replace("-", "_")
    if normalized == "mixed13":
        return "mixed_13"
    return normalized


def require_imagenet_info_root(info_root: Path) -> None:
    if all((info_root / filename).exists() for filename in REQUIRED_IMAGENET_INFO_FILES):
        return
    if all((info_root / filename).exists() for filename in BREEDS_IMAGENET_INFO_FILES):
        return
    missing = [filename for filename in REQUIRED_IMAGENET_INFO_FILES if not (info_root / filename).exists()]
    breeds_missing = [
        filename for filename in BREEDS_IMAGENET_INFO_FILES if not (info_root / filename).exists()
    ]
    if missing:
        joined = ", ".join(missing)
        breeds_joined = ", ".join(breeds_missing)
        raise FileNotFoundError(
            f"ImageNet hierarchy metadata is missing from {info_root}: {joined}. "
            "Expected either ImageNet files "
            "(imagenet_class_index.json, wordnet.is_a.txt, words.txt) or BREEDS files "
            f"(dataset_class_info.json, class_hierarchy.txt, node_names.txt; missing: {breeds_joined})."
        )


def load_imagenet_class_index(info_root: Path) -> tuple[dict[str, int], dict[str, str]]:
    require_imagenet_info_root(info_root)
    path = info_root / "imagenet_class_index.json"
    if not path.exists():
        path = info_root / "dataset_class_info.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    synset_to_index: dict[str, int] = {}
    synset_to_label: dict[str, str] = {}
    if isinstance(raw, dict):
        for raw_index, value in raw.items():
            if not isinstance(value, list | tuple) or len(value) < 2:
                raise ValueError(f"Invalid ImageNet class index entry {raw_index!r}: {value!r}")
            synset = str(value[0])
            synset_to_index[synset] = int(raw_index)
            synset_to_label[synset] = str(value[1]).replace("_", " ")
    elif isinstance(raw, list):
        for value in raw:
            if not isinstance(value, list | tuple) or len(value) < 3:
                raise ValueError(f"Invalid BREEDS class info entry: {value!r}")
            class_index = int(value[0])
            synset = str(value[1])
            synset_to_index[synset] = class_index
            synset_to_label[synset] = str(value[2]).replace("_", " ")
    else:
        raise ValueError(f"Unsupported class index metadata format in {path}")
    return synset_to_index, synset_to_label


def load_wordnet_names(info_root: Path) -> dict[str, str]:
    require_imagenet_info_root(info_root)
    path = info_root / "words.txt"
    if not path.exists():
        path = info_root / "node_names.txt"
    names: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        synset, _, label = stripped.partition("\t")
        if synset and label:
            names[synset] = label
    return names


def load_wordnet_children(info_root: Path) -> dict[str, set[str]]:
    """Load WordNet child links.

    The ImageNet ``wordnet.is_a.txt`` format is ``child parent`` per line.
    BREEDS ``class_hierarchy.txt`` uses ``parent child`` per line.
    """
    require_imagenet_info_root(info_root)
    path = info_root / "wordnet.is_a.txt"
    child_parent_format = True
    if not path.exists():
        path = info_root / "class_hierarchy.txt"
        child_parent_format = False
    children: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 2:
            raise ValueError(f"Invalid wordnet.is_a.txt line: {line!r}")
        if child_parent_format:
            child, parent = parts
        else:
            parent, child = parts
        children.setdefault(parent, set()).add(child)
    return children


def descendants_for_synset(root_synset: str, children: dict[str, set[str]]) -> set[str]:
    descendants = {root_synset}
    queue: deque[str] = deque([root_synset])
    while queue:
        current = queue.popleft()
        for child in sorted(children.get(current, ())):
            if child in descendants:
                continue
            descendants.add(child)
            queue.append(child)
    return descendants


def build_superclass_mapping(
    dataset_name: str,
    *,
    info_root: Path,
    available_synsets: set[str] | None = None,
) -> SuperclassMapping:
    normalized_name = normalize_superclass_dataset_name(dataset_name)
    if normalized_name not in SUPERCLASS_DATASETS:
        raise ValueError(f"Unsupported superclass dataset: {dataset_name}")

    class_index, class_labels = load_imagenet_class_index(info_root)
    imagenet_synsets = set(class_index)
    if available_synsets is not None:
        imagenet_synsets &= available_synsets
    children = load_wordnet_children(info_root)

    definitions = SUPERCLASS_DATASETS[normalized_name]
    raw_leaf_sets: list[tuple[SuperclassDefinition, tuple[str, ...]]] = []
    for definition in definitions:
        descendants = descendants_for_synset(definition.synset, children)
        leaf_synsets = tuple(sorted(descendants & imagenet_synsets))
        if not leaf_synsets:
            raise ValueError(
                f"Superclass {definition.label} ({definition.synset}) has no ImageNet-1K "
                f"descendants available for dataset {normalized_name}."
            )
        raw_leaf_sets.append((definition, leaf_synsets))

    # CustomImageNet presets can include semantically nested WordNet nodes
    # (e.g. mixed_13 has both car and truck). Give later, more specific
    # preset entries priority and remove their leaves from earlier groups.
    assigned_later: set[str] = set()
    resolved_reversed: list[tuple[SuperclassDefinition, tuple[str, ...]]] = []
    for definition, leaf_synsets in reversed(raw_leaf_sets):
        resolved = tuple(synset for synset in leaf_synsets if synset not in assigned_later)
        if not resolved:
            raise ValueError(
                f"Superclass {definition.label} ({definition.synset}) has no non-overlapping "
                f"ImageNet-1K descendants after overlap resolution."
            )
        assigned_later.update(resolved)
        resolved_reversed.append((definition, resolved))

    groups: list[SuperclassGroup] = []
    for index, (definition, leaf_synsets) in enumerate(reversed(resolved_reversed)):
        groups.append(
            SuperclassGroup(
                index=index,
                synset=definition.synset,
                label=definition.label,
                descendants=leaf_synsets,
            )
        )

    mapping_payload = {
        "name": normalized_name,
        "groups": [
            {
                "index": group.index,
                "synset": group.synset,
                "label": group.label,
                "descendants": list(group.descendants),
            }
            for group in groups
        ],
    }
    digest = hashlib.sha256(
        json.dumps(mapping_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return SuperclassMapping(
        name=normalized_name,
        groups=tuple(groups),
        imagenet_class_index=class_index,
        imagenet_class_labels=class_labels,
        mapping_hash=digest,
    )


def available_split_synsets(split_dir: Path) -> set[str]:
    if not split_dir.exists():
        raise FileNotFoundError(f"ImageNet split directory not found: {split_dir}")
    return {path.name for path in split_dir.iterdir() if path.is_dir()}


def round_robin_limit(paths_by_synset: list[tuple[str, list[Path]]], limit: int) -> list[tuple[str, Path]]:
    selected: list[tuple[str, Path]] = []
    offset = 0
    while len(selected) < limit:
        added = False
        for synset, paths in paths_by_synset:
            if offset < len(paths):
                selected.append((synset, paths[offset]))
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        offset += 1
    return selected


def build_superclass_image_records(
    *,
    imagenet_root: Path,
    split: str,
    dataset_name: str,
    info_root: Path,
    images_per_superclass: int | None,
    candidate_multiplier: int = 1,
) -> tuple[list[ImageRecord], SuperclassMapping]:
    split_path = imagenet_root if not split else imagenet_root / split
    available = available_split_synsets(split_path)
    mapping = build_superclass_mapping(
        dataset_name,
        info_root=info_root,
        available_synsets=available,
    )
    cap = None
    if images_per_superclass is not None:
        cap = images_per_superclass * max(1, candidate_multiplier)

    records: list[ImageRecord] = []
    for group in mapping.groups:
        paths_by_synset: list[tuple[str, list[Path]]] = []
        for synset in group.descendants:
            class_dir = split_path / synset
            paths = list_class_images(class_dir) if class_dir.exists() else []
            if paths:
                paths_by_synset.append((synset, paths))
        if cap is None:
            selected = [
                (synset, path)
                for synset, paths in paths_by_synset
                for path in paths
            ]
        else:
            selected = round_robin_limit(paths_by_synset, cap)
        for synset, image_path in selected:
            records.append(
                ImageRecord(
                    path=image_path,
                    synset=synset,
                    class_label=group.label,
                    class_index=group.index,
                    image_id=image_path.stem,
                    fine_class_label=mapping.imagenet_class_labels.get(synset),
                    fine_class_index=mapping.imagenet_class_index.get(synset),
                    superclass_synset=group.synset,
                )
            )
    return records, mapping
