import json
from pathlib import Path

from prompt_attack.data.superclass import (
    MIXED_13_CLASSES,
    build_superclass_image_records,
    build_superclass_mapping,
)


def _write_fake_imagenet_info(info_root: Path) -> list[str]:
    info_root.mkdir()
    class_index = {}
    edges = []
    words = []
    leaf_synsets = []
    for index, superclass in enumerate(MIXED_13_CLASSES):
        leaf = f"n9{index:07d}"
        leaf_synsets.append(leaf)
        class_index[str(index)] = [leaf, f"{superclass.label}_leaf"]
        edges.append(f"{leaf} {superclass.synset}\n")
        words.append(f"{superclass.synset}\t{superclass.label}\n")
        words.append(f"{leaf}\t{superclass.label} leaf\n")
    (info_root / "imagenet_class_index.json").write_text(
        json.dumps(class_index),
        encoding="utf-8",
    )
    (info_root / "wordnet.is_a.txt").write_text("".join(edges), encoding="utf-8")
    (info_root / "words.txt").write_text("".join(words), encoding="utf-8")
    return leaf_synsets


def test_mixed13_mapping_uses_wordnet_descendants(tmp_path: Path) -> None:
    info_root = tmp_path / "info"
    leaf_synsets = _write_fake_imagenet_info(info_root)

    mapping = build_superclass_mapping(
        "mixed_13",
        info_root=info_root,
        available_synsets=set(leaf_synsets),
    )

    assert mapping.categories == [superclass.label for superclass in MIXED_13_CLASSES]
    assert len(mapping.groups) == 13
    assert mapping.groups[0].descendants == (leaf_synsets[0],)
    assert mapping.synset_to_group[leaf_synsets[-1]].label == MIXED_13_CLASSES[-1].label
    assert len(mapping.mapping_hash) == 64


def test_mixed13_records_are_superclass_labeled(tmp_path: Path) -> None:
    info_root = tmp_path / "info"
    leaf_synsets = _write_fake_imagenet_info(info_root)
    split = tmp_path / "imagenet" / "train"
    split.mkdir(parents=True)
    for synset in leaf_synsets:
        class_dir = split / synset
        class_dir.mkdir()
        for index in range(2):
            (class_dir / f"{synset}_{index}.JPEG").write_bytes(b"not-opened")

    records, mapping = build_superclass_image_records(
        imagenet_root=tmp_path / "imagenet",
        split="train",
        dataset_name="mixed13",
        info_root=info_root,
        images_per_superclass=1,
    )

    assert len(records) == 13
    assert [record.class_index for record in records] == list(range(13))
    assert [record.class_label for record in records] == mapping.categories
    assert records[0].fine_class_label == "dog leaf"
    assert records[0].fine_class_index == 0
    assert records[0].superclass_synset == MIXED_13_CLASSES[0].synset


def test_mixed13_mapping_accepts_breeds_metadata_format(tmp_path: Path) -> None:
    info_root = tmp_path / "info"
    info_root.mkdir()
    leaf_synsets = []
    class_info = []
    hierarchy = []
    node_names = []
    for index, superclass in enumerate(MIXED_13_CLASSES):
        leaf = f"n8{index:07d}"
        leaf_synsets.append(leaf)
        class_info.append([index, leaf, f"{superclass.label} leaf"])
        hierarchy.append(f"{superclass.synset} {leaf}\n")
        node_names.append(f"{superclass.synset}\t{superclass.label}\n")
        node_names.append(f"{leaf}\t{superclass.label} leaf\n")
    (info_root / "dataset_class_info.json").write_text(
        json.dumps(class_info),
        encoding="utf-8",
    )
    (info_root / "class_hierarchy.txt").write_text("".join(hierarchy), encoding="utf-8")
    (info_root / "node_names.txt").write_text("".join(node_names), encoding="utf-8")

    mapping = build_superclass_mapping(
        "mixed_13",
        info_root=info_root,
        available_synsets=set(leaf_synsets),
    )

    assert mapping.categories == [superclass.label for superclass in MIXED_13_CLASSES]
    assert mapping.groups[0].descendants == (leaf_synsets[0],)
    assert mapping.imagenet_class_labels[leaf_synsets[0]] == "dog leaf"
