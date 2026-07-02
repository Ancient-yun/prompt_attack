from argparse import Namespace
import importlib.util
from pathlib import Path

import torch


def _load_run_fixed10_uap_module():
    path = Path("scripts/run_fixed10_uap.py")
    spec = importlib.util.spec_from_file_location("run_fixed10_uap", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_fixed10_uap = _load_run_fixed10_uap_module()


def test_run_name_uses_explicit_override() -> None:
    args = Namespace(run_name="local_clip_t32_e2_gb16")

    assert run_fixed10_uap.run_name(args, train_ipc=None, test_ipc=None) == "local_clip_t32_e2_gb16"


def test_continuation_metadata_reads_source_prompt(tmp_path: Path) -> None:
    prompt_path = tmp_path / "learned_prompt.pt"
    torch.save(
        {
            "learnable_embeddings": torch.zeros((2, 4)),
            "metadata": {"run_name": "source_run", "epochs": 1},
        },
        prompt_path,
    )
    args = Namespace(init_prompt=prompt_path, base_epochs=1, epochs=1)

    metadata = run_fixed10_uap.continuation_metadata(args)

    assert metadata["init_prompt_path"] == str(prompt_path)
    assert metadata["base_epochs"] == 1
    assert metadata["additional_epochs"] == 1
    assert metadata["total_epochs"] == 2
    assert metadata["source_run_name"] == "source_run"
    assert metadata["source_prompt_metadata"]["epochs"] == 1


def test_build_stage_config_accepts_mixed13_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    info_root = tmp_path / "imagenet_info"
    imagenet_root = tmp_path / "imagenet"
    args = Namespace(
        config=Path("configs/flux2_resnet18.yaml"),
        imagenet_root=imagenet_root,
        imagenet_info_root=info_root,
        class_mode="mixed_13",
        victim_name="resnet18_mixed13",
        victim_weights=None,
        victim_checkpoint=checkpoint,
        global_batch_size=2,
        attack_batch_size=None,
        generator="mock",
        generator_batch_size=2,
        height=224,
        width=224,
        num_inference_steps=1,
        semantic_model=None,
        objective="margin_dino",
        include_clean_incorrect=False,
        num_tokens=16,
        learnable_token_initializer="object",
        learnable_token_init_seed=0,
        init_prompt=None,
        lr=0.1,
        epochs=1,
        steps=1,
        lambda_sem=0.0,
        semantic_loss_weight=3.0,
        attack_margin=0.0,
        wandb_mode="disabled",
        wandb_project="test",
        log_images=False,
        grid_save_policy="representative",
        max_saved_grids=4,
        save_original_images=False,
        image_format="png",
        image_quality=95,
        image_save_workers=1,
    )

    config = run_fixed10_uap.build_stage_config(
        args,
        split="train",
        images_per_class=1,
        output_root=tmp_path / "out",
        name="mixed13_test",
        steps=1,
    )

    assert config.data.class_mode == "mixed_13"
    assert config.data.imagenet_root == imagenet_root
    assert config.data.imagenet_info_root == info_root
    assert config.victim.name == "resnet18_mixed13"
    assert config.victim.checkpoint_path == checkpoint
