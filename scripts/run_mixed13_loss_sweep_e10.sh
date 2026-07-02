#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/uap_mixed13

SWEEP_LOG="outputs/uap_mixed13/mixed13_loss_sweep_e10.log"

BASE_ARGS=(
  --root outputs/uap_mixed13
  --imagenet-root /data/imagenet
  --imagenet-info-root external/imagenet_hierarchy_wordnet
  --class-mode mixed_13
  --victim-name resnet18_mixed13
  --victim-checkpoint outputs/victims/mixed13_resnet18/best.pt
  --train-images-per-class 20
  --test-images-per-class 20
  --epochs 5
  --base-epochs 5
  --global-batch-size 16
  --generator-batch-size 1
  --num-inference-steps 4
  --num-tokens 32
  --lr 0.1
  --wandb-mode disabled
  --grid-save-policy representative
  --max-saved-grids 60
  --image-format png
  --image-save-workers 4
)

run_one() {
  local source_name="$1"
  local target_name="$2"
  shift 2

  local source_prompt="outputs/uap_mixed13/${source_name}/prompt/learned_prompt.pt"
  local output_dir="outputs/uap_mixed13/${target_name}"
  local log_path="outputs/uap_mixed13/${target_name}.log"

  if [[ ! -f "${source_prompt}" ]]; then
    echo "==== MISSING ${source_prompt} ====" | tee -a "${SWEEP_LOG}"
    return 1
  fi

  if [[ -f "${output_dir}/metrics/summary.json" ]]; then
    echo "==== SKIP ${target_name}: summary exists ====" | tee -a "${SWEEP_LOG}"
    return
  fi

  echo "==== START ${target_name} from ${source_name} $(date -Is) ====" | tee -a "${SWEEP_LOG}"
  python scripts/run_fixed10_uap.py \
    "${BASE_ARGS[@]}" \
    --init-prompt "${source_prompt}" \
    --run-name "${target_name}" \
    "$@" >"${log_path}" 2>&1
  echo "==== DONE ${target_name} $(date -Is) ====" | tee -a "${SWEEP_LOG}"
}

run_one mixed13_resnet18_t32_mdino_ipc20_gb16_e5 \
  mixed13_resnet18_t32_mdino_ipc20_gb16_e10 \
  --objective margin_dino \
  --lambda-sem 0.0 \
  --semantic-loss-weight 3

run_one mixed13_resnet18_t32_cr_ipc20_gb16_e5 \
  mixed13_resnet18_t32_cr_ipc20_gb16_e10 \
  --objective cr \
  --lambda-sem 0.0 \
  --semantic-loss-weight 3

run_one mixed13_resnet18_t32_crdino_lam0p5_ipc20_gb16_e5 \
  mixed13_resnet18_t32_crdino_lam0p5_ipc20_gb16_e10 \
  --objective cr_dino \
  --lambda-sem 0.5 \
  --semantic-loss-weight 3

run_one mixed13_resnet18_t32_margin_ipc20_gb16_e5 \
  mixed13_resnet18_t32_margin_ipc20_gb16_e10 \
  --objective untargeted_margin \
  --lambda-sem 0.0 \
  --semantic-loss-weight 3

run_one mixed13_resnet18_t32_mclip_ipc20_gb16_e5 \
  mixed13_resnet18_t32_mclip_ipc20_gb16_e10 \
  --objective margin_clip_img2img \
  --lambda-sem 0.0 \
  --semantic-loss-weight 3

run_one mixed13_resnet18_t32_clip_ipc20_gb16_e5 \
  mixed13_resnet18_t32_clip_ipc20_gb16_e10 \
  --objective clip_img2img \
  --lambda-sem 0.0 \
  --semantic-loss-weight 3

run_one mixed13_resnet18_t32_dino_ipc20_gb16_e5 \
  mixed13_resnet18_t32_dino_ipc20_gb16_e10 \
  --objective dino_img2img \
  --lambda-sem 0.0 \
  --semantic-loss-weight 3

echo "==== ALL DONE $(date -Is) ====" | tee -a "${SWEEP_LOG}"
