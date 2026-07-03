#!/usr/bin/env bash
# Anti-overlay experiment (experiments 2+3 from the handoff diagnosis).
#
# The margin_dino baseline reaches 82% ASR only by learning a fixed universal overlay
# (residual cosine 0.72, SSIM collapses to ~0.02) that DINO/CLIP are blind to. This run
# tries to break the overlay with three interventions at once:
#   - margin_lpips : hinge attack + LPIPS pixel/perceptual anchor (sees the overlay)
#   - --eot        : resample the diffusion seed each step (no fixed-path memorisation)
#   - --num-tokens 8 : reduced capacity to store a texture
#
# Everything else matches the margin_dino e5 baseline for comparability.
set -euo pipefail

mkdir -p outputs/uap_mixed13

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
  --global-batch-size 16
  --generator-batch-size 1
  --num-inference-steps 4
  --lr 0.1
  --wandb-mode disabled
  --grid-save-policy representative
  --max-saved-grids 60
  --image-format png
  --image-save-workers 4
)

run_one() {
  local name="$1"; shift
  local output_dir="outputs/uap_mixed13/${name}"
  local log_path="outputs/uap_mixed13/${name}.log"
  if [[ -f "${output_dir}/metrics/summary.json" ]]; then
    echo "==== SKIP ${name}: summary exists ===="
    return
  fi
  echo "==== START ${name} $(date -Is) ===="
  python scripts/run_uap.py "${BASE_ARGS[@]}" --run-name "${name}" "$@" >"${log_path}" 2>&1
  echo "==== DONE ${name} $(date -Is) ===="
}

# Primary anti-overlay run: LPIPS anchor + EOT + 8 tokens.
run_one mixed13_resnet18_t8_mlpips_eot_ipc20_gb16_e5 \
  --objective margin_lpips \
  --semantic-model lpips \
  --lambda-sem 0.0 \
  --semantic-loss-weight 3 \
  --num-tokens 8 \
  --eot

echo "==== ALL DONE $(date -Is) ===="
