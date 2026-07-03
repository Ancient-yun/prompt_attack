#!/usr/bin/env bash
set -euo pipefail

cd /NHNHOME/WORKSPACE/0226010134_A/daeyun/prompt_attack

echo "==== cr_dino token variants clean-correct batched-filter: 16 then 32 then 64 ===="
echo "code=$(git rev-parse --short HEAD)"
grep -n "CLEAN_FILTER_BATCH_SIZE" src/prompt_attack/attacks/runner.py | head -1

COMMON_ARGS=(
  --imagenet-root /NHNHOME/WORKSPACE/0226010134_A/data/ImageNet/2012
  --root outputs/uap_fixed10_b200_gpu4_crdino_lam0p5_token_variants_clean_batched
  --train-images-per-class all
  --test-images-per-class all
  --epochs 1
  --global-batch-size 16
  --generator-batch-size 16
  --num-inference-steps 4
  --lr 0.1
  --lambda-sem 0.5
  --objective cr_dino
  --wandb-mode online
  --log-images
)

for tokens in 16 32 64; do
  echo "==== START tokens=${tokens} objective=cr_dino lambda_sem=0.5 lr=0.1 ===="
  CUDA_VISIBLE_DEVICES=4 \
    UV_LINK_MODE=copy \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    WANDB_MODE=online \
    uv run python scripts/run_uap.py "${COMMON_ARGS[@]}" --num-tokens "${tokens}"
  echo "==== DONE tokens=${tokens} objective=cr_dino lambda_sem=0.5 ===="
done
