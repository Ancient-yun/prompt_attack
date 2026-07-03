#!/usr/bin/env bash
# Full e5 run: CLIP zero-shot identity oracle (margin_oracle, w=0.5 sweet spot from
# section 7) combined with the axis-mode strength schedule (section 8). See
# scripts/run_mixed13_oracle_axis_pilot.sh for the full rationale and legitimacy-gate
# design (oracle CLIP true-class probability, not SSIM).
set -euo pipefail

python scripts/run_uap.py \
  --root outputs/uap_mixed13 \
  --imagenet-root /data/ImageNet/2012 \
  --imagenet-info-root external/imagenet_hierarchy_wordnet \
  --class-mode mixed_13 \
  --victim-name resnet18_mixed13 \
  --victim-checkpoint outputs/victims/mixed13_resnet18/best.pt \
  --train-images-per-class 20 \
  --test-images-per-class 20 \
  --epochs 5 \
  --global-batch-size 16 \
  --generator-batch-size 1 \
  --num-inference-steps 4 \
  --num-tokens 32 \
  --lr 0.1 \
  --objective margin_oracle \
  --semantic-model clip_oracle \
  --lambda-sem 0.0 \
  --semantic-loss-weight 0.5 \
  --strength-schedule \
  --train-strengths 0.2,0.6,1.0 \
  --eval-strengths 0.1,0.3,0.5,0.7,1.0 \
  --legitimacy-ssim-threshold 0 \
  --legitimacy-semantic-threshold 0.5 \
  --wandb-mode disabled \
  --grid-save-policy representative \
  --max-saved-grids 60 \
  --image-format png \
  --image-save-workers 4 \
  --run-name mixed13_oracle_axis_w0p5_ipc20_gb16_e5
