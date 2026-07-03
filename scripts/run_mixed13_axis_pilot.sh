#!/usr/bin/env bash
# Tiny real-FLUX pilot for the MAELS-inspired anchor/axis strength-schedule attack.
# Validates the two-param-group optimizer and axis parameterization under real FLUX
# gradients (mock generator already validated the mechanics) before committing to the
# ~4.5h full e5 run. Expected wall-clock: ~15 minutes.
set -euo pipefail

python scripts/run_fixed10_uap.py \
  --root outputs/uap_mixed13 \
  --imagenet-root /data/ImageNet/2012 \
  --imagenet-info-root external/imagenet_hierarchy_wordnet \
  --class-mode mixed_13 \
  --victim-name resnet18_mixed13 \
  --victim-checkpoint outputs/victims/mixed13_resnet18/best.pt \
  --train-images-per-class 4 \
  --test-images-per-class 4 \
  --epochs 1 \
  --global-batch-size 16 \
  --generator-batch-size 1 \
  --num-inference-steps 4 \
  --num-tokens 8 \
  --num-anchor-tokens 4 \
  --lr 0.1 \
  --objective margin_lpips \
  --semantic-model lpips \
  --lambda-sem 0.0 \
  --semantic-loss-weight 3 \
  --strength-schedule \
  --train-strengths 0.3,1.0 \
  --eval-strengths 0.3,0.7,1.0 \
  --eot \
  --wandb-mode disabled \
  --grid-save-policy representative \
  --max-saved-grids 20 \
  --image-format png \
  --image-save-workers 4 \
  --run-name mixed13_axis_pilot_e1
