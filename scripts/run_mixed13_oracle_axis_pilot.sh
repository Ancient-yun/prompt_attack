#!/usr/bin/env bash
# Tiny real-FLUX pilot combining the CLIP zero-shot identity oracle (margin_oracle,
# section 7) with the MAELS-inspired axis-mode strength schedule (section 8).
#
# Section 7 found w=0.5 (semantic_loss_weight) was the oracle sweet spot (4.2% preserved
# ASR) but the untargeted-universal ceiling stayed low (~4%). Section 8 found the
# strength-schedule's early-success weighting works mechanically (successes land at the
# lowest swept strength) but LPIPS suppressed the attack near zero. This pilot tests
# whether combining them -- oracle's permissive semantic guard + strength-schedule's
# early-success pressure -- can raise the oracle-preserved ASR above the ~4% ceiling.
#
# Legitimacy gate uses the oracle's own CLIP true-class probability (semantic_similarity
# for objective=margin_oracle), not SSIM, since the oracle explicitly permits large pixel
# changes (pose/viewpoint) as long as CLIP still reads the correct class -- matching
# section 7's own "oracle-preserved ASR" definition (victim wrong AND CLIP prob > 0.5).
set -euo pipefail

python scripts/run_uap.py \
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
  --num-tokens 32 \
  --lr 0.1 \
  --objective margin_oracle \
  --semantic-model clip_oracle \
  --lambda-sem 0.0 \
  --semantic-loss-weight 0.5 \
  --strength-schedule \
  --train-strengths 0.3,1.0 \
  --eval-strengths 0.3,0.7,1.0 \
  --legitimacy-ssim-threshold 0 \
  --legitimacy-semantic-threshold 0.5 \
  --wandb-mode disabled \
  --grid-save-policy representative \
  --max-saved-grids 20 \
  --image-format png \
  --image-save-workers 4 \
  --run-name mixed13_oracle_axis_pilot_e1
