#!/usr/bin/env bash
# Controlled A/B comparison of the ATTACK TERM only, everything else identical.
#   margin_oracle : attack = margin hinge   relu(z_true - max z_other)
#   sat_ce_oracle : attack = MAELS Eq.8      -log(1 - e^{-CE(F, y)})
# Both use the same CLIP zero-shot oracle preservation guard (w=0.5) and the same
# axis-mode strength schedule, so any difference is attributable to the attack term.
#
# Usage:
#   bash scripts/run_mixed13_loss_compare.sh sat_ce_oracle satce   # CE version first
#   bash scripts/run_mixed13_loss_compare.sh margin_oracle margin  # margin control
set -euo pipefail

OBJ="$1"; TAG="$2"
RUN=mixed13_${TAG}_axis_w0p5_ipc20_gb16_e5
LOG=outputs/uap_mixed13/${RUN}.log
mkdir -p outputs/uap_mixed13

if [[ -f "outputs/uap_mixed13/${RUN}/metrics/summary.json" ]]; then
  echo "==== SKIP ${RUN}: summary exists ===="
  exit 0
fi

echo "==== START ${RUN} (obj=${OBJ}) $(date -Is) ===="
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
  --objective "${OBJ}" \
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
  --run-name "${RUN}" 2>&1 | tee "${LOG}"
echo "==== DONE ${RUN} $(date -Is) ===="
