#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# run_cross_domain_eval.sh
#
# Cross-domain evaluation: use encoders trained on own data (10 classes)
# to classify RadioML data (24 classes) by training a new classifier head.
#
# Supported backbones: denomae, dino
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
RADIOML_ROOT="/mnt/d/Rowan/discrete-llm-amc/data/RadioML"

# ── Configuration ────────────────────────────────────────────────────────
BACKBONE="${1:-denomae}"   # denomae or dino (pass as first arg)
TRAIN_EPOCHS=20
LR=1e-3
BATCH_SIZE=32
GPU="0"

# ── Weights & settings per backbone ─────────────────────────────────────
if [ "$BACKBONE" = "denomae" ]; then
    WEIGHTS="${PROJECT_ROOT}/models/denoMAE2_finetunedClassifier.pth"
    IMAGE_SIZE=224
    OUTPUT_DIR="${PROJECT_ROOT}/exp/cross_domain_denomae"
elif [ "$BACKBONE" = "dino" ]; then
    WEIGHTS="${PROJECT_ROOT}/exp/dino_classifier.pth"
    IMAGE_SIZE=96
    OUTPUT_DIR="${PROJECT_ROOT}/exp/cross_domain_dino"
else
    echo "Unknown backbone: $BACKBONE (use 'denomae' or 'dino')"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  Cross-Domain Evaluation"
echo "  Backbone:    $BACKBONE"
echo "  Weights:     $WEIGHTS"
echo "  Image size:  $IMAGE_SIZE"
echo "  Epochs:      $TRAIN_EPOCHS"
echo "  Output:      $OUTPUT_DIR"
echo "═══════════════════════════════════════════════════════════════"

python -m src.evaluation.cross_domain_eval \
    --backbone "$BACKBONE" \
    --weights "$WEIGHTS" \
    --data_root "$RADIOML_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --train_epochs "$TRAIN_EPOCHS" \
    --lr "$LR" \
    --batch_size "$BATCH_SIZE" \
    --image_size "$IMAGE_SIZE" \
    --gpu "$GPU"

echo ""
echo "Done. Results in: $OUTPUT_DIR"
