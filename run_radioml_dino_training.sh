#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Train DINOv2 on RadioML constellation images (SNR 0 → 20 dB)
#
# Stage 1:  Autoencoder  (self-supervised reconstruction pre-training)
# Stage 2:  Classifier   (supervised fine-tuning from autoencoder encoder)
#
# This mirrors the original dino_training.sh used for the "own" dataset,
# but trains on RadioML 2018.01A data from snr_0db, snr_10db, snr_20db.
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# ─── Configuration ────────────────────────────────────────────────────────
DATA_ROOT="/mnt/d/Rowan/discrete-llm-amc/data/RadioML"
SNR_LEVELS="snr_0db snr_10db snr_20db"

# Output model paths
AE_WEIGHTS="exp/radioml_dino_autoencoder.pth"
CLS_WEIGHTS="exp/radioml_dino_classifier.pth"

# Training hyper-parameters
NUM_WORKERS=10
BATCH_SIZE=128
EVAL_STEP=5
IMAGE_SIZE=96

AE_EPOCHS=50
AE_LR=1e-4

CLS_EPOCHS=50
CLS_LR=5e-4

# Early stopping
PATIENCE=5
MIN_DELTA=1e-4

# ─── Stage 1: Autoencoder ────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo "  Stage 1 — DINOv2 Autoencoder on RadioML (SNR 0–20 dB)"
echo "══════════════════════════════════════════════════════════════"

python -m src.representation_learning.radioml_dino_training \
    --stage autoencoder \
    --data_root "$DATA_ROOT" \
    --snr_levels $SNR_LEVELS \
    --save_path "$AE_WEIGHTS" \
    --num_epochs $AE_EPOCHS \
    --learning_rate $AE_LR \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --eval_step $EVAL_STEP \
    --image_size $IMAGE_SIZE \
    --patience $PATIENCE \
    --min_delta $MIN_DELTA

echo ""
echo "  Autoencoder saved → $AE_WEIGHTS"
echo ""

# ─── Stage 2: Classifier ─────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo "  Stage 2 — DINOv2 Classifier on RadioML (SNR 0–20 dB)"
echo "══════════════════════════════════════════════════════════════"

python -m src.representation_learning.radioml_dino_training \
    --stage classifier \
    --data_root "$DATA_ROOT" \
    --snr_levels $SNR_LEVELS \
    --pretrained_path "$AE_WEIGHTS" \
    --save_path "$CLS_WEIGHTS" \
    --num_epochs $CLS_EPOCHS \
    --learning_rate $CLS_LR \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --eval_step $EVAL_STEP \
    --image_size $IMAGE_SIZE \
    --patience $PATIENCE \
    --min_delta $MIN_DELTA

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Done!  Classifier saved → $CLS_WEIGHTS"
echo "══════════════════════════════════════════════════════════════"
