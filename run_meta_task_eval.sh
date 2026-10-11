#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# RQ3 — Meta-Task-Regularized DenoMAE2 Few-Shot Adaptation (held-out SNR)
# ═══════════════════════════════════════════════════════════════════════════
#
# Trains conditions B (episodic control) and C (Meta-Task) from the same
# finetuned DenoMAE2 checkpoint, then evaluates both at 1-/5-/10-shot on
# held-out RadioML SNRs (-10dB, -20dB) never seen during supervised
# finetuning or this episodic stage. See
# /home/woreom/.claude/plans/declarative-soaring-bubble.md for the design.
#
# Usage:
#   chmod +x run_meta_task_eval.sh && ./run_meta_task_eval.sh
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

# Avoids CUDA OOM/fragmentation observed during episodic training of a
# 12-layer ViT + 8-layer decoder at 24-way batches on a single 24GB GPU.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GREEN='\033[0;32m'
NC='\033[0m'
log_step() {
    echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  $1${NC}"
    echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}"
}

# ─── Paths ──────────────────────────────────────────────────────────────
INIT_WEIGHTS="exp/denomae_ft_radioml/denoMAE2_rml_finetunedClassifier_best.pth"
TRAIN_DATA="exp/denomae_ft_radioml/merged_train"
RADIOML_ROOT="/mnt/d/Rowan/discrete-llm-amc/data/RadioML"
OUT_DIR="exp/meta_task_fewshot"

# ─── Episodic training config ──────────────────────────────────────────
NUM_EPISODES=5000
TRAIN_N_WAY=10   # reduced from the full 24 RadioML classes: at 24-way, a
                 # 12-layer ViT encoder + 8-layer decoder batch OOMs a 24GB
                 # GPU during training (confirmed by testing). Episodic
                 # meta-training conventionally uses a different N than
                 # eval anyway. Held-out eval below stays at the full
                 # 24-way target task (no_grad, much lighter).
SHOT=5
QUERY=5
LR=1e-4   # 1e-5 left proto_loss flat (~2.0-2.3) for 2000 steps and left
          # both conditions near chance even in-distribution -- raised 10x
          # to actually move a ViT encoder's weights within this budget.
AUX_WEIGHT=1.0

mkdir -p "$OUT_DIR"

log_step "Stage B: episodic control (prototype loss only)"
python -m src.fewshot_meta_task.episodic_train \
    --mode control \
    --train_data_path "$TRAIN_DATA" \
    --init_weights "$INIT_WEIGHTS" \
    --output_path "$OUT_DIR/encoder_control.pth" \
    --num_episodes "$NUM_EPISODES" \
    --n_way "$TRAIN_N_WAY" \
    --shot "$SHOT" --query "$QUERY" --lr "$LR"

log_step "Stage C: Meta-Task (prototype loss + DenoMAE2 native auxiliary loss)"
python -m src.fewshot_meta_task.episodic_train \
    --mode meta_task \
    --train_data_path "$TRAIN_DATA" \
    --init_weights "$INIT_WEIGHTS" \
    --output_path "$OUT_DIR/encoder_meta_task.pth" \
    --num_episodes "$NUM_EPISODES" \
    --n_way "$TRAIN_N_WAY" \
    --shot "$SHOT" --query "$QUERY" --lr "$LR" --aux_weight "$AUX_WEIGHT"

log_step "Held-out SNR few-shot evaluation (1-/5-/10-shot, paired episodes, full 24-way)"
python -m src.fewshot_meta_task.fewshot_eval \
    --checkpoints control="$OUT_DIR/encoder_control.pth" meta_task="$OUT_DIR/encoder_meta_task.pth" \
    --snr_dirs snr_-10db="$RADIOML_ROOT/snr_-10db/train/img" snr_-20db="$RADIOML_ROOT/snr_-20db/train/img" \
    --shots 1 5 10 \
    --query "$QUERY" \
    --n_episodes 200 \
    --output_csv "$OUT_DIR/results.csv"

log_step "Done. Results: $OUT_DIR/results.csv"
