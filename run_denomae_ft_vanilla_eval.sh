#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# run_denomae_ft_vanilla_eval.sh
#
# Full pipeline using finetuned DenoMAE2 encoder + vanilla DeepSeek-R1:
#   1. Build FAISS index per SNR (finetuned DenoMAE2 encoder)
#   2. Generate top-k predictions (FAISS kNN)
#   3. Convert prediction keys (.png → .npy)
#   4. Build train + test pkls + RAG index
#   5. Evaluate with vanilla (base) DeepSeek-R1 (no LoRA adapter)
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# ─── Paths ────────────────────────────────────────────────────────────────
DATA_ROOT="/mnt/d/Rowan/discrete-llm-amc/data/RadioML"
EXP_DIR="${PROJECT_ROOT}/exp"
MODEL_DIR="/mnt/d/Rowan/discrete-llm-amc/models"
OUTPUT_DIR="${EXP_DIR}/denomae_ft_vanilla_deepseek_eval"

# ─── Finetuned DenoMAE2 weights ──────────────────────────────────────────
ENCODER_WEIGHTS="${EXP_DIR}/denomae_ft_radioml/denoMAE2_rml_finetunedClassifier_best.pth"

# ─── Encoder config ──────────────────────────────────────────────────────
BACKBONE="denomae"
NUM_CLASSES=24
IMAGE_SIZE=224
N_COMPONENTS=10
BATCH_SIZE=32

# ─── Data / pipeline settings ────────────────────────────────────────────
DATASET_TYPE="radioml"
PREDICTION_SOURCE="faiss_filled_denomae_ft"
N_BINS=5
TOP_K=5
KNN_K=50
FEATURE_TYPE="embeddings"
USE_RAG=true
RAG_K=10
MIN_RAG_CLASSES=0
PROMPT_VERSION="v2"
FAISS_TAG="denomae_ft"

# ─── SNR levels to evaluate ──────────────────────────────────────────────
EVAL_SNR_LEVELS=("snr_-20db" "snr_-10db" "snr_0db" "snr_10db" "snr_20db")

# ─── LLM config (vanilla — no adapter) ───────────────────────────────────
UNSLOTH_MODEL="unsloth/DeepSeek-R1-Distill-Qwen-7B"
PROMPT_TYPE="discret_prompts"
NUM_TRIES=1
INFERENCE_BATCH_SIZE=8
MAX_NEW_TOKENS=512

# ─── Colors ───────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

log_step() {
    echo -e "\n${GREEN}══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  $1${NC}"
    echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}"
}

# ─── Validate weights ────────────────────────────────────────────────────
if [[ ! -f "$ENCODER_WEIGHTS" ]]; then
    echo -e "${RED}ERROR: Finetuned DenoMAE2 weights not found: ${ENCODER_WEIGHTS}${NC}"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
EVAL_CSV="${OUTPUT_DIR}/vanilla_deepseek_denomae_ft_results.csv"

# ─── Naming helpers ───────────────────────────────────────────────────────
_raw_json() {
    echo "top${TOP_K}_${PREDICTION_SOURCE}_predictions.json"
}
_converted_json() {
    local bb_suffix=""
    [[ "$BACKBONE" != "dino" ]] && bb_suffix="_${BACKBONE}"
    echo "ntop${TOP_K}_${PREDICTION_SOURCE}${bb_suffix}_predictions.json"
}
_pkl_name() {
    local mode="$1" snr="$2"
    echo "${mode}_${PREDICTION_SOURCE}_${BACKBONE}_emb${N_COMPONENTS}_${snr}_${N_BINS}_${TOP_K}_data.pkl"
}

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Finetuned DenoMAE2 + Vanilla DeepSeek-R1 Evaluation       ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  ENCODER_WEIGHTS: ${ENCODER_WEIGHTS}"
echo "║  DATA_ROOT:       ${DATA_ROOT}"
echo "║  BACKBONE:        ${BACKBONE}"
echo "║  PRED_SOURCE:     ${PREDICTION_SOURCE}"
echo "║  MODEL:           ${UNSLOTH_MODEL} (vanilla, no adapter)"
echo "║  EVAL_SNR_LEVELS: ${EVAL_SNR_LEVELS[*]}"
echo "║  OUTPUT_DIR:      ${OUTPUT_DIR}"
echo "║  EVAL_CSV:        ${EVAL_CSV}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Phases 1–4: Data pipeline (per SNR)
# ═══════════════════════════════════════════════════════════════════════════
for snr in "${EVAL_SNR_LEVELS[@]}"; do
    log_step "DATA PIPELINE — ${snr}"
    SNR_DIR="${DATA_ROOT}/${snr}"

    # ── Phase 1: Constellation images (skip if exist) ────────────────
    img_dir_train="${SNR_DIR}/train/img"
    img_dir_test="${SNR_DIR}/test/img"
    if [[ -d "$img_dir_train" && -d "$img_dir_test" ]]; then
        echo -e "${YELLOW}  ⊘ Constellation images exist — skipping Phase 1${NC}"
    else
        echo -e "${CYAN}  Phase 1: Generating constellation images for ${snr} …${NC}"
        python -m src.representation_learning.generate_radioml_images \
            --data_root "$DATA_ROOT" \
            --snr_levels "$snr"
    fi

    # ── Phase 2: Build FAISS index (finetuned DenoMAE2) ──────────────
    FAISS_INDEX_PATH="${SNR_DIR}/train/faiss_knn_${FAISS_TAG}"
    if [[ -f "${FAISS_INDEX_PATH}.index" ]]; then
        echo -e "${YELLOW}  ⊘ FAISS index (${FAISS_TAG}) exists — skipping Phase 2${NC}"
    else
        echo -e "${CYAN}  Phase 2: Building FAISS index (${FAISS_TAG}) for ${snr} …${NC}"
        python -m src.representation_learning.inference build_faiss \
            --backbone "$BACKBONE" \
            --weights "$ENCODER_WEIGHTS" \
            --num_classes "$NUM_CLASSES" \
            --train_path "${SNR_DIR}/train" \
            --output "$FAISS_INDEX_PATH" \
            --image_size "$IMAGE_SIZE"
    fi

    # ── Phase 3: Top-k predictions ───────────────────────────────────
    RAW_JSON="$(_raw_json)"
    if [[ -f "${SNR_DIR}/${RAW_JSON}" ]]; then
        echo -e "${YELLOW}  ⊘ Top-k predictions exist — skipping Phase 3${NC}"
    else
        echo -e "${CYAN}  Phase 3: Generating top-k predictions for ${snr} …${NC}"
        python -m src.representation_learning.inference predict \
            --backbone "$BACKBONE" \
            --weights "$ENCODER_WEIGHTS" \
            --num_classes "$NUM_CLASSES" \
            --dataset_path "${SNR_DIR}/test" \
            --topk "$TOP_K" \
            --output "${SNR_DIR}/${RAW_JSON}" \
            --image_size "$IMAGE_SIZE" \
            --faiss_index_path "$FAISS_INDEX_PATH" \
            --knn_k "$KNN_K" \
            --fill_topk
    fi

    # ── Phase 4: Convert .png keys → .npy keys ──────────────────────
    CONVERTED_JSON="$(_converted_json)"
    if [[ -f "${SNR_DIR}/${CONVERTED_JSON}" ]]; then
        echo -e "${YELLOW}  ⊘ Converted predictions exist — skipping Phase 4${NC}"
    else
        echo -e "${CYAN}  Phase 4: Converting prediction keys for ${snr} …${NC}"
        python -m src.representation_learning.convert_predictions \
            --input  "${SNR_DIR}/${RAW_JSON}" \
            --output "${SNR_DIR}/${CONVERTED_JSON}"
    fi

    # ── Phase 5: Build train + test pkls ─────────────────────────────
    train_pkl="${SNR_DIR}/$(_pkl_name train "$snr")"
    if [[ -f "$train_pkl" ]]; then
        echo -e "${YELLOW}  ⊘ Train pkl exists — skipping Phase 5a${NC}"
    else
        echo -e "${CYAN}  Phase 5a: Building train pkl for ${snr} …${NC}"
        python -m src.prompt.generated_dataset \
            --mode train \
            --dataset_folder="$snr" \
            --noise_mode "$snr" \
            --n_bins "$N_BINS" \
            --top_k "$TOP_K" \
            --prediction_source "$PREDICTION_SOURCE" \
            --data_root "$DATA_ROOT" \
            --feature_type embeddings \
            --encoder_weights "$ENCODER_WEIGHTS" \
            --backbone "$BACKBONE" \
            --n_components "$N_COMPONENTS" \
            --batch_size "$BATCH_SIZE" \
            --num_classes "$NUM_CLASSES" \
            --use_rag --rag_k "$RAG_K" --min_classes "$MIN_RAG_CLASSES" \
            --prompt_version "$PROMPT_VERSION" \
            --dataset_type "$DATASET_TYPE"
    fi

    test_pkl="${SNR_DIR}/$(_pkl_name test "$snr")"
    if [[ -f "$test_pkl" ]]; then
        echo -e "${YELLOW}  ⊘ Test pkl exists — skipping Phase 5b${NC}"
    else
        echo -e "${CYAN}  Phase 5b: Building test pkl for ${snr} …${NC}"
        python -m src.prompt.generated_dataset \
            --mode test \
            --dataset_folder="$snr" \
            --noise_mode "$snr" \
            --n_bins "$N_BINS" \
            --top_k "$TOP_K" \
            --prediction_source "$PREDICTION_SOURCE" \
            --data_root "$DATA_ROOT" \
            --feature_type embeddings \
            --encoder_weights "$ENCODER_WEIGHTS" \
            --backbone "$BACKBONE" \
            --n_components "$N_COMPONENTS" \
            --batch_size "$BATCH_SIZE" \
            --num_classes "$NUM_CLASSES" \
            --use_rag --rag_k "$RAG_K" --min_classes "$MIN_RAG_CLASSES" \
            --prompt_version "$PROMPT_VERSION" \
            --dataset_type "$DATASET_TYPE"
    fi
done

# ═══════════════════════════════════════════════════════════════════════════
# Phase 6: Evaluate with vanilla DeepSeek-R1 (no adapter)
# ═══════════════════════════════════════════════════════════════════════════
log_step "Phase 6 — Vanilla DeepSeek-R1 evaluation (no LoRA adapter)"

for snr in "${EVAL_SNR_LEVELS[@]}"; do
    log_step "EVAL — vanilla × ${snr}"

    test_pkl="${DATA_ROOT}/${snr}/$(_pkl_name test "$snr")"
    if [[ ! -f "$test_pkl" ]]; then
        echo -e "${RED}  Test pkl missing: ${test_pkl} — skipping${NC}"
        continue
    fi

    python -c "
from src.evaluation.unsloth_eval import main
main(
    dataset_folder='${snr}',
    prompt_type='${PROMPT_TYPE}',
    model_name='${UNSLOTH_MODEL}',
    noise_mode='${snr}',
    n_bins=${N_BINS},
    top_k=${TOP_K},
    num_tries=${NUM_TRIES},
    prediction_source='${PREDICTION_SOURCE}',
    feature_type='${FEATURE_TYPE}',
    n_components=${N_COMPONENTS},
    cache_dir='${MODEL_DIR}',
    data_root='${DATA_ROOT}',
    output_dir='${OUTPUT_DIR}',
    adapter_path=None,
    inference_batch_size=${INFERENCE_BATCH_SIZE},
    max_new_tokens=${MAX_NEW_TOKENS},
    prompt_version='${PROMPT_VERSION}',
    backbone='${BACKBONE}',
)
"

    python -c "
import os, csv
from src.evaluation.unsloth_eval import read_results, get_class_names
from src.evaluation.utils import (
    sort_results_by_prompt, get_unique_prompts, print_metrics,
    acc, clean_acc, pass_acc, majority_acc,
)

_CLASS_NAMES = get_class_names('radioml')

results = read_results(
    dataset_folder='${snr}',
    prompt_type='${PROMPT_TYPE}',
    model_name='${UNSLOTH_MODEL}',
    noise_mode='${snr}',
    n_bins=${N_BINS},
    top_k=${TOP_K},
    prediction_source='${PREDICTION_SOURCE}',
    feature_type='${FEATURE_TYPE}',
    n_components=${N_COMPONENTS},
    output_dir='${OUTPUT_DIR}',
    prompt_version='${PROMPT_VERSION}',
    backbone='${BACKBONE}',
)
sorted_results = sort_results_by_prompt(results)
n_unique = len(get_unique_prompts(results))
print(f'[VANILLA × ${snr}]  Unique prompts: {n_unique}')
print_metrics(sorted_results, _CLASS_NAMES)

csv_path = '${EVAL_CSV}'
file_exists = os.path.isfile(csv_path)
_, _, accuracy = acc(sorted_results)
_, _, clean_accuracy = clean_acc(sorted_results, _CLASS_NAMES)
_, _, pass_accuracy = pass_acc(sorted_results)
_, _, maj_accuracy = majority_acc(sorted_results)

with open(csv_path, 'a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow([
            'Source', 'Adapter', 'SNR', 'N_Prompts',
            'Accuracy', 'Clean_Accuracy', 'Pass@1', 'Majority',
        ])
    writer.writerow([
        '${PREDICTION_SOURCE}', 'vanilla', '${snr}', n_unique,
        f'{accuracy:.4f}', f'{clean_accuracy:.4f}',
        f'{pass_accuracy:.4f}', f'{maj_accuracy:.4f}',
    ])
print(f'  → Results appended to {csv_path}')
"
done

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
log_step "DONE — Vanilla DeepSeek + Finetuned DenoMAE2 evaluation complete"
echo ""
if [[ -f "$EVAL_CSV" ]]; then
    echo "  Results:"
    column -t -s',' "$EVAL_CSV"
fi
