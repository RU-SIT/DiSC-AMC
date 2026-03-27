#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# run_radioml_native_eval.sh — RadioML evaluation with RadioML-trained DINO
#
# Same pipeline as run_radioml_eval.sh (cross-eval) but uses the DINOv2
# model trained *on RadioML data* (radioml_dino_classifier.pth) instead
# of the "own"-trained dino_classifier.pth.
#
# For each SNR level:
#   Phase 0:  Generate constellation diagram images (skip if exist)
#   Phase 1:  Build FAISS index from training images (RadioML encoder)
#   Phase 2:  Generate top-k predictions (FAISS kNN voting)
#   Phase 3:  Convert prediction keys (.png → .npy)
#   Phase 4:  Build train pkl + RAG index
#   Phase 5:  Build test pkl (with V2 prompts from FAISS predictions)
#   Phase 6:  Run inference per adapter → compute metrics → append CSV
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ─── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

DATA_ROOT="/mnt/d/Rowan/discrete-llm-amc/data/RadioML"
EXP_DIR="${PROJECT_ROOT}/exp"
OUTPUT_DIR="${EXP_DIR}/radioml_native_eval"
MODEL_DIR="/mnt/d/Rowan/discrete-llm-amc/models"

# ─── Encoder / classifier (RadioML-trained DINOv2) ───────────────────────
BACKBONE="dino"
CLASSIFIER_WEIGHTS="${EXP_DIR}/radioml_dino_classifier.pth"
ENCODER_WEIGHTS="${CLASSIFIER_WEIGHTS}"   # same file, used for embeddings
NUM_CLASSES=24                            # RadioML: 24 modulation classes
IMAGE_SIZE=96
N_COMPONENTS=10
BATCH_SIZE=32

# ─── Dataset generation ──────────────────────────────────────────────────
DATASET_TYPE="radioml"
# Prediction sources to evaluate (both filled and non-filled FAISS with RML encoder)
PREDICTION_SOURCES=("faiss_filled_rml" "faiss_rml")
N_BINS=5
TOP_K=5
KNN_K=50                         # kNN neighbours for FAISS voting
FEATURE_TYPE="embeddings"
USE_RAG=true
RAG_K=10
MIN_RAG_CLASSES=0
PROMPT_VERSION="v2"               # use V2 prompts (FAISS provides top-k)

# ─── Inference ────────────────────────────────────────────────────────────
UNSLOTH_MODEL="unsloth/DeepSeek-R1-Distill-Qwen-7B"
PROMPT_TYPE="discret_prompts"     # V2 discrete prompts (top-k from FAISS)
NUM_TRIES=1
INFERENCE_BATCH_SIZE=8
MAX_NEW_TOKENS=512

# ─── SNR levels to evaluate ──────────────────────────────────────────────
SNR_LEVELS=("snr_-20db" "snr_-10db" "snr_0db" "snr_10db" "snr_20db")

# ─── Adapters finetuned on "own" dataset ──────────────────────────────────
ADAPTERS=(
    "${EXP_DIR}/ft_unlabeled_10k_faiss_emb10_discret_DeepSeek-R1-Distill-Qwen-7B_ep5_r16_v01/lora_adapter"
    # "${EXP_DIR}/ft_unlabeled_10k_faiss_filled_emb10_discret_DeepSeek-R1-Distill-Qwen-7B_ep5_r16_v01/lora_adapter"
)
ADAPTER_NAMES=("faiss_emb10" 
# "faiss_filled_emb10"
)

# ─── Tag for FAISS index files (separate from naming.py convention) ────
FAISS_TAG="rml_dino"   # appended to FAISS index dir name

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_step() {
    echo -e "\n${GREEN}══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  $1${NC}"
    echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}"
}

# Helpers to derive filenames per prediction source
_raw_json() {
    local src="$1"
    echo "top${TOP_K}_${src}_predictions.json"
}
_converted_json() {
    local src="$1"
    echo "ntop${TOP_K}_${src}_predictions.json"
}
_pkl_name() {
    local mode="$1" snr="$2" src="$3"
    local bb_tag=""
    [[ "$BACKBONE" != "dino" ]] && bb_tag="_${BACKBONE}"
    echo "${mode}_${src}${bb_tag}_emb${N_COMPONENTS}_${snr}_${N_BINS}_${TOP_K}_data.pkl"
}
_needs_fill() {
    # Returns 0 (true) if the source name contains "filled"
    [[ "$1" == *filled* ]]
}

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
mkdir -p "$OUTPUT_DIR"
CSV_PATH="${OUTPUT_DIR}/radioml_native_eval_results.csv"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║    RadioML Native-DINO Evaluation                           ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  DATA_ROOT:          ${DATA_ROOT}"
echo "║  CLASSIFIER:         ${CLASSIFIER_WEIGHTS}"
echo "║  NUM_CLASSES:        ${NUM_CLASSES}"
echo "║  PREDICTION_SOURCES: ${PREDICTION_SOURCES[*]}"
echo "║  SNR_LEVELS:         ${SNR_LEVELS[*]}"
echo "║  ADAPTERS:           ${ADAPTER_NAMES[*]}"
echo "║  PROMPT_TYPE:        ${PROMPT_TYPE}"
echo "║  OUTPUT_DIR:         ${OUTPUT_DIR}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ─────────────────────────────────────────────────────────────────────────
# Phases 0–5: Preprocessing (per SNR × per prediction source)
# ─────────────────────────────────────────────────────────────────────────
for snr in "${SNR_LEVELS[@]}"; do
    log_step "PREPROCESSING — ${snr}"
    SNR_DIR="${DATA_ROOT}/${snr}"

    # ── Phase 0: Generate constellation diagram images ───────────────
    img_dir_train="${SNR_DIR}/train/img"
    img_dir_test="${SNR_DIR}/test/img"
    if [[ -d "$img_dir_train" && -d "$img_dir_test" ]]; then
        echo -e "${YELLOW}  ⊘ Constellation images already exist — skipping Phase 0${NC}"
    else
        echo -e "${CYAN}  Phase 0: Generating constellation images for ${snr} …${NC}"
        python -m src.representation_learning.generate_radioml_images \
            --data_root "$DATA_ROOT" \
            --snr_levels "$snr"
    fi

    # ── Phase 1: Build FAISS index (RadioML-trained encoder) ─────────
    FAISS_INDEX_PATH="${SNR_DIR}/train/faiss_knn_${FAISS_TAG}"
    if [[ -f "${FAISS_INDEX_PATH}.index" ]]; then
        echo -e "${YELLOW}  ⊘ FAISS index (${FAISS_TAG}) exists — skipping Phase 1${NC}"
    else
        echo -e "${CYAN}  Phase 1: Building FAISS index (${FAISS_TAG}) for ${snr} …${NC}"
        python -m src.representation_learning.inference build_faiss \
            --backbone "$BACKBONE" \
            --weights "$CLASSIFIER_WEIGHTS" \
            --num_classes "$NUM_CLASSES" \
            --train_path "${SNR_DIR}/train" \
            --output "$FAISS_INDEX_PATH" \
            --image_size "$IMAGE_SIZE"
    fi

    # ── Per-source phases (2–5) ──────────────────────────────────────
    for PRED_SRC in "${PREDICTION_SOURCES[@]}"; do
        echo -e "${CYAN}  ── Source: ${PRED_SRC} ──${NC}"

        RAW_JSON="$(_raw_json "$PRED_SRC")"
        CONVERTED_JSON="$(_converted_json "$PRED_SRC")"

        # ── Phase 2: Top-k predictions ───────────────────────────────
        fill_flag=""
        if _needs_fill "$PRED_SRC"; then
            fill_flag="--fill_topk"
        fi
        if [[ -f "${SNR_DIR}/${RAW_JSON}" ]]; then
            echo -e "${YELLOW}  ⊘ Top-k predictions (${PRED_SRC}) exist — skipping Phase 2${NC}"
        else
            echo -e "${CYAN}  Phase 2: Generating top-k predictions (${PRED_SRC}) for ${snr} …${NC}"
            python -m src.representation_learning.inference predict \
                --backbone "$BACKBONE" \
                --weights "$CLASSIFIER_WEIGHTS" \
                --num_classes "$NUM_CLASSES" \
                --dataset_path "${SNR_DIR}/test" \
                --topk "$TOP_K" \
                --output "${SNR_DIR}/${RAW_JSON}" \
                --image_size "$IMAGE_SIZE" \
                --faiss_index_path "$FAISS_INDEX_PATH" \
                --knn_k "$KNN_K" \
                $fill_flag
        fi

        # ── Phase 3: Convert .png keys → .npy keys ──────────────────
        if [[ -f "${SNR_DIR}/${CONVERTED_JSON}" ]]; then
            echo -e "${YELLOW}  ⊘ Converted predictions (${PRED_SRC}) exist — skipping Phase 3${NC}"
        else
            echo -e "${CYAN}  Phase 3: Converting prediction keys (${PRED_SRC}) for ${snr} …${NC}"
            python -m src.representation_learning.convert_predictions \
                --input  "${SNR_DIR}/${RAW_JSON}" \
                --output "${SNR_DIR}/${CONVERTED_JSON}"
        fi

        # ── Phase 4: Build train pkl + RAG index ─────────────────────
        train_pkl="${SNR_DIR}/$(_pkl_name train "$snr" "$PRED_SRC")"
        if [[ -f "$train_pkl" ]]; then
            echo -e "${YELLOW}  ⊘ Train pkl (${PRED_SRC}) exists — skipping Phase 4${NC}"
        else
            echo -e "${CYAN}  Phase 4: Building train pkl (${PRED_SRC}) for ${snr} …${NC}"
            python -m src.prompt.generated_dataset \
                --mode train \
                --dataset_folder="$snr" \
                --noise_mode "$snr" \
                --n_bins "$N_BINS" \
                --top_k "$TOP_K" \
                --prediction_source "$PRED_SRC" \
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

        # ── Phase 5: Build test pkl (V2 prompts) ─────────────────────
        test_pkl="${SNR_DIR}/$(_pkl_name test "$snr" "$PRED_SRC")"
        if [[ -f "$test_pkl" ]]; then
            echo -e "${YELLOW}  ⊘ Test pkl (${PRED_SRC}) exists — skipping Phase 5${NC}"
        else
            echo -e "${CYAN}  Phase 5: Building test pkl (${PRED_SRC}) for ${snr} …${NC}"
            python -m src.prompt.generated_dataset \
                --mode test \
                --dataset_folder="$snr" \
                --noise_mode "$snr" \
                --n_bins "$N_BINS" \
                --top_k "$TOP_K" \
                --prediction_source "$PRED_SRC" \
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
done

# ─────────────────────────────────────────────────────────────────────────
# Phase 6: Inference + Metrics (per source × per adapter × per SNR)
# ─────────────────────────────────────────────────────────────────────────
for PRED_SRC in "${PREDICTION_SOURCES[@]}"; do
for idx in "${!ADAPTERS[@]}"; do
    adapter="${ADAPTERS[$idx]}"
    adapter_name="${ADAPTER_NAMES[$idx]}"

    for snr in "${SNR_LEVELS[@]}"; do
        log_step "INFERENCE — ${PRED_SRC} × ${adapter_name} × ${snr}"

        # ── 6a. Run inference ─────────────────────────────────────────
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
    prediction_source='${PRED_SRC}',
    feature_type='${FEATURE_TYPE}',
    n_components=${N_COMPONENTS},
    cache_dir='${MODEL_DIR}',
    data_root='${DATA_ROOT}',
    output_dir='${OUTPUT_DIR}',
    adapter_path='${adapter}',
    inference_batch_size=${INFERENCE_BATCH_SIZE},
    max_new_tokens=${MAX_NEW_TOKENS},
    prompt_version='${PROMPT_VERSION}',
    backbone='${BACKBONE}',
)
"

        # ── 6b. Compute metrics + append CSV ──────────────────────────
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
    prediction_source='${PRED_SRC}',
    feature_type='${FEATURE_TYPE}',
    n_components=${N_COMPONENTS},
    output_dir='${OUTPUT_DIR}',
    prompt_version='${PROMPT_VERSION}',
    backbone='${BACKBONE}',
)
sorted_results = sort_results_by_prompt(results)
n_unique = len(get_unique_prompts(results))
print(f'[${PRED_SRC} × ${adapter_name} × ${snr}]  Unique prompts: {n_unique}')
print_metrics(sorted_results, _CLASS_NAMES)

# Append to CSV
csv_path = '${CSV_PATH}'
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
        '${PRED_SRC}', '${adapter_name}', '${snr}', n_unique,
        f'{accuracy:.4f}', f'{clean_accuracy:.4f}',
        f'{pass_accuracy:.4f}', f'{maj_accuracy:.4f}',
    ])
print(f'  → Results appended to {csv_path}')
"
    done
done
done

log_step "DONE — All results saved to ${CSV_PATH}"
echo ""
echo "  Summary:"
column -t -s',' "$CSV_PATH"
