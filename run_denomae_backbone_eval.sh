#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# run_denomae_backbone_eval.sh — parameterized version of
# run_denomae_ft_vanilla_eval.sh / run_denomae_metatask_vanilla_eval.sh.
#
# Runs the full DiSC-AMC pipeline (FAISS index -> top-k -> pkls -> vanilla
# DeepSeek-R1 inference) for one encoder checkpoint across a given set of
# RadioML SNRs. Takes its config from environment variables so the same
# script can drive several backbone comparisons without copy-pasting.
#
# Required env vars:
#   ENCODER_WEIGHTS       path to the classifier checkpoint (.pth)
#   PREDICTION_SOURCE     must be a registered key in src/naming.py SOURCES
#   FAISS_TAG             unique tag for this backbone's FAISS index files
#   OUTPUT_DIR            where results/pkls/jsons for this run go
#   CONDITION_LABEL       label written into the results CSV's Adapter column
# Optional:
#   EVAL_SNR_LEVELS       space-separated, default all 5
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

: "${ENCODER_WEIGHTS:?ENCODER_WEIGHTS not set}"
: "${PREDICTION_SOURCE:?PREDICTION_SOURCE not set}"
: "${FAISS_TAG:?FAISS_TAG not set}"
: "${OUTPUT_DIR:?OUTPUT_DIR not set}"
: "${CONDITION_LABEL:?CONDITION_LABEL not set}"

DATA_ROOT="/mnt/d/Rowan/discrete-llm-amc/data/RadioML"
MODEL_DIR="/mnt/d/Rowan/discrete-llm-amc/models"

BACKBONE="denomae"
NUM_CLASSES=24
IMAGE_SIZE=224
N_COMPONENTS=10
BATCH_SIZE=32

DATASET_TYPE="radioml"
N_BINS=5
TOP_K=5
KNN_K=50
FEATURE_TYPE="embeddings"
USE_RAG=true
RAG_K=10
MIN_RAG_CLASSES=0
PROMPT_VERSION="v2"

EVAL_SNR_LEVELS=(${EVAL_SNR_LEVELS:-"snr_-20db" "snr_-10db" "snr_0db" "snr_10db" "snr_20db"})

UNSLOTH_MODEL="unsloth/DeepSeek-R1-Distill-Qwen-7B"
PROMPT_TYPE="discret_prompts"
NUM_TRIES=1
INFERENCE_BATCH_SIZE=8
MAX_NEW_TOKENS=512

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

if [[ ! -f "$ENCODER_WEIGHTS" ]]; then
    echo -e "${RED}ERROR: Encoder weights not found: ${ENCODER_WEIGHTS}${NC}"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
EVAL_CSV="${OUTPUT_DIR}/${CONDITION_LABEL}_results.csv"

_raw_json() { echo "top${TOP_K}_${PREDICTION_SOURCE}_predictions.json"; }
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
echo "║  CONDITION:       ${CONDITION_LABEL}"
echo "║  ENCODER_WEIGHTS: ${ENCODER_WEIGHTS}"
echo "║  PRED_SOURCE:     ${PREDICTION_SOURCE}"
echo "║  EVAL_SNR_LEVELS: ${EVAL_SNR_LEVELS[*]}"
echo "║  OUTPUT_DIR:      ${OUTPUT_DIR}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

for snr in "${EVAL_SNR_LEVELS[@]}"; do
    log_step "DATA PIPELINE — ${CONDITION_LABEL} × ${snr}"
    SNR_DIR="${DATA_ROOT}/${snr}"

    img_dir_train="${SNR_DIR}/train/img"
    img_dir_test="${SNR_DIR}/test/img"
    if [[ -d "$img_dir_train" && -d "$img_dir_test" ]]; then
        echo -e "${YELLOW}  ⊘ Constellation images exist — skipping${NC}"
    else
        echo -e "${CYAN}  Generating constellation images for ${snr} …${NC}"
        python -m src.representation_learning.generate_radioml_images \
            --data_root "$DATA_ROOT" --snr_levels "$snr"
    fi

    FAISS_INDEX_PATH="${SNR_DIR}/train/faiss_knn_${FAISS_TAG}"
    if [[ -f "${FAISS_INDEX_PATH}.index" ]]; then
        echo -e "${YELLOW}  ⊘ FAISS index (${FAISS_TAG}) exists — skipping${NC}"
    else
        echo -e "${CYAN}  Building FAISS index (${FAISS_TAG}) for ${snr} …${NC}"
        python -m src.representation_learning.inference build_faiss \
            --backbone "$BACKBONE" --weights "$ENCODER_WEIGHTS" \
            --num_classes "$NUM_CLASSES" --train_path "${SNR_DIR}/train" \
            --output "$FAISS_INDEX_PATH" --image_size "$IMAGE_SIZE"
    fi

    RAW_JSON="$(_raw_json)"
    if [[ -f "${SNR_DIR}/${RAW_JSON}" ]]; then
        echo -e "${YELLOW}  ⊘ Top-k predictions exist — skipping${NC}"
    else
        echo -e "${CYAN}  Generating top-k predictions for ${snr} …${NC}"
        python -m src.representation_learning.inference predict \
            --backbone "$BACKBONE" --weights "$ENCODER_WEIGHTS" \
            --num_classes "$NUM_CLASSES" --dataset_path "${SNR_DIR}/test" \
            --topk "$TOP_K" --output "${SNR_DIR}/${RAW_JSON}" \
            --image_size "$IMAGE_SIZE" --faiss_index_path "$FAISS_INDEX_PATH" \
            --knn_k "$KNN_K" --fill_topk
    fi

    CONVERTED_JSON="$(_converted_json)"
    if [[ -f "${SNR_DIR}/${CONVERTED_JSON}" ]]; then
        echo -e "${YELLOW}  ⊘ Converted predictions exist — skipping${NC}"
    else
        echo -e "${CYAN}  Converting prediction keys for ${snr} …${NC}"
        python -m src.representation_learning.convert_predictions \
            --input "${SNR_DIR}/${RAW_JSON}" --output "${SNR_DIR}/${CONVERTED_JSON}"
    fi

    train_pkl="${SNR_DIR}/$(_pkl_name train "$snr")"
    if [[ -f "$train_pkl" ]]; then
        echo -e "${YELLOW}  ⊘ Train pkl exists — skipping${NC}"
    else
        echo -e "${CYAN}  Building train pkl for ${snr} …${NC}"
        python -m src.prompt.generated_dataset \
            --mode train --dataset_folder="$snr" --noise_mode "$snr" \
            --n_bins "$N_BINS" --top_k "$TOP_K" --prediction_source "$PREDICTION_SOURCE" \
            --data_root "$DATA_ROOT" --feature_type embeddings \
            --encoder_weights "$ENCODER_WEIGHTS" --backbone "$BACKBONE" \
            --n_components "$N_COMPONENTS" --batch_size "$BATCH_SIZE" \
            --num_classes "$NUM_CLASSES" --use_rag --rag_k "$RAG_K" \
            --min_classes "$MIN_RAG_CLASSES" --prompt_version "$PROMPT_VERSION" \
            --dataset_type "$DATASET_TYPE"
    fi

    test_pkl="${SNR_DIR}/$(_pkl_name test "$snr")"
    if [[ -f "$test_pkl" ]]; then
        echo -e "${YELLOW}  ⊘ Test pkl exists — skipping${NC}"
    else
        echo -e "${CYAN}  Building test pkl for ${snr} …${NC}"
        python -m src.prompt.generated_dataset \
            --mode test --dataset_folder="$snr" --noise_mode "$snr" \
            --n_bins "$N_BINS" --top_k "$TOP_K" --prediction_source "$PREDICTION_SOURCE" \
            --data_root "$DATA_ROOT" --feature_type embeddings \
            --encoder_weights "$ENCODER_WEIGHTS" --backbone "$BACKBONE" \
            --n_components "$N_COMPONENTS" --batch_size "$BATCH_SIZE" \
            --num_classes "$NUM_CLASSES" --use_rag --rag_k "$RAG_K" \
            --min_classes "$MIN_RAG_CLASSES" --prompt_version "$PROMPT_VERSION" \
            --dataset_type "$DATASET_TYPE"
    fi
done

log_step "Phase 6 — Vanilla DeepSeek-R1 evaluation (no LoRA adapter) — ${CONDITION_LABEL}"

for snr in "${EVAL_SNR_LEVELS[@]}"; do
    log_step "EVAL — ${CONDITION_LABEL} × ${snr}"

    test_pkl="${DATA_ROOT}/${snr}/$(_pkl_name test "$snr")"
    if [[ ! -f "$test_pkl" ]]; then
        echo -e "${RED}  Test pkl missing: ${test_pkl} — skipping${NC}"
        continue
    fi

    python -c "
from src.evaluation.unsloth_eval import main
main(
    dataset_folder='${snr}', prompt_type='${PROMPT_TYPE}', model_name='${UNSLOTH_MODEL}',
    noise_mode='${snr}', n_bins=${N_BINS}, top_k=${TOP_K}, num_tries=${NUM_TRIES},
    prediction_source='${PREDICTION_SOURCE}', feature_type='${FEATURE_TYPE}',
    n_components=${N_COMPONENTS}, cache_dir='${MODEL_DIR}', data_root='${DATA_ROOT}',
    output_dir='${OUTPUT_DIR}', adapter_path=None,
    inference_batch_size=${INFERENCE_BATCH_SIZE}, max_new_tokens=${MAX_NEW_TOKENS},
    prompt_version='${PROMPT_VERSION}', backbone='${BACKBONE}',
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
    dataset_folder='${snr}', prompt_type='${PROMPT_TYPE}', model_name='${UNSLOTH_MODEL}',
    noise_mode='${snr}', n_bins=${N_BINS}, top_k=${TOP_K}, prediction_source='${PREDICTION_SOURCE}',
    feature_type='${FEATURE_TYPE}', n_components=${N_COMPONENTS}, output_dir='${OUTPUT_DIR}',
    prompt_version='${PROMPT_VERSION}', backbone='${BACKBONE}',
)
sorted_results = sort_results_by_prompt(results)
n_unique = len(get_unique_prompts(results))
print(f'[${CONDITION_LABEL} x ${snr}]  Unique prompts: {n_unique}')
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
        writer.writerow(['Source', 'Adapter', 'SNR', 'N_Prompts', 'Accuracy', 'Clean_Accuracy', 'Pass@1', 'Majority'])
    writer.writerow(['${PREDICTION_SOURCE}', '${CONDITION_LABEL}', '${snr}', n_unique,
                      f'{accuracy:.4f}', f'{clean_accuracy:.4f}', f'{pass_accuracy:.4f}', f'{maj_accuracy:.4f}'])
print(f'  -> Results appended to {csv_path}')
"
done

log_step "DONE — ${CONDITION_LABEL}"
[[ -f "$EVAL_CSV" ]] && column -t -s',' "$EVAL_CSV"
