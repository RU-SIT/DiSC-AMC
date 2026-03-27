#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# run_radioml_native_finetuning_v2.sh
#
# End-to-end pipeline:
#   Phase 0:  Finetune DenoMAE2 classifier on RadioML train data
#   Phase 1:  Generate constellation diagram images (skip if exist)
#   Phase 2:  Build FAISS index per SNR (using finetuned DenoMAE2)
#   Phase 3:  Generate top-k predictions (FAISS kNN voting)
#   Phase 4:  Convert prediction keys (.png → .npy)
#   Phase 5:  Build train + test pkls + RAG index
#   Phase 6:  Evaluate with base LLM (no LLM finetuning)
#   Phase 7:  QLoRA finetune LLM (anti-overfitting)
#   Phase 8:  Evaluate with finetuned LLM
#
# Anti-overfitting controls on both DenoMAE2 and LLM finetuning.
# Results go to two separate CSVs.
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ─── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

DATA_ROOT="/mnt/d/Rowan/discrete-llm-amc/data/RadioML"
EXP_DIR="${PROJECT_ROOT}/exp"
MODEL_DIR="/mnt/d/Rowan/discrete-llm-amc/models"
OUTPUT_DIR="${EXP_DIR}/radioml_denomae_ft_eval"

# ─── DenoMAE2 pretrained weights (base, before RadioML finetuning) ────────
DENOMAE_PRETRAINED="${PROJECT_ROOT}/models/denoMAE2_best.pth"
DENOMAE_FT_DIR="${EXP_DIR}/denomae_ft_radioml"
DENOMAE_FT_WEIGHTS="${DENOMAE_FT_DIR}/denoMAE2_rml_finetunedClassifier.pth"

# ─── DenoMAE2 finetuning hyperparameters (anti-overfitting) ──────────────
DENOMAE_EPOCHS=200
DENOMAE_LR="5e-3"
DENOMAE_WEIGHT_DECAY=0.05
DENOMAE_BATCH_SIZE=32
DENOMAE_PATIENCE=5          # early stopping: stop after 5 eval rounds w/o improvement
DENOMAE_EVAL_EVERY=5        # evaluate every epoch
DENOMAE_GPU="0,1"

# ─── Encoder config (after finetuning) ───────────────────────────────────
BACKBONE="denomae"
ENCODER_WEIGHTS="${DENOMAE_FT_WEIGHTS}"
NUM_CLASSES=24
IMAGE_SIZE=224               # DenoMAE2 expects 224×224
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

# ─── Which SNR levels to use for finetuning (train) ──────────────────────
FT_SNR_LEVELS=("snr_0db" "snr_10db" "snr_20db")

# ─── Which SNR levels to evaluate ────────────────────────────────────────
EVAL_SNR_LEVELS=("snr_-20db" "snr_-10db" "snr_0db" "snr_10db" "snr_20db")

# ─── FAISS tag for index files (separate from existing denomae indices) ──
FAISS_TAG="denomae_ft"

# ─── LLM / finetuning configuration (anti-overfitting) ───────────────────
UNSLOTH_MODEL="unsloth/DeepSeek-R1-Distill-Qwen-7B"
PROMPT_TYPE="discret_prompts"
PROMPT_STYLE="discret"
USE_THINKING=true
COMPLETION_VERSION="${PROMPT_VERSION}"

LORA_R=8
LORA_ALPHA=8
EPOCHS=3
FT_BATCH_SIZE=2
GRAD_ACCUM=4
LR="2e-5"
MAX_SEQ_LEN=1024
WARMUP_STEPS=10
SEED=3407
VAL_SPLIT=0.15
SAVE_MERGED=false
LR_SCHEDULER="cosine"

# ─── Evaluation ──────────────────────────────────────────────────────────
NUM_TRIES=1
INFERENCE_BATCH_SIZE=8
MAX_NEW_TOKENS=512

# ─── RUN control ─────────────────────────────────────────────────────────
RUN_DENOMAE_FINETUNE=true   # Phase 0: finetune DenoMAE2
RUN_DATA_PIPELINE=true      # Phases 1-5: build FAISS/pkls
RUN_BASE_EVAL=true          # Phase 6: evaluate with base LLM adapter
RUN_LLM_FINETUNE=true       # Phase 7: QLoRA finetune LLM
RUN_FT_EVAL=true            # Phase 8: evaluate with finetuned LLM

# ─── Base LLM adapter (for Phase 6 evaluation) ──────────────────────────
BASE_ADAPTER="${EXP_DIR}/ft_unlabeled_10k_faiss_filled_emb10_discret_DeepSeek-R1-Distill-Qwen-7B_ep5_r16_v01/lora_adapter"
BASE_ADAPTER_NAME="faiss_filled_emb10"

# ═══════════════════════════════════════════════════════════════════════════
# Derived variables / helpers
# ═══════════════════════════════════════════════════════════════════════════
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

_shorten_model() {
    local m="$1"
    m="${m#unsloth/}"
    m="${m%%-unsloth*}"
    echo "$m"
}
MODEL_SHORT="$(_shorten_model "$UNSLOTH_MODEL")"
N_COMPONENTS_EVAL=$N_COMPONENTS

# FAISS / pkl naming helpers
_raw_json() {
    echo "top${TOP_K}_${PREDICTION_SOURCE}_predictions.json"
}
_converted_json() {
    echo "ntop${TOP_K}_${PREDICTION_SOURCE}_predictions.json"
}
_pkl_name() {
    local mode="$1" snr="$2"
    local bb_tag=""
    [[ "$BACKBONE" != "dino" ]] && bb_tag="_${BACKBONE}"
    echo "${mode}_${PREDICTION_SOURCE}${bb_tag}_emb${N_COMPONENTS}_${snr}_${N_BINS}_${TOP_K}_data.pkl"
}
_needs_fill() {
    [[ "$PREDICTION_SOURCE" == *filled* ]]
}

# Prediction source faiss_filled_denomae_ft is registered in naming.py

mkdir -p "$OUTPUT_DIR"
BASE_EVAL_CSV="${OUTPUT_DIR}/radioml_denomae_ft_eval_results.csv"
FT_EVAL_CSV="${OUTPUT_DIR}/radioml_denomae_ft_antiovf_results.csv"

# Merged pkl path for LLM finetuning
_bb_tag=""
[[ "$BACKBONE" != "dino" ]] && _bb_tag="_${BACKBONE}"
MERGED_PKL_PATH="${DATA_ROOT}/train_${PREDICTION_SOURCE}${_bb_tag}_emb${N_COMPONENTS}_merged_${N_BINS}_${TOP_K}_data.pkl"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  RadioML DenoMAE2 + LLM Full Pipeline                      ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  DATA_ROOT:          ${DATA_ROOT}"
echo "║  DENOMAE_BASE:       ${DENOMAE_PRETRAINED}"
echo "║  DENOMAE_FT:         epoch=${DENOMAE_EPOCHS}, lr=${DENOMAE_LR}, patience=${DENOMAE_PATIENCE}"
echo "║  NUM_CLASSES:        ${NUM_CLASSES}"
echo "║  BACKBONE:           ${BACKBONE}"
echo "║  FAISS_TAG:          ${FAISS_TAG}"
echo "║  PRED_SOURCE:        ${PREDICTION_SOURCE}"
echo "║  MODEL:              ${UNSLOTH_MODEL}"
echo "║  FT_SNR_LEVELS:      ${FT_SNR_LEVELS[*]}"
echo "║  EVAL_SNR_LEVELS:    ${EVAL_SNR_LEVELS[*]}"
echo "║  LLM LORA:           r=${LORA_R}, alpha=${LORA_ALPHA}"
echo "║  LLM TRAINING:       ${EPOCHS}ep, bs=${FT_BATCH_SIZE}×${GRAD_ACCUM}, lr=${LR}"
echo "║  LR_SCHEDULER:       ${LR_SCHEDULER}"
echo "║  BASE_EVAL_CSV:      ${BASE_EVAL_CSV}"
echo "║  FT_EVAL_CSV:        ${FT_EVAL_CSV}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Phase 0: Finetune DenoMAE2 on RadioML
# ═══════════════════════════════════════════════════════════════════════════
if [[ "$RUN_DENOMAE_FINETUNE" == "true" ]]; then
    log_step "Phase 0 — Finetune DenoMAE2 on RadioML"
    mkdir -p "$DENOMAE_FT_DIR"

    # ── Build merged ImageFolder train/test directories via symlinks ─────
    MERGED_TRAIN_DIR="${DENOMAE_FT_DIR}/merged_train"
    MERGED_TEST_DIR="${DENOMAE_FT_DIR}/merged_test"

    if [[ -d "$MERGED_TRAIN_DIR" ]]; then
        echo -e "${YELLOW}  ⊘ Merged train dir already exists — skipping symlink step${NC}"
    else
        echo "  Building merged ImageFolder from ${FT_SNR_LEVELS[*]} …"
        mkdir -p "$MERGED_TRAIN_DIR"
        for snr in "${FT_SNR_LEVELS[@]}"; do
            img_dir="${DATA_ROOT}/${snr}/train/img"
            if [[ ! -d "$img_dir" ]]; then
                echo "ERROR: Missing train img directory: $img_dir" >&2
                exit 1
            fi
            # Parse class name from filename: CLASS_sample_N.png → CLASS
            for f in "${img_dir}"/*.png; do
                [[ -f "$f" ]] || continue
                base="$(basename "$f")"
                # Extract class: everything before the last _sample_
                class_name="${base%_sample_*}"
                target="${MERGED_TRAIN_DIR}/${class_name}"
                mkdir -p "$target"
                # Prefix with SNR to avoid filename collisions across SNR levels
                ln -sf "$f" "${target}/${snr}_${base}" 2>/dev/null || true
            done
        done
        echo "  → Merged train dir: $(find "$MERGED_TRAIN_DIR" -type l | wc -l) symlinks"
        echo "  → Classes: $(ls "$MERGED_TRAIN_DIR" | wc -l)"
    fi

    # Build test ImageFolder from a single SNR level (highest quality)
    TEST_SNR_FOR_VAL="${FT_SNR_LEVELS[-1]}"  # last = snr_20db
    DENOMAE_TEST_PATH="${DENOMAE_FT_DIR}/merged_test"

    if [[ -d "$DENOMAE_TEST_PATH" ]]; then
        echo -e "${YELLOW}  ⊘ Test ImageFolder already exists — skipping${NC}"
    else
        echo "  Building test ImageFolder from ${TEST_SNR_FOR_VAL}/test/img …"
        mkdir -p "$DENOMAE_TEST_PATH"
        test_img_dir="${DATA_ROOT}/${TEST_SNR_FOR_VAL}/test/img"
        for f in "${test_img_dir}"/*.png; do
            [[ -f "$f" ]] || continue
            base="$(basename "$f")"
            class_name="${base%_sample_*}"
            target="${DENOMAE_TEST_PATH}/${class_name}"
            mkdir -p "$target"
            ln -sf "$f" "${target}/${base}" 2>/dev/null || true
        done
        echo "  → Test dir: $(find "$DENOMAE_TEST_PATH" -type l | wc -l) symlinks"
    fi

    echo "  DenoMAE2 train path: ${MERGED_TRAIN_DIR}"
    echo "  DenoMAE2 test path:  ${DENOMAE_TEST_PATH}"

    # ── Run DenoMAE2 finetuning ──────────────────────────────────────────
    if [[ -f "$DENOMAE_FT_WEIGHTS" ]]; then
        echo -e "${YELLOW}  ⊘ Finetuned DenoMAE2 weights already exist — skipping${NC}"
        echo "    ${DENOMAE_FT_WEIGHTS}"
    else
        echo -e "${CYAN}  Training DenoMAE2 classifier (${DENOMAE_EPOCHS} epochs, lr=${DENOMAE_LR}, patience=${DENOMAE_PATIENCE}) …${NC}"
        python -m src.denoMAE2.finetune \
            --train_data_path "$MERGED_TRAIN_DIR" \
            --test_data_path "$DENOMAE_TEST_PATH" \
            --image_size 224 224 \
            --patch_size 16 \
            --embed_dim 768 \
            --decoder_embed_dim 512 \
            --encoder_depth 12 \
            --decoder_depth 8 \
            --encoder_num_heads 12 \
            --decoder_num_heads 8 \
            --batch_size "$DENOMAE_BATCH_SIZE" \
            --num_epochs "$DENOMAE_EPOCHS" \
            --learning_rate "$DENOMAE_LR" \
            --num_classes "$NUM_CLASSES" \
            --weight_decay "$DENOMAE_WEIGHT_DECAY" \
            --pretrained_model_path "$DENOMAE_PRETRAINED" \
            --output_model_path "$DENOMAE_FT_WEIGHTS" \
            --gpu "$DENOMAE_GPU" \
            --patience "$DENOMAE_PATIENCE" \
            --eval_every "$DENOMAE_EVAL_EVERY" \
            --load

        echo ""
        echo "  DenoMAE2 finetuned → ${DENOMAE_FT_WEIGHTS}"
    fi
else
    log_step "Phase 0 — Skipped (RUN_DENOMAE_FINETUNE=false)"
    if [[ ! -f "$DENOMAE_FT_WEIGHTS" ]]; then
        echo -e "${RED}  WARNING: Finetuned weights not found at ${DENOMAE_FT_WEIGHTS}${NC}"
        echo -e "${RED}  Phases 1-8 may fail. Set RUN_DENOMAE_FINETUNE=true to train.${NC}"
    fi
fi

# Update encoder weights to the finetuned version
ENCODER_WEIGHTS="${DENOMAE_FT_WEIGHTS}"

# ═══════════════════════════════════════════════════════════════════════════
# Phases 1–5: Data pipeline (per SNR)
# ═══════════════════════════════════════════════════════════════════════════
if [[ "$RUN_DATA_PIPELINE" == "true" ]]; then
for snr in "${EVAL_SNR_LEVELS[@]}"; do
    log_step "DATA PIPELINE — ${snr}"
    SNR_DIR="${DATA_ROOT}/${snr}"

    # ── Phase 1: Generate constellation diagram images ───────────────
    img_dir_train="${SNR_DIR}/train/img"
    img_dir_test="${SNR_DIR}/test/img"
    if [[ -d "$img_dir_train" && -d "$img_dir_test" ]]; then
        echo -e "${YELLOW}  ⊘ Constellation images already exist — skipping Phase 1${NC}"
    else
        echo -e "${CYAN}  Phase 1: Generating constellation images for ${snr} …${NC}"
        python -m src.representation_learning.generate_radioml_images \
            --data_root "$DATA_ROOT" \
            --snr_levels "$snr"
    fi

    # ── Phase 2: Build FAISS index (finetuned DenoMAE2 encoder) ──────
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
    CONVERTED_JSON="$(_converted_json)"

    fill_flag=""
    if _needs_fill; then
        fill_flag="--fill_topk"
    fi
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
            $fill_flag
    fi

    # ── Phase 4: Convert .png keys → .npy keys ──────────────────────
    if [[ -f "${SNR_DIR}/${CONVERTED_JSON}" ]]; then
        echo -e "${YELLOW}  ⊘ Converted predictions exist — skipping Phase 4${NC}"
    else
        echo -e "${CYAN}  Phase 4: Converting prediction keys for ${snr} …${NC}"
        python -m src.representation_learning.convert_predictions \
            --input  "${SNR_DIR}/${RAW_JSON}" \
            --output "${SNR_DIR}/${CONVERTED_JSON}"
    fi

    # ── Phase 5: Build train pkl + RAG index + test pkl ──────────────
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
else
    log_step "Phases 1-5 — Skipped (RUN_DATA_PIPELINE=false)"
fi

# ═══════════════════════════════════════════════════════════════════════════
# Phase 6: Evaluate with base LLM adapter (no LLM finetuning)
# ═══════════════════════════════════════════════════════════════════════════
if [[ "$RUN_BASE_EVAL" == "true" ]]; then
log_step "Phase 6 — Base LLM evaluation"

if [[ ! -d "$BASE_ADAPTER" ]]; then
    echo -e "${RED}  WARNING: Base adapter not found at ${BASE_ADAPTER}${NC}"
    echo -e "${RED}  Skipping Phase 6.${NC}"
else
for snr in "${EVAL_SNR_LEVELS[@]}"; do
    log_step "BASE EVAL — ${BASE_ADAPTER_NAME} × ${snr}"

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
    n_components=${N_COMPONENTS_EVAL},
    cache_dir='${MODEL_DIR}',
    data_root='${DATA_ROOT}',
    output_dir='${OUTPUT_DIR}',
    adapter_path='${BASE_ADAPTER}',
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
    n_components=${N_COMPONENTS_EVAL},
    output_dir='${OUTPUT_DIR}',
    prompt_version='${PROMPT_VERSION}',
    backbone='${BACKBONE}',
)
sorted_results = sort_results_by_prompt(results)
n_unique = len(get_unique_prompts(results))
print(f'[BASE × ${BASE_ADAPTER_NAME} × ${snr}]  Unique prompts: {n_unique}')
print_metrics(sorted_results, _CLASS_NAMES)

csv_path = '${BASE_EVAL_CSV}'
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
        '${PREDICTION_SOURCE}', '${BASE_ADAPTER_NAME}', '${snr}', n_unique,
        f'{accuracy:.4f}', f'{clean_accuracy:.4f}',
        f'{pass_accuracy:.4f}', f'{maj_accuracy:.4f}',
    ])
print(f'  → Results appended to {csv_path}')
"
done
fi
else
    log_step "Phase 6 — Skipped (RUN_BASE_EVAL=false)"
fi

# ═══════════════════════════════════════════════════════════════════════════
# Phase 7: QLoRA finetune LLM
# ═══════════════════════════════════════════════════════════════════════════
FT_BASE_NAME="ft_radioml_${PREDICTION_SOURCE}_${FAISS_TAG}_emb${N_COMPONENTS}_${PROMPT_STYLE}_${MODEL_SHORT}_ep${EPOCHS}_r${LORA_R}_antiovf"

if [[ "$RUN_LLM_FINETUNE" == "true" ]]; then
    log_step "Phase 7 — QLoRA LLM finetuning"

    # Uses MERGED_PKL_PATH defined at top of script

    if [[ -f "$MERGED_PKL_PATH" ]]; then
        echo -e "${YELLOW}  ⊘ Merged train pkl already exists — skipping${NC}"
        echo "    ${MERGED_PKL_PATH}"
    else
        PKL_PATHS=""
        for snr in "${FT_SNR_LEVELS[@]}"; do
            pkl="${DATA_ROOT}/${snr}/$(_pkl_name train "$snr")"
            if [[ ! -f "$pkl" ]]; then
                echo "ERROR: Missing train pkl: $pkl" >&2
                exit 1
            fi
            PKL_PATHS="${PKL_PATHS} ${pkl}"
        done

        echo "  Merging ${#FT_SNR_LEVELS[@]} train pkls …"
        python -c "
import pickle, sys

paths = '''${PKL_PATHS}'''.split()
print(f'  Loading {len(paths)} pkls …')

LIST_KEYS = [
    'signal_paths', 'signals', 'stats', 'discret_stats',
    'labels', 'snrs', 'old_prompts', 'old_discret_prompts',
]

merged = None
for p in paths:
    with open(p, 'rb') as f:
        d = pickle.load(f)
    if merged is None:
        merged = {k: list(d[k]) if k in LIST_KEYS else d[k] for k in d}
        merged['prompts'] = list(d.get('prompts', []))
        merged['discret_prompts'] = list(d.get('discret_prompts', []))
    else:
        for k in LIST_KEYS:
            merged[k].extend(d[k])
        merged['prompts'].extend(d.get('prompts', []))
        merged['discret_prompts'].extend(d.get('discret_prompts', []))

merged['num_samples'] = len(merged['labels'])
merged['#classes'] = len(set(merged['labels']))
merged['#snr'] = len(set(merged['snrs']))

out = '${MERGED_PKL_PATH}'
with open(out, 'wb') as f:
    pickle.dump(merged, f)
print(f'  Merged pkl: {merged[\"num_samples\"]} samples, '
      f'{merged[\"#classes\"]} classes, {merged[\"#snr\"]} SNR levels')
print(f'  → {out}')
"
    fi

    # ── 7b. Setup experiment directory ───────────────────────────────
    _FT_VERSION=1
    while [[ -d "${EXP_DIR}/${FT_BASE_NAME}_v$(printf '%02d' $_FT_VERSION)" ]]; do
        _FT_VERSION=$((_FT_VERSION + 1))
    done
    FT_EXP_NAME="${FT_BASE_NAME}_v$(printf '%02d' $_FT_VERSION)"
    FT_OUTPUT_DIR="${EXP_DIR}/${FT_EXP_NAME}"
    ADAPTER_DIR="${FT_OUTPUT_DIR}/lora_adapter"
    mkdir -p "$FT_OUTPUT_DIR"
    echo -e "${GREEN}  NEW experiment: ${FT_OUTPUT_DIR}${NC}"

    # ── 7c. Train ────────────────────────────────────────────────────
    log_step "Phase 7c — QLoRA training (${MODEL_SHORT}, r=${LORA_R}, ${EPOCHS}ep, lr=${LR}, sched=${LR_SCHEDULER})"

    think_flag=""
    [[ "$USE_THINKING" != "true" ]] && think_flag="--no_thinking"

    merged_flag=""
    [[ "$SAVE_MERGED" == "true" ]] && merged_flag="--save_merged"

    python -m src.finetuning.train \
        --model_name "$UNSLOTH_MODEL" \
        --pkl_path "$MERGED_PKL_PATH" \
        --output_dir "$FT_OUTPUT_DIR" \
        --prompt_style "$PROMPT_STYLE" \
        --lora_r "$LORA_R" \
        --lora_alpha "$LORA_ALPHA" \
        --epochs "$EPOCHS" \
        --batch_size "$FT_BATCH_SIZE" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --lr "$LR" \
        --max_seq_length "$MAX_SEQ_LEN" \
        --warmup_steps "$WARMUP_STEPS" \
        --seed "$SEED" \
        --cache_dir "$MODEL_DIR" \
        --val_split "$VAL_SPLIT" \
        --completion_version "$COMPLETION_VERSION" \
        --lr_scheduler_type "$LR_SCHEDULER" \
        $think_flag \
        $merged_flag

    echo ""
    echo "  Adapter saved → ${ADAPTER_DIR}"
else
    log_step "Phase 7 — Skipped (RUN_LLM_FINETUNE=false)"

    # Find latest existing experiment
    _v=1; _latest=""
    while [[ -d "${EXP_DIR}/${FT_BASE_NAME}_v$(printf '%02d' $_v)" ]]; do
        _latest="${EXP_DIR}/${FT_BASE_NAME}_v$(printf '%02d' $_v)"
        _v=$((_v + 1))
    done
    if [[ -n "$_latest" ]]; then
        FT_OUTPUT_DIR="$_latest"
        ADAPTER_DIR="${FT_OUTPUT_DIR}/lora_adapter"
        echo -e "${GREEN}  Using existing experiment: ${FT_OUTPUT_DIR}${NC}"
    else
        echo -e "${RED}  WARNING: No existing experiment for ${FT_BASE_NAME}_v* — Phase 8 will fail${NC}"
        FT_OUTPUT_DIR=""
        ADAPTER_DIR=""
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════
# Phase 8: Evaluate with finetuned LLM
# ═══════════════════════════════════════════════════════════════════════════
if [[ "$RUN_FT_EVAL" == "true" && -n "$ADAPTER_DIR" ]]; then
log_step "Phase 8 — Finetuned LLM evaluation"

for snr in "${EVAL_SNR_LEVELS[@]}"; do
    log_step "FT EVAL — ${snr}"

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
    n_components=${N_COMPONENTS_EVAL},
    cache_dir='${MODEL_DIR}',
    data_root='${DATA_ROOT}',
    output_dir='${FT_OUTPUT_DIR}',
    adapter_path='${ADAPTER_DIR}',
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
    n_components=${N_COMPONENTS_EVAL},
    output_dir='${FT_OUTPUT_DIR}',
    prompt_version='${PROMPT_VERSION}',
    backbone='${BACKBONE}',
)
sorted_results = sort_results_by_prompt(results)
n_unique = len(get_unique_prompts(results))
print(f'[FT × ${snr}]  Unique prompts: {n_unique}')
print_metrics(sorted_results, _CLASS_NAMES)

csv_path = '${FT_EVAL_CSV}'
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
        '${PREDICTION_SOURCE}', '${FT_OUTPUT_DIR##*/}', '${snr}', n_unique,
        f'{accuracy:.4f}', f'{clean_accuracy:.4f}',
        f'{pass_accuracy:.4f}', f'{maj_accuracy:.4f}',
    ])
print(f'  → Results appended to {csv_path}')
"
done
else
    if [[ "$RUN_FT_EVAL" != "true" ]]; then
        log_step "Phase 8 — Skipped (RUN_FT_EVAL=false)"
    else
        log_step "Phase 8 — Skipped (no adapter directory)"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
log_step "DONE — Pipeline complete"
echo ""
if [[ -f "$BASE_EVAL_CSV" ]]; then
    echo "  Base LLM eval results:"
    column -t -s',' "$BASE_EVAL_CSV"
    echo ""
fi
if [[ -f "$FT_EVAL_CSV" ]]; then
    echo "  Finetuned LLM eval results:"
    column -t -s',' "$FT_EVAL_CSV"
fi
