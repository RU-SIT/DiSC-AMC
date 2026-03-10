#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# run_radioml_native_finetuning.sh
#
# QLoRA finetuning of an LLM on RadioML data using the RadioML-native DINO
# encoder embeddings, then evaluation across all SNR levels.
#
# Pipeline:
#   Phase A:  Merge train pkls from selected SNR levels
#   Phase B:  QLoRA finetuning
#   Phase C:  Generate test pkls (per SNR, skip if exist)
#   Phase D:  Inference + metrics (per SNR)
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ─── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

DATA_ROOT="/mnt/d/Rowan/discrete-llm-amc/data/RadioML"
EXP_DIR="${PROJECT_ROOT}/exp"
MODEL_DIR="/mnt/d/Rowan/discrete-llm-amc/models"
OUTPUT_DIR="${EXP_DIR}/radioml_native_eval"

# ─── Encoder (RadioML-trained DINOv2) ─────────────────────────────────────
BACKBONE="dino"
ENCODER_WEIGHTS="${EXP_DIR}/radioml_dino_classifier.pth"
NUM_CLASSES=24
IMAGE_SIZE=96
N_COMPONENTS=10
BATCH_SIZE=32

# ─── Data / pipeline settings ────────────────────────────────────────────
DATASET_TYPE="radioml"
# Prediction sources to finetune + evaluate (both filled and non-filled)
PREDICTION_SOURCES=("faiss_filled_rml" "faiss_rml")
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

# ─── LLM / finetuning configuration ──────────────────────────────────────
UNSLOTH_MODEL="unsloth/DeepSeek-R1-Distill-Qwen-7B"
PROMPT_TYPE="discret_prompts"
PROMPT_STYLE="discret"
USE_THINKING=true
COMPLETION_VERSION="${PROMPT_VERSION}"

LORA_R=16
LORA_ALPHA=16
EPOCHS=5
FT_BATCH_SIZE=2
GRAD_ACCUM=4
LR="5e-5"
MAX_SEQ_LEN=1024
WARMUP_STEPS=5
SEED=3407
VAL_SPLIT=0.1
SAVE_MERGED=false

# ─── Evaluation ──────────────────────────────────────────────────────────
NUM_TRIES=1
INFERENCE_BATCH_SIZE=8
MAX_NEW_TOKENS=512

# ─── RUN control ─────────────────────────────────────────────────────────
RUN_FINETUNE=true    # true → train new adapter;  false → evaluate only

# ═══════════════════════════════════════════════════════════════════════════
# Derived variables / helpers
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

_shorten_model() {
    local m="$1"
    m="${m#unsloth/}"
    m="${m%%-unsloth*}"
    echo "$m"
}
MODEL_SHORT="$(_shorten_model "$UNSLOTH_MODEL")"
N_COMPONENTS_EVAL=$N_COMPONENTS

_test_pkl_path() {
    local src="$1" snr="$2"
    echo "${DATA_ROOT}/${snr}/test_${src}_emb${N_COMPONENTS}_${snr}_${N_BINS}_${TOP_K}_data.pkl"
}

mkdir -p "$OUTPUT_DIR"
CSV_PATH="${OUTPUT_DIR}/radioml_native_ft_results.csv"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║    RadioML Native-DINO Finetuning + Evaluation              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  DATA_ROOT:          ${DATA_ROOT}"
echo "║  ENCODER:            ${ENCODER_WEIGHTS}"
echo "║  NUM_CLASSES:        ${NUM_CLASSES}"
echo "║  PRED_SOURCES:       ${PREDICTION_SOURCES[*]}"
echo "║  MODEL:              ${UNSLOTH_MODEL}"
echo "║  FT_SNR_LEVELS:      ${FT_SNR_LEVELS[*]}"
echo "║  EVAL_SNR_LEVELS:    ${EVAL_SNR_LEVELS[*]}"
echo "║  LORA:               r=${LORA_R}, alpha=${LORA_ALPHA}"
echo "║  TRAINING:           ${EPOCHS}ep, bs=${FT_BATCH_SIZE}×${GRAD_ACCUM}, lr=${LR}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Loop over prediction sources  (each gets its own merge + finetune + eval)
# ═══════════════════════════════════════════════════════════════════════════
for PRED_SRC in "${PREDICTION_SOURCES[@]}"; do

log_step "═══ SOURCE: ${PRED_SRC} ═══"

# ── Derived paths for this source ────────────────────────────────────────
MERGED_PKL_PATH="${DATA_ROOT}/train_${PRED_SRC}_emb${N_COMPONENTS}_merged_${N_BINS}_${TOP_K}_data.pkl"
FT_BASE_NAME="ft_radioml_${PRED_SRC}_emb${N_COMPONENTS}_${PROMPT_STYLE}_${MODEL_SHORT}_ep${EPOCHS}_r${LORA_R}"

# ── Setup experiment directory ───────────────────────────────────────────
if [[ "$RUN_FINETUNE" == "true" ]]; then
    _FT_VERSION=1
    while [[ -d "${EXP_DIR}/${FT_BASE_NAME}_v$(printf '%02d' $_FT_VERSION)" ]]; do
        _FT_VERSION=$((_FT_VERSION + 1))
    done
    FT_EXP_NAME="${FT_BASE_NAME}_v$(printf '%02d' $_FT_VERSION)"
    FT_OUTPUT_DIR="${EXP_DIR}/${FT_EXP_NAME}"
    ADAPTER_DIR="${FT_OUTPUT_DIR}/lora_adapter"
    mkdir -p "$FT_OUTPUT_DIR"
    echo -e "${GREEN}  NEW experiment: ${FT_OUTPUT_DIR}${NC}"
else
    _v=1; _latest=""
    while [[ -d "${EXP_DIR}/${FT_BASE_NAME}_v$(printf '%02d' $_v)" ]]; do
        _latest="${EXP_DIR}/${FT_BASE_NAME}_v$(printf '%02d' $_v)"
        _v=$((_v + 1))
    done
    if [[ -z "$_latest" ]]; then
        echo "ERROR: No existing experiment for ${FT_BASE_NAME}_v*" >&2
        exit 1
    fi
    FT_OUTPUT_DIR="$_latest"
    ADAPTER_DIR="${FT_OUTPUT_DIR}/lora_adapter"
    echo -e "${GREEN}  Existing experiment: ${FT_OUTPUT_DIR}${NC}"
fi
echo "  ADAPTER_DIR:  ${ADAPTER_DIR}"
echo "  MERGED_PKL:   ${MERGED_PKL_PATH}"

# ═══════════════════════════════════════════════════════════════════════════
# Phase A: Merge train pkls from selected SNR levels
# ═══════════════════════════════════════════════════════════════════════════
log_step "Phase A — Merge train pkls (${PRED_SRC}) from ${FT_SNR_LEVELS[*]}"

if [[ -f "$MERGED_PKL_PATH" ]]; then
    echo -e "${YELLOW}  ⊘ Merged train pkl already exists — skipping${NC}"
    echo "    ${MERGED_PKL_PATH}"
else
    PKL_PATHS=""
    for snr in "${FT_SNR_LEVELS[@]}"; do
        pkl="${DATA_ROOT}/${snr}/train_${PRED_SRC}_emb${N_COMPONENTS}_${snr}_${N_BINS}_${TOP_K}_data.pkl"
        if [[ ! -f "$pkl" ]]; then
            echo "ERROR: Missing train pkl: $pkl" >&2
            echo "  Run run_radioml_native_eval.sh first to generate Phase 4 artefacts." >&2
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

# ═══════════════════════════════════════════════════════════════════════════
# Phase B: QLoRA finetuning
# ═══════════════════════════════════════════════════════════════════════════
if [[ "$RUN_FINETUNE" == "true" ]]; then
    log_step "Phase B — QLoRA finetuning (${PRED_SRC}, ${MODEL_SHORT}, r=${LORA_R}, ${EPOCHS}ep)"

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
        $think_flag \
        $merged_flag

    echo ""
    echo "  Adapter saved → ${ADAPTER_DIR}"
else
    log_step "Phase B — Skipped (RUN_FINETUNE=false)"
    echo "  Using existing adapter: ${ADAPTER_DIR}"
fi

# ═══════════════════════════════════════════════════════════════════════════
# Phase C+D: Test pkl generation + inference + metrics (per SNR)
# ═══════════════════════════════════════════════════════════════════════════
for snr in "${EVAL_SNR_LEVELS[@]}"; do
    log_step "EVAL — ${PRED_SRC} × ${snr}"
    SNR_DIR="${DATA_ROOT}/${snr}"

    # ── Phase C: Generate test pkl if needed ─────────────────────────
    test_pkl="$(_test_pkl_path "$PRED_SRC" "$snr")"
    if [[ -f "$test_pkl" ]]; then
        echo -e "${YELLOW}  ⊘ Test pkl exists — skipping Phase C${NC}"
    else
        echo -e "${CYAN}  Phase C: Building test pkl (${PRED_SRC}) for ${snr} …${NC}"
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

    # ── Phase D: Inference + metrics ─────────────────────────────────
    echo -e "${CYAN}  Phase D: Running inference (${PRED_SRC}) for ${snr} …${NC}"
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
    n_components=${N_COMPONENTS_EVAL},
    cache_dir='${MODEL_DIR}',
    data_root='${DATA_ROOT}',
    output_dir='${FT_OUTPUT_DIR}',
    adapter_path='${ADAPTER_DIR}',
    inference_batch_size=${INFERENCE_BATCH_SIZE},
    max_new_tokens=${MAX_NEW_TOKENS},
    prompt_version='${PROMPT_VERSION}',
)
"

    # ── Metrics + CSV ────────────────────────────────────────────────
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
    n_components=${N_COMPONENTS_EVAL},
    output_dir='${FT_OUTPUT_DIR}',
    prompt_version='${PROMPT_VERSION}',
)
sorted_results = sort_results_by_prompt(results)
n_unique = len(get_unique_prompts(results))
print(f'[${PRED_SRC} × ${snr}]  Unique prompts: {n_unique}')
print_metrics(sorted_results, _CLASS_NAMES)

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
        '${PRED_SRC}', '${FT_OUTPUT_DIR##*/}', '${snr}', n_unique,
        f'{accuracy:.4f}', f'{clean_accuracy:.4f}',
        f'{pass_accuracy:.4f}', f'{maj_accuracy:.4f}',
    ])
print(f'  → Results appended to {csv_path}')
"
done

done   # ── end PREDICTION_SOURCES loop ──

log_step "DONE — All results saved to ${CSV_PATH}"
echo ""
echo "  Summary:"
column -t -s',' "$CSV_PATH"
