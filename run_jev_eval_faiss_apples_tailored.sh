#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# DiSC-AMC — Jev (BeatAPI) Evaluation Runner — TAILORED PROMPT
# apples-to-apples with the paper's 71.5% DeepSeek-R1-Distill-Qwen-7B number
# ═══════════════════════════════════════════════════════════════════════════
#
# Identical config to run_jev_eval_faiss_apples.sh (plain "faiss" kNN
# candidate retrieval, matching exp/comparisons/.../config.json for the
# 71.5% run), but strips the generative-LLM CoT / output-format
# instructions from every prompt before sending it to Jev (see
# src/evaluation/jev_eval.py: tailor_prompt_for_jev). Output is tagged
# "jev-1.13-tailored" so it never collides with the default-prompt run's
# results file; the actual BeatAPI request still uses "jev-1.13".
#
# Usage:
#   chmod +x run_jev_eval_faiss_apples_tailored.sh && ./run_jev_eval_faiss_apples_tailored.sh
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

GREEN='\033[0;32m'
NC='\033[0m'

log_step() {
    echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  $1${NC}"
    echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}"
}

# ═══════════════════════════════════════════════════════════════════════════
# MANUAL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_ROOT="/mnt/d/Rowan/DiSC-AMC"
SRC_EVAL="${PROJECT_ROOT}/src/evaluation"
EXP_DIR="${PROJECT_ROOT}/exp"

DATA_ROOT="/mnt/d/Rowan/discrete-llm-amc/data/own"

DATASET_FOLDER="unlabeled_10k"
DATASET_TYPE="own"
BACKBONE="dino"
PREDICTION_SOURCE="faiss"
TOP_K=5
N_BINS=5
NOISE_MODE="noisySignal"
FEATURE_TYPE="embeddings"
N_COMPONENTS=10
OOD_TRAIN_FOLDER=""
USE_RAG="false"
RAG_K=0
PROMPT_TYPE="discret_prompts"
PROMPT_VERSION="v1"
NUM_TRIES=1

# ─── Jev / BeatAPI ───────────────────────────────────────────────────────
JEV_API_MODEL="jev-1.13"        # actual model string sent to BeatAPI
JEV_MODEL="jev-1.13-tailored"   # output-file tag only (keeps this run separate)
MAX_WORKERS=4
SAVE_EVERY=25

# ═══════════════════════════════════════════════════════════════════════════
# AUTOMATIC / DERIVED VARIABLES (do not edit below this line)
# ═══════════════════════════════════════════════════════════════════════════

N_COMPONENTS_EVAL=0
if [[ "$FEATURE_TYPE" == "embeddings" ]]; then
    N_COMPONENTS_EVAL=$N_COMPONENTS
fi

USE_RAG_PY="False"
RAG_K_EVAL=0
if [[ "$USE_RAG" == "true" ]]; then
    USE_RAG_PY="True"
    RAG_K_EVAL=$RAG_K
fi

_build_exp_base_name() {
    local feat_tag=""
    if [[ "$PREDICTION_SOURCE" != "dnn" ]]; then
        feat_tag+="${PREDICTION_SOURCE}"
    fi
    if [[ -n "$OOD_TRAIN_FOLDER" ]]; then
        [[ -n "$feat_tag" ]] && feat_tag+="_"
        feat_tag+="ood"
    fi
    if [[ "$FEATURE_TYPE" == "embeddings" && "$N_COMPONENTS" -gt 0 ]]; then
        [[ -n "$feat_tag" ]] && feat_tag+="_"
        feat_tag+="emb${N_COMPONENTS}"
    fi
    if [[ "$USE_RAG" == "true" && "$RAG_K" -gt 0 ]]; then
        [[ -n "$feat_tag" ]] && feat_tag+="_"
        feat_tag+="rag${RAG_K}"
    fi
    _EXP_BASE_NAME="${DATASET_FOLDER}"
    [[ -n "$feat_tag" ]] && _EXP_BASE_NAME+="_${feat_tag}"
    _EXP_BASE_NAME+="_${PROMPT_TYPE}_jev_${JEV_MODEL}"
}

setup_experiment() {
    _build_exp_base_name
    local version=1
    while [[ -d "${EXP_DIR}/${_EXP_BASE_NAME}_v$(printf '%02d' $version)" ]]; do
        version=$((version + 1))
    done
    EXP_RUN_DIR="${EXP_DIR}/${_EXP_BASE_NAME}_v$(printf '%02d' $version)"
    mkdir -p "$EXP_RUN_DIR"
    echo -e "${GREEN}  NEW experiment: ${EXP_RUN_DIR}${NC}"
}

find_latest_experiment() {
    _build_exp_base_name
    local latest=""
    local version=1
    while [[ -d "${EXP_DIR}/${_EXP_BASE_NAME}_v$(printf '%02d' $version)" ]]; do
        latest="${EXP_DIR}/${_EXP_BASE_NAME}_v$(printf '%02d' $version)"
        version=$((version + 1))
    done
    if [[ -z "$latest" ]]; then
        echo "ERROR: No existing experiment folder found for ${_EXP_BASE_NAME}_v*" >&2
        exit 1
    fi
    EXP_RUN_DIR="$latest"
    echo -e "${GREEN}  Using existing experiment: ${EXP_RUN_DIR}${NC}"
}

step7_query_jev() {
    log_step "STEP 7 — Query Jev (${JEV_API_MODEL}, TAILORED prompt) — faiss apples-to-apples"
    cd "$SRC_EVAL"
    PYTHONPATH="${PROJECT_ROOT}" python3 -c "
from src.evaluation.jev_eval import main
main(
    dataset_folder='${DATASET_FOLDER}',
    prompt_type='${PROMPT_TYPE}',
    model_name='${JEV_MODEL}',
    noise_mode='${NOISE_MODE}',
    n_bins=${N_BINS},
    top_k=${TOP_K},
    num_tries=${NUM_TRIES},
    prediction_source='${PREDICTION_SOURCE}',
    feature_type='${FEATURE_TYPE}',
    n_components=${N_COMPONENTS_EVAL},
    ood_train_folder='${OOD_TRAIN_FOLDER}',
    use_rag=${USE_RAG_PY},
    rag_k=${RAG_K_EVAL},
    output_dir='${EXP_RUN_DIR}',
    prompt_version='${PROMPT_VERSION}',
    backbone='${BACKBONE}',
    dataset_type='${DATASET_TYPE}',
    max_workers=${MAX_WORKERS},
    save_every=${SAVE_EVERY},
    data_root='${DATA_ROOT}',
    tailor_prompt=True,
    use_criteria_descriptions=True,
    api_model_name='${JEV_API_MODEL}',
)
"
}

step8_metrics_jev() {
    log_step "STEP 8 — Compute accuracy (Jev, TAILORED prompt, faiss apples-to-apples)"
    cd "$SRC_EVAL"
    PYTHONPATH="${PROJECT_ROOT}" python3 -c "
from src.evaluation.jev_eval import read_results, get_class_names
from src.evaluation.utils import sort_results_by_prompt, get_unique_prompts, print_metrics

_CLASS_NAMES = get_class_names('${DATASET_TYPE}')

results = read_results(
    dataset_folder='${DATASET_FOLDER}',
    prompt_type='${PROMPT_TYPE}',
    model_name='${JEV_MODEL}',
    noise_mode='${NOISE_MODE}',
    n_bins=${N_BINS},
    top_k=${TOP_K},
    prediction_source='${PREDICTION_SOURCE}',
    feature_type='${FEATURE_TYPE}',
    n_components=${N_COMPONENTS_EVAL},
    ood_train_folder='${OOD_TRAIN_FOLDER}',
    use_rag=${USE_RAG_PY},
    rag_k=${RAG_K_EVAL},
    output_dir='${EXP_RUN_DIR}',
    prompt_version='${PROMPT_VERSION}',
    backbone='${BACKBONE}',
)
sorted_results = sort_results_by_prompt(results)
print(f'Unique prompts: {len(get_unique_prompts(results))}')
print_metrics(sorted_results, _CLASS_NAMES)
"
}

# ═══════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════

setup_experiment
# find_latest_experiment

echo "DATA_ROOT=$DATA_ROOT"
echo "DATASET_FOLDER=$DATASET_FOLDER"
echo "PREDICTION_SOURCE=$PREDICTION_SOURCE"
echo "FEATURE_TYPE=$FEATURE_TYPE (n_components=$N_COMPONENTS)"
echo "JEV_MODEL(tag)=$JEV_MODEL  JEV_API_MODEL=$JEV_API_MODEL"
echo "EXP_RUN_DIR=$EXP_RUN_DIR"

step7_query_jev
step8_metrics_jev

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Done. Results → ${EXP_RUN_DIR}${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
