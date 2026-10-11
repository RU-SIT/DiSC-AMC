#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# DiSC-AMC — Jev (BeatAPI) Evaluation Runner — apples-to-apples with the
# paper's 71.5% DeepSeek-R1-Distill-Qwen-7B number
# ═══════════════════════════════════════════════════════════════════════════
#
# Same as run_jev_eval_best.sh (FAISS + PCA embeddings, 5 bins, ktop=5) but
# uses PREDICTION_SOURCE="faiss" (plain kNN voting, NOT the "_filled"
# brute-force-padded variant). This matches exactly the config in
# exp/comparisons/unlabeled_10k_faiss_emb10_disc_unsloth_DeepSeek-R1-Distill-Qwen-7B_v01/config.json
# — the run that produced the paper's cited 71.5% in-distribution figure
# (results_clean.csv: unsloth/DeepSeek-R1-Distill-Qwen-7B, faiss, embeddings,
# acc=(143, 200, 0.715)).
#
# Why this matters: "faiss" lets the candidate set collapse to as few as 1
# class when kNN voting is unanimous (62/200 queries in that run had only
# 1 candidate, trivially "free" at 98.4% accuracy) — unlike "faiss_filled",
# which brute-force-pads every query to exactly 5 candidates (always a
# genuine 5-way choice). Comparing Jev against the *same* candidate-count
# distribution as the 71.5% run is the only fair head-to-head.
#
# The required pkl already exists in the external data root:
#   /mnt/d/Rowan/discrete-llm-amc/data/own/unlabeled_10k/test_faiss_emb10_noisySignal_5_5_data.pkl
#
# Usage:
#   chmod +x run_jev_eval_faiss_apples.sh && ./run_jev_eval_faiss_apples.sh
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

# ─── Project paths ───────────────────────────────────────────────────────
PROJECT_ROOT="/mnt/d/Rowan/DiSC-AMC"
SRC_EVAL="${PROJECT_ROOT}/src/evaluation"
EXP_DIR="${PROJECT_ROOT}/exp"

# External data root — the plain "faiss" + embeddings pkl variant lives
# here, not in this repo's local data/own (which only has centroid/faiss_filled).
DATA_ROOT="/mnt/d/Rowan/discrete-llm-amc/data/own"

# ─── Dataset — must match an EXISTING test .pkl under DATA_ROOT ─────────
DATASET_FOLDER="unlabeled_10k"
DATASET_TYPE="own"              # "own" | "radioml"
BACKBONE="dino"
PREDICTION_SOURCE="faiss"       # plain FAISS kNN voting (NOT padded to top-k)
TOP_K=5
N_BINS=5
NOISE_MODE="noisySignal"
FEATURE_TYPE="embeddings"       # PCA-compressed encoder embeddings, matches the 71.5% run
N_COMPONENTS=10
OOD_TRAIN_FOLDER=""             # in-distribution
USE_RAG="false"                 # matches config.json: rag.use_rag = false
RAG_K=0
PROMPT_TYPE="discret_prompts"
PROMPT_VERSION="v1"             # the 71.5% run's filename has no "pv2" tag → v1
NUM_TRIES=1

# ─── Jev / BeatAPI ───────────────────────────────────────────────────────
JEV_MODEL="jev-1.13"
MAX_WORKERS=4                   # lower than run_jev_eval.sh's 10 — avoids the
                                 # lockstep 429 thundering-herd seen at high concurrency
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

# ─── Create a NEW experiment folder (for step 7 / writing) ──────────────
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

# ─── Find the LATEST existing experiment folder (for step 8 / reading) ──
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

# ═══════════════════════════════════════════════════════════════════════════
# Step functions
# ═══════════════════════════════════════════════════════════════════════════

step7_query_jev() {
    log_step "STEP 7 — Query Jev (${JEV_MODEL}, via BeatAPI) — faiss (apples-to-apples w/ 71.5% DeepSeek run)"
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
)
"
}

step8_metrics_jev() {
    log_step "STEP 8 — Compute accuracy (Jev, faiss apples-to-apples config)"
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

setup_experiment              # ← NEW run (creates exp/ folder for step 7)
# find_latest_experiment      # ← switch to this to re-read an existing run's results only

echo "DATA_ROOT=$DATA_ROOT"
echo "DATASET_FOLDER=$DATASET_FOLDER"
echo "PREDICTION_SOURCE=$PREDICTION_SOURCE"
echo "FEATURE_TYPE=$FEATURE_TYPE (n_components=$N_COMPONENTS)"
echo "JEV_MODEL=$JEV_MODEL"
echo "EXP_RUN_DIR=$EXP_RUN_DIR"

step7_query_jev
step8_metrics_jev

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Done. Results → ${EXP_RUN_DIR}${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
