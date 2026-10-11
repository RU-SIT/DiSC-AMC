#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# DiSC-AMC — Jev (BeatAPI) Evaluation Runner — apples-to-apples with the
# paper's best RadioML number (Table III: DINOv2, +20 dB, 76.25%)
# ═══════════════════════════════════════════════════════════════════════════
#
# Table III evaluates DiSC-AMC on RadioML.2018.01a (24 classes) with a
# FINE-TUNED DINOv2 encoder backbone feeding a training-free LLM
# (DeepSeek-R1-Distill-Qwen-7B, bins=5, kNN=10, ktop=5). The best cell in
# that table is DINOv2 @ +20 dB = 76.25%.
#
# Verified locally: exp/radioml_native_eval/radioml_native_eval_results.csv
# and exp/dino_overfit_radioml_native_eval/radioml_native_eval_results.csv
# both report faiss_filled_emb10 @ snr_20db = 0.7625 (183/240) — matching
# Table III exactly. The underlying result file's candidate-option count is
# ALWAYS exactly 5 (a genuine 24-choose-5-shortlist task every query), i.e.
# this used PREDICTION_SOURCE="faiss_filled_rml" (the RadioML-native-encoder,
# brute-force-padded source), not plain "faiss_rml" (which lets the
# candidate set collapse below 5, inflating accuracy the same way it did
# for the SynthMod-10 71.5% figure — see run_jev_eval_faiss_apples.sh).
#
# The required pkl lives in the external RadioML data root (not this repo's
# local data/own):
#   /mnt/d/Rowan/discrete-llm-amc/data/RadioML/snr_20db/
#       test_faiss_filled_rml_emb10_snr_20db_5_5_data.pkl
#
# Note prompt_version="v2" — the matching response file is tagged "pv2"
# (source-aware prompt format), unlike the SynthMod-10 runs which use v1.
#
# Usage:
#   chmod +x run_jev_eval_radioml_best.sh && ./run_jev_eval_radioml_best.sh
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

# External data root — RadioML pkl variants live here, not in this repo.
DATA_ROOT="/mnt/d/Rowan/discrete-llm-amc/data/RadioML"

# ─── Dataset — must match an EXISTING test .pkl under DATA_ROOT ─────────
DATASET_FOLDER="snr_20db"       # best SNR slice in Table III (76.25%)
DATASET_TYPE="radioml"          # 24-class RadioML.2018.01a label set
BACKBONE="dino"
PREDICTION_SOURCE="faiss_filled_rml"  # FAISS kNN (padded) — RadioML-native DINO encoder
TOP_K=5
N_BINS=5
NOISE_MODE="snr_20db"           # reused as the pkl's noise-mode tag, matches DATASET_FOLDER
FEATURE_TYPE="embeddings"       # PCA-compressed encoder embeddings
N_COMPONENTS=10
OOD_TRAIN_FOLDER=""             # not applicable — RadioML has its own train split
USE_RAG="false"
RAG_K=0
PROMPT_TYPE="discret_prompts"
PROMPT_VERSION="v2"             # source-aware prompt format ("pv2" tag), matches the 76.25% run
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
    log_step "STEP 7 — Query Jev (${JEV_MODEL}, via BeatAPI) — RadioML snr_20db, faiss_filled_rml"
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
    log_step "STEP 8 — Compute accuracy (Jev, RadioML snr_20db, faiss_filled_rml)"
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
