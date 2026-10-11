#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# DiSC-AMC — Jev (BeatAPI) Evaluation Runner
# ═══════════════════════════════════════════════════════════════════════════
#
# Runs ONLY the inference + accuracy steps against an already-built test
# .pkl — no dataset regeneration. Config below matches the paper's Table VI /
# Fig. 8-9 training-free setup (centroid candidate retrieval, RAG=false,
# discretized stats, 5 bins, ktop=5), so Jev's accuracy is directly
# comparable to the reported Gemini-2.5-Flash (45.5%) / Gemini-2.5-Pro
# (51.0%) rows on the same 200-query evaluation set.
#
# Usage:
#   chmod +x run_jev_eval.sh && ./run_jev_eval.sh
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

# ─── Dataset — must match an EXISTING test .pkl under data/own/ ─────────
# unlabeled_10k/test_centroid_noisySignal_5_5_data.pkl already exists
# (Table VI's in-distribution 200-query set); no generation step is run.
DATASET_FOLDER="unlabeled_10k"
DATASET_TYPE="own"            # "own" | "radioml"
BACKBONE="dino"
PREDICTION_SOURCE="centroid"  # must match the existing pkl's prediction_source tag
TOP_K=5
N_BINS=5
NOISE_MODE="noisySignal"
FEATURE_TYPE="stats"          # stats | embeddings — must match the existing pkl
N_COMPONENTS=0                # only used if FEATURE_TYPE=embeddings
OOD_TRAIN_FOLDER=""           # set only if DATASET_FOLDER is an OOD test set
USE_RAG="false"
RAG_K=0
PROMPT_TYPE="discret_prompts" # discret_prompts | old_discret_prompts | prompts | old_prompts
PROMPT_VERSION="v1"
NUM_TRIES=1                   # paper evaluates each query once (no self-consistency voting)

# ─── Jev / BeatAPI ───────────────────────────────────────────────────────
JEV_MODEL="jev-1.13"          # jev-1.13 (paid, normal rate limit) | jev-1.13-free (1-10 req/min)
MAX_WORKERS=10                # concurrent in-flight requests
SAVE_EVERY=25                 # checkpoint results to disk every N completions

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
#
# NOTE: these `cd` into SRC_EVAL (not PROJECT_ROOT). jev_eval.load_data()
# resolves its data path as '../../data/own/<folder>' relative to the
# *current working directory*, and load_dotenv() only searches upward from
# cwd — both only resolve correctly when cwd = src/evaluation. PYTHONPATH is
# set explicitly so `from src.evaluation...` still imports from repo root.

step7_query_jev() {
    log_step "STEP 7 — Query Jev (${JEV_MODEL}, via BeatAPI)"
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
)
"
}

step8_metrics_jev() {
    log_step "STEP 8 — Compute accuracy (Jev)"
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

# setup_experiment            # ← NEW run (creates exp/ folder for step 7)
find_latest_experiment        # ← switch to this to re-read an existing run's results only

echo "DATASET_FOLDER=$DATASET_FOLDER"
echo "PREDICTION_SOURCE=$PREDICTION_SOURCE"
echo "JEV_MODEL=$JEV_MODEL"
echo "EXP_RUN_DIR=$EXP_RUN_DIR"

# step7_query_jev
step8_metrics_jev

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Done. Results → ${EXP_RUN_DIR}${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
