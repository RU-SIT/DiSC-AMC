#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# DiSC-AMC — End-to-End Pipeline Runner
# ═══════════════════════════════════════════════════════════════════════════
#
# Usage:
#   1. Adjust the variables below to match your setup.
#   2. Comment out any STEP you don't want to run.
#   3. Run:  chmod +x run_pipeline.sh && ./run_pipeline.sh
#
# Each step is guarded by a function so you can comment / uncomment
# individual calls at the bottom of the file.
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ─── Colors ─────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
NC='\033[0m' # No Color

log_step() {
    echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  $1${NC}"
    echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}"
}

# ═══════════════════════════════════════════════════════════════════════════
# MANUAL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# ─── Project paths ───────────────────────────────────────────────────────
PROJECT_ROOT="/mnt/d/Rowan/DiSC-AMC/"
SRC_REPR="${PROJECT_ROOT}/src/representation_learning"
SRC_PROMPT="${PROJECT_ROOT}/src/prompt"
SRC_EVAL="${PROJECT_ROOT}/src/evaluation"
DATA_ROOT="/mnt/d/Rowan/discrete-llm-amc/data/own"
EXP_DIR="${PROJECT_ROOT}/exp"
MODEL_DIR="/mnt/d/Rowan/discrete-llm-amc/models"

# ─── Dataset ─────────────────────────────────────────────────────────────
DATASET_FOLDER="-11_-15dB"                # folder name under DATA_ROOT
TRAIN_DATASET_FOLDER="unlabeled_10k"      # OOD: load train .pkl from this folder
                                          # set to "" to use DATASET_FOLDER for both
DATASET_TYPE="own"           # "own" (flat dir, class in filename) or
                             # "radioml" (test/{Class}/*.npy, SNR in parent dir)

# ─── Model / backbone ───────────────────────────────────────────────────
BACKBONE="dino"             # dino | resnet
IMAGE_SIZE=96
BATCH_SIZE=32
NUM_WORKERS=4

# ─── Training (Step 2) ──────────────────────────────────────────────────
PRETRAINED_PATH="${EXP_DIR}/dino_autoencoder.pth"
CLASSIFIER_PATH="${EXP_DIR}/dino_classifier.pth"
NUM_EPOCHS=100
LEARNING_RATE="1e-4"
EVAL_STEP=5
FREEZE_ENCODER=false        # true → add --freeze_encoder flag

# ─── Centroids (Step 3) ─────────────────────────────────────────────────
FIND_CLOSEST=true           # true → also print closest sample per class

# ─── Prediction & discretisation (Steps 4-6) ────────────────────────────
PREDICTION_SOURCE="faiss_filled" # dnn | centroid | rf | faiss | faiss_filled (defined in src/naming.py)
TOP_K=5
NOISE_MODE="noisySignal"     # noisySignal | noiselessSignal
N_BINS=5
KNN_K=50                     # kNN neighbours for FAISS voting (only when faiss)

# ─── Feature type (Step 6) ──────────────────────────────────────────────
# RAG (Retrieval-Augmented Generation) — optional
USE_RAG=true                # true → build/use FAISS index for example selection
RAG_K=10                     # number of nearest neighbours per test signal
MIN_RAG_CLASSES=0            # min distinct classes among RAG *few-shot examples* (widens
                             # similarity search until N classes appear; 0 = no constraint)
                             # NOTE: this does NOT affect classification options in the prompt.
                             # For guaranteed TOP_K distinct options use PREDICTION_SOURCE=faiss_filled
FEATURE_TYPE="stats"         # stats | embeddings
PROMPT_VERSION="v1"          # v1 (original) | v2 (source-aware with shortlisting/feature context)
# If FEATURE_TYPE="embeddings", set these:
N_COMPONENTS=10              # PCA components to keep
ENCODER_WEIGHTS="${EXP_DIR}/dino_classifier.pth"

# ─── LLM evaluation (Step 7) ────────────────────────────────────────────
PROMPT_TYPE="discret_prompts"   # discret_prompts | old_discret_prompts | prompts | old_prompts
NUM_TRIES=1

# Provider-specific
GEMINI_MODEL="gemini-2.5-flash"
OPENAI_MODEL="o3-mini"
UNSLOTH_MODEL="unsloth/DeepSeek-R1-Distill-Qwen-7B" # Options: "unsloth/gpt-oss-20b-unsloth-bnb-4bit", "unsloth/gemma-3-27b-it-unsloth-bnb-4bit", "unsloth/DeepSeek-R1-Distill-Qwen-7B", "unsloth/DeepSeek-R1-Distill-Qwen-32B-unsloth-bnb-4bit"
INFERENCE_BATCH_SIZE=8         # prompts per model.generate() call (higher = faster, more VRAM)
MAX_NEW_TOKENS=512             # token budget per response (3000 is wasteful for classification)

# ═══════════════════════════════════════════════════════════════════════════
# AUTOMATIC / DERIVED VARIABLES (Do not edit below this line)
# ═══════════════════════════════════════════════════════════════════════════

# ─── Derived Variables ───────────────────────────────────────────────────
DATASET_PATH="${DATA_ROOT}/${DATASET_FOLDER}"

# Centroid output path
CENTROID_OUTPUT="${DATASET_PATH}/train/class_centers.json"
if [[ -n "$TRAIN_DATASET_FOLDER" && "$TRAIN_DATASET_FOLDER" != "$DATASET_FOLDER" ]]; then
    CENTROID_OUTPUT="${DATA_ROOT}/${TRAIN_DATASET_FOLDER}/train/class_centers.json"
    log_step "OOD mode: reusing train centroids from ${TRAIN_DATASET_FOLDER}"
fi

# FAISS index path (for PREDICTION_SOURCE=faiss)
FAISS_INDEX_PATH="${DATASET_PATH}/train/faiss_knn_${BACKBONE}"
if [[ -n "$TRAIN_DATASET_FOLDER" && "$TRAIN_DATASET_FOLDER" != "$DATASET_FOLDER" ]]; then
    FAISS_INDEX_PATH="${DATA_ROOT}/${TRAIN_DATASET_FOLDER}/train/faiss_knn_${BACKBONE}"
fi

# Derived filenames
RAW_JSON="top${TOP_K}_${PREDICTION_SOURCE}_${BACKBONE}_predictions.json"
CONVERTED_JSON="ntop${TOP_K}_${PREDICTION_SOURCE}_${BACKBONE}_predictions.json"

# Derived OOD / embedding vars for evaluation steps
OOD_TRAIN_FOLDER=""
if [[ -n "$TRAIN_DATASET_FOLDER" && "$TRAIN_DATASET_FOLDER" != "$DATASET_FOLDER" ]]; then
    OOD_TRAIN_FOLDER="$TRAIN_DATASET_FOLDER"
fi

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

# ═══════════════════════════════════════════════════════════════════════════
# Experiment management
# ═══════════════════════════════════════════════════════════════════════════

_shorten_prompt_type() {
    case "$1" in
        discret_prompts)      echo "disc" ;;
        old_discret_prompts)  echo "old_disc" ;;
        prompts)              echo "cont" ;;
        old_prompts)          echo "old_cont" ;;
        *)                    echo "$1" ;;
    esac
}

_shorten_model() {
    # Strip provider prefix and keep recognisable short name
    local m="$1"
    m="${m#unsloth/}"        # strip unsloth/
    m="${m%%-unsloth*}"      # strip -unsloth-bnb-4bit etc.
    echo "$m"
}

# Build the experiment folder base name (without version suffix).
# Sets _EXP_BASE_NAME for use by setup/find functions.
_build_exp_base_name() {
    local provider="${EXP_PROVIDER:-unknown}"
    local model_short
    case "$provider" in
        gemini)   model_short="$(_shorten_model "$GEMINI_MODEL")" ;;
        openai)   model_short="$(_shorten_model "$OPENAI_MODEL")" ;;
        unsloth)  model_short="$(_shorten_model "$UNSLOTH_MODEL")" ;;
        *)        model_short="unknown" ;;
    esac

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

    local prompt_short
    prompt_short="$(_shorten_prompt_type "$PROMPT_TYPE")"

    _EXP_BASE_NAME="${DATASET_FOLDER}"
    [[ -n "$feat_tag" ]] && _EXP_BASE_NAME+="_${feat_tag}"
    _EXP_BASE_NAME+="_${prompt_short}_${provider}_${model_short}"
}

# ─── Create a NEW experiment folder (for step 7 / writing) ──────────────
setup_experiment() {
    _build_exp_base_name

    # Auto-increment version suffix
    local version=1
    while [[ -d "${EXP_DIR}/${_EXP_BASE_NAME}_v$(printf '%02d' $version)" ]]; do
        version=$((version + 1))
    done
    local exp_name="${_EXP_BASE_NAME}_v$(printf '%02d' $version)"

    EXP_RUN_DIR="${EXP_DIR}/${exp_name}"
    mkdir -p "$EXP_RUN_DIR"
    echo -e "${GREEN}  NEW experiment: ${EXP_RUN_DIR}${NC}"

    # ── Write config.json ────────────────────────────────────────────────
    local provider="${EXP_PROVIDER:-unknown}"
    local git_hash=""
    git_hash=$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo "n/a")

    cat > "${EXP_RUN_DIR}/config.json" <<EOCFG
{
  "experiment_id": "${exp_name}",
  "timestamp": "$(date -Iseconds)",
  "git_hash": "${git_hash}",
  "dataset": {
    "dataset_folder": "${DATASET_FOLDER}",
    "train_dataset_folder": "${TRAIN_DATASET_FOLDER}",
    "data_root": "${DATA_ROOT}",
    "noise_mode": "${NOISE_MODE}"
  },
  "model": {
    "backbone": "${BACKBONE}",
    "prediction_source": "${PREDICTION_SOURCE}",
    "top_k": ${TOP_K},
    "knn_k": ${KNN_K}
  },
  "features": {
    "feature_type": "${FEATURE_TYPE}",
    "n_components": ${N_COMPONENTS},
    "n_bins": ${N_BINS}
  },
  "rag": {
    "use_rag": ${USE_RAG},
    "rag_k": ${RAG_K}
  },
  "evaluation": {
    "prompt_type": "${PROMPT_TYPE}",
    "num_tries": ${NUM_TRIES},
    "provider": "${provider}",
    "llm_model": "$(_shorten_model "$(case $provider in gemini) echo $GEMINI_MODEL;; openai) echo $OPENAI_MODEL;; unsloth) echo $UNSLOTH_MODEL;; esac)")",
    "llm_model_full": "$(case $provider in gemini) echo $GEMINI_MODEL;; openai) echo $OPENAI_MODEL;; unsloth) echo $UNSLOTH_MODEL;; esac)"
  },
  "paths": {
    "project_root": "${PROJECT_ROOT}",
    "exp_dir": "${EXP_DIR}",
    "encoder_weights": "${ENCODER_WEIGHTS}",
    "classifier_path": "${CLASSIFIER_PATH}"
  }
}
EOCFG
    echo "  config.json written."

    # ── Start logging (tee to both terminal and log file) ────────────────
    EXP_LOG="${EXP_RUN_DIR}/pipeline.log"
    exec > >(tee -a "$EXP_LOG") 2>&1
    echo "═══ Pipeline log started at $(date -Iseconds) ═══"
}

# ─── Find the LATEST existing experiment folder (for step 8 / reading) ──
find_latest_experiment() {
    _build_exp_base_name

    # Find highest version that exists
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

    # Append to existing log
    EXP_LOG="${EXP_RUN_DIR}/pipeline.log"
    exec > >(tee -a "$EXP_LOG") 2>&1
    echo "═══ Pipeline log resumed at $(date -Iseconds) ═══"
}

# ═══════════════════════════════════════════════════════════════════════════
# Step functions
# ═══════════════════════════════════════════════════════════════════════════

step2_train_classifier() {
    log_step "STEP 2 — Train the classifier"
    local freeze_flag=""
    [[ "$FREEZE_ENCODER" == "true" ]] && freeze_flag="--freeze_encoder"

    cd "$PROJECT_ROOT"
    python -m src.representation_learning.classifier_training \
        --model "$BACKBONE" \
        --base_data_path "${DATASET_PATH}/train" \
        --pretrained_path "$PRETRAINED_PATH" \
        --save_path "$CLASSIFIER_PATH" \
        --num_epochs "$NUM_EPOCHS" \
        --learning_rate "$LEARNING_RATE" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --eval_step "$EVAL_STEP" \
        --image_size "$IMAGE_SIZE" \
        $freeze_flag
}

step3_compute_centroids() {
    log_step "STEP 3 — Compute class centroids"
    local closest_flag=""
    [[ "$FIND_CLOSEST" == "true" ]] && closest_flag="--find_closest"

    cd "$PROJECT_ROOT"
    python -m src.representation_learning.compute_centroids \
        --backbone "$BACKBONE" \
        --weights "$CLASSIFIER_PATH" \
        --dataset_path "${DATASET_PATH}/train" \
        --output "$CENTROID_OUTPUT" \
        --image_size "$IMAGE_SIZE" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        $closest_flag
}

step3b_build_faiss_index() {
    log_step "STEP 3b — Build FAISS index for kNN prediction"
    cd "$PROJECT_ROOT"
    python -m src.representation_learning.inference build_faiss \
        --backbone "$BACKBONE" \
        --weights "$CLASSIFIER_PATH" \
        --train_path "${DATASET_PATH}/train" \
        --output "$FAISS_INDEX_PATH" \
        --image_size "$IMAGE_SIZE"
}

step4a_evaluate_test() {
    log_step "STEP 4a — Evaluate on test set (accuracy report)"
    cd "$PROJECT_ROOT"
    python -m src.representation_learning.inference evaluate \
        --backbone "$BACKBONE" \
        --weights "$CLASSIFIER_PATH" \
        --test_path "${DATASET_PATH}/test" \
        --centroid_path "$CENTROID_OUTPUT" \
        --topk "$TOP_K" \
        --batch_size "$BATCH_SIZE" \
        --image_size "$IMAGE_SIZE"
}

step4b_predict_topk() {
    log_step "STEP 4b — Top-k predictions (${PREDICTION_SOURCE})"
    cd "$PROJECT_ROOT"

    local extra_flags=""
    if [[ "$PREDICTION_SOURCE" == "centroid" ]]; then
        extra_flags="--centroid_path $CENTROID_OUTPUT"
    elif [[ "$PREDICTION_SOURCE" == "faiss" ]]; then
        extra_flags="--faiss_index_path $FAISS_INDEX_PATH --knn_k $KNN_K"
    elif [[ "$PREDICTION_SOURCE" == "faiss_filled" ]]; then
        extra_flags="--faiss_index_path $FAISS_INDEX_PATH --knn_k $KNN_K --fill_topk"
    fi

    python -m src.representation_learning.inference predict \
        --backbone "$BACKBONE" \
        --weights "$CLASSIFIER_PATH" \
        --dataset_path "${DATASET_PATH}/test" \
        --topk "$TOP_K" \
        --output "${DATASET_PATH}/${RAW_JSON}" \
        --image_size "$IMAGE_SIZE" \
        $extra_flags

    echo "  → Saved ${RAW_JSON}"
}

step5_convert_keys() {
    log_step "STEP 5 — Convert .png keys → .npy keys"
    cd "$PROJECT_ROOT"
    python -m src.representation_learning.convert_predictions \
        --input  "${DATASET_PATH}/${RAW_JSON}" \
        --output "${DATASET_PATH}/${CONVERTED_JSON}"

    echo "  → Saved ${CONVERTED_JSON}"
}

step6_generate_datasets() {
    log_step "STEP 6 — Generate LLM prompt datasets (.pkl)"
    cd "$PROJECT_ROOT"

    # Extra flags for embedding mode
    local emb_flags=""
    if [[ "$FEATURE_TYPE" == "embeddings" ]]; then
        emb_flags="--feature_type embeddings"
        emb_flags+=" --encoder_weights $ENCODER_WEIGHTS"
        emb_flags+=" --backbone $BACKBONE"
        emb_flags+=" --n_components $N_COMPONENTS"
        emb_flags+=" --batch_size $BATCH_SIZE"
    fi

      # RAG flags (optional)
    local rag_flags=""
    if [[ "$USE_RAG" == "true" ]]; then
        rag_flags="--use_rag --rag_k $RAG_K --min_classes $MIN_RAG_CLASSES"
    fi

    # OOD flag: when TRAIN_DATASET_FOLDER is set and differs from
    # DATASET_FOLDER, skip the train step (reuse existing train .pkl)
    # and pass --train_dataset_folder to the test step.
    local ood_flag=""
    local skip_train=false
    if [[ -n "$TRAIN_DATASET_FOLDER" && "$TRAIN_DATASET_FOLDER" != "$DATASET_FOLDER" ]]; then
        ood_flag="--train_dataset_folder=$TRAIN_DATASET_FOLDER"
        skip_train=true
        echo "${GREEN}  OOD mode: reusing train .pkl from ${TRAIN_DATASET_FOLDER}${NC}"
    fi

    if [[ "$skip_train" == "false" ]]; then
        echo "  6a. Building TRAIN data (feature_type=${FEATURE_TYPE}) …"
        python -m src.prompt.generated_dataset \
            --mode train \
            --dataset_folder="$DATASET_FOLDER" \
            --noise_mode "$NOISE_MODE" \
            --n_bins "$N_BINS" \
            --top_k "$TOP_K" \
            --prediction_source "$PREDICTION_SOURCE" \
            --data_root "$DATA_ROOT" \
            $emb_flags \
            $rag_flags \
            --prompt_version "$PROMPT_VERSION" \
            --dataset_type "$DATASET_TYPE"
    fi

    echo "  6b. Building TEST data (feature_type=${FEATURE_TYPE}) …"
    python -m src.prompt.generated_dataset \
        --mode test \
        --dataset_folder="$DATASET_FOLDER" \
        --noise_mode "$NOISE_MODE" \
        --n_bins "$N_BINS" \
        --top_k "$TOP_K" \
        --prediction_source "$PREDICTION_SOURCE" \
        --data_root "$DATA_ROOT" \
        $ood_flag \
        $emb_flags \
        $rag_flags \
        --prompt_version "$PROMPT_VERSION" \
        --dataset_type "$DATASET_TYPE"
}

step7_query_gemini() {
    log_step "STEP 7 — Query Gemini (${GEMINI_MODEL})"
    cd "$PROJECT_ROOT"
    python -c "
from src.evaluation.gemini_googleai import main
main(
    dataset_folder='${DATASET_FOLDER}',
    prompt_type='${PROMPT_TYPE}',
    model_name='${GEMINI_MODEL}',
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
)
"
}

step7_query_openai() {
    log_step "STEP 7 — Query OpenAI (${OPENAI_MODEL})"
    cd "$PROJECT_ROOT"
    python -c "
from src.evaluation.gpt_openai import main
main(
    dataset_folder='${DATASET_FOLDER}',
    prompt_type='${PROMPT_TYPE}',
    model_name='${OPENAI_MODEL}',
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
)
"
}

step7_query_unsloth() {
    log_step "STEP 7 — Query Unsloth (${UNSLOTH_MODEL})"
    cd "$PROJECT_ROOT"
    python -c "
from src.evaluation.unsloth_eval import main
main(
    dataset_folder='${DATASET_FOLDER}',
    prompt_type='${PROMPT_TYPE}',
    model_name='${UNSLOTH_MODEL}',
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
    cache_dir='${MODEL_DIR}',
    data_root='${DATA_ROOT}',
    output_dir='${EXP_RUN_DIR}',
    inference_batch_size=${INFERENCE_BATCH_SIZE},
    max_new_tokens=${MAX_NEW_TOKENS},
    prompt_version='${PROMPT_VERSION}',
    backbone='${BACKBONE}',
)
"
}

step8_metrics_gemini() {
    log_step "STEP 8 — Compute metrics (Gemini)"
    cd "$PROJECT_ROOT"
    python -c "
from src.evaluation.gemini_googleai import read_results, get_class_names
from src.evaluation.utils import sort_results_by_prompt, get_unique_prompts, print_metrics

_CLASS_NAMES = get_class_names('${DATASET_TYPE}')

results = read_results(
    dataset_folder='${DATASET_FOLDER}',
    prompt_type='${PROMPT_TYPE}',
    model_name='${GEMINI_MODEL}',
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

step8_metrics_openai() {
    log_step "STEP 8 — Compute metrics (OpenAI)"
    cd "$PROJECT_ROOT"
    python -c "
from src.evaluation.gpt_openai import read_results, get_class_names
from src.evaluation.utils import sort_results_by_prompt, get_unique_prompts, print_metrics

_CLASS_NAMES = get_class_names('${DATASET_TYPE}')

results = read_results(
    dataset_folder='${DATASET_FOLDER}',
    prompt_type='${PROMPT_TYPE}',
    model_name='${OPENAI_MODEL}',
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

step8_metrics_unsloth() {
    log_step "STEP 8 — Compute metrics (Unsloth)"
    cd "$PROJECT_ROOT"
    python -c "
from src.evaluation.unsloth_eval import read_results, get_class_names
from src.evaluation.utils import sort_results_by_prompt, get_unique_prompts, print_metrics

_CLASS_NAMES = get_class_names('${DATASET_TYPE}')

results = read_results(
    dataset_folder='${DATASET_FOLDER}',
    prompt_type='${PROMPT_TYPE}',
    model_name='${UNSLOTH_MODEL}',
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
    backbone='${BACKBONE}',
)
sorted_results = sort_results_by_prompt(results)
print(f'Unique prompts: {len(get_unique_prompts(results))}')
print_metrics(sorted_results, _CLASS_NAMES)
"
}


# ═══════════════════════════════════════════════════════════════════════════
# RUN — Comment out any step you don't need
# ═══════════════════════════════════════════════════════════════════════════

# ─── Set the provider BEFORE calling setup/find ─────────────────────────
# Uncomment exactly ONE of the following:
# EXP_PROVIDER="gemini"
# EXP_PROVIDER="openai"
EXP_PROVIDER="unsloth"

# ─── Choose ONE: ────────────────────────────────────────────────────────
setup_experiment              # ← Use for NEW runs (step 7): creates exp/ folder
# find_latest_experiment      # ← Use for READ-ONLY (step 8): reuses latest folder
# EXP_RUN_DIR="..."          # ← Or set manually to a specific folder path

echo "DATA_ROOT=$DATA_ROOT"
echo "BACKBONE=$BACKBONE"
echo "PREDICTION_SOURCE=$PREDICTION_SOURCE"
echo "TOP_K=$TOP_K"
echo "USE_RAG=$USE_RAG"
echo "RAG_K=$RAG_K"
echo "FEATURE_TYPE=$FEATURE_TYPE"
echo "N_COMPONENTS=$N_COMPONENTS"
echo "RAW_JSON=$RAW_JSON"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "DATASET_FOLDER=$DATASET_FOLDER"
echo "DATASET_PATH=$DATASET_PATH"
echo "CENTROID_OUTPUT=$CENTROID_OUTPUT"
echo "FAISS_INDEX_PATH=$FAISS_INDEX_PATH"
echo "KNN_K=$KNN_K"
echo "EXP_RUN_DIR=$EXP_RUN_DIR"

# step2_train_classifier
# step3_compute_centroids
# step3b_build_faiss_index   # only needed for PREDICTION_SOURCE=faiss
# step4a_evaluate_test
step4b_predict_topk
step5_convert_keys
# step6_generate_datasets

# Uncomment the provider(s) you want to run:
# step7_query_gemini
# step7_query_openai
# step7_query_unsloth

# Uncomment to compute metrics from saved results:
# step8_metrics_gemini
# step8_metrics_openai
# step8_metrics_unsloth

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Pipeline complete.  Results → ${EXP_RUN_DIR}${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
