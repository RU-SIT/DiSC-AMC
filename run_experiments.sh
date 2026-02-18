#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# Batch Experiment Runner — loops over a grid of configurations
# ═══════════════════════════════════════════════════════════════════════════
# Usage:  chmod +x run_experiments.sh && ./run_experiments.sh [--resume]
#
# Runs all experiment combinations:
#   2 models × 2 datasets × 2 pred_sources × 2 RAG × 2 feature_types = 32
#
# Steps 3, 3b (centroids, FAISS) run once per TRAIN dataset.
# Steps 4b, 5, 6 run once per (dataset, pred_source, rag, feature_type).
# Steps 7, 8 run once per model within each data-prep group.
#
# Options:
#   --resume   Skip completed experiments and reuse incomplete exp folders
#              (folders that exist but lack *_responses.json).
#              Without --resume, incomplete folders are overwritten with a
#              new version number.
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ─── CLI argument parsing ────────────────────────────────────────────────
RESUME=false
for arg in "$@"; do
    case "$arg" in
        --resume) RESUME=true ;;
        -h|--help)
            echo "Usage: $0 [--resume]"
            echo "  --resume  Skip completed experiments & reuse incomplete exp folders"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            echo "Usage: $0 [--resume]" >&2
            exit 1
            ;;
    esac
done

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# ─── Experiment grid ─────────────────────────────────────────────────────
MODELS=(
    # "unsloth/DeepSeek-R1-Distill-Qwen-7B"
    "unsloth/GLM-4.6V-Flash"
)

# Each row: DATASET_FOLDER | TRAIN_DATASET_FOLDER (empty = same as DATASET_FOLDER)
DATASETS=(
    "unlabeled_10k|"
    "-11_-15dB|unlabeled_10k"
)

PREDICTION_SOURCES=("centroid" "faiss" ) #"faiss_filled"
RAG_OPTIONS=("true" "false")
FEATURE_TYPES=("stats" "embeddings")

# ─── Fixed config ────────────────────────────────────────────────────────
PROJECT_ROOT="/mnt/d/Rowan/DiSC-AMC/"
DATA_ROOT="/mnt/d/Rowan/discrete-llm-amc/data/own"
EXP_DIR="${PROJECT_ROOT}/exp"
MODEL_DIR="/mnt/d/Rowan/discrete-llm-amc/models"

BACKBONE="dino"
IMAGE_SIZE=96
BATCH_SIZE=32
NUM_WORKERS=4
TOP_K=5
NOISE_MODE="noisySignal"
N_BINS=5
KNN_K=50
RAG_K=10
MIN_CLASSES=0
N_COMPONENTS=10
PROMPT_VERSION="v1"          # v1 (original) | v2 (source-aware)
ENCODER_WEIGHTS="${EXP_DIR}/dino_classifier.pth"
PRETRAINED_PATH="${EXP_DIR}/dino_autoencoder.pth"
CLASSIFIER_PATH="${EXP_DIR}/dino_classifier.pth"
PROMPT_TYPE="discret_prompts"
NUM_TRIES=1

# ─── Helpers ─────────────────────────────────────────────────────────────

_shorten_model() {
    local m="$1"
    m="${m#unsloth/}"
    m="${m%%-unsloth*}"
    echo "$m"
}

_shorten_prompt_type() {
    case "$1" in
        discret_prompts)      echo "disc" ;;
        old_discret_prompts)  echo "old_disc" ;;
        prompts)              echo "cont" ;;
        old_prompts)          echo "old_cont" ;;
        *)                    echo "$1" ;;
    esac
}

_build_exp_dir_name() {
    # Args: model_name
    local model_short="$(_shorten_model "$1")"
    local feat_tag=""
    [[ "$PREDICTION_SOURCE" != "dnn" ]] && feat_tag+="${PREDICTION_SOURCE}"
    if [[ -n "$OOD_TRAIN_FOLDER" ]]; then
        [[ -n "$feat_tag" ]] && feat_tag+="_"; feat_tag+="ood"
    fi
    if [[ "$FEATURE_TYPE" == "embeddings" && "$N_COMPONENTS" -gt 0 ]]; then
        [[ -n "$feat_tag" ]] && feat_tag+="_"; feat_tag+="emb${N_COMPONENTS}"
    fi
    if [[ "$USE_RAG" == "true" && "$RAG_K" -gt 0 ]]; then
        [[ -n "$feat_tag" ]] && feat_tag+="_"; feat_tag+="rag${RAG_K}"
    fi

    local prompt_short="$(_shorten_prompt_type "$PROMPT_TYPE")"
    local base="${DATASET_FOLDER}"
    [[ -n "$feat_tag" ]] && base+="_${feat_tag}"
    base+="_${prompt_short}_unsloth_${model_short}"
    echo "$base"
}

_recompute_derived() {
    DATASET_PATH="${DATA_ROOT}/${DATASET_FOLDER}"

    # For OOD: centroids/FAISS come from the TRAIN dataset
    TRAIN_PATH="$DATASET_PATH"
    if [[ -n "$TRAIN_DATASET_FOLDER" && "$TRAIN_DATASET_FOLDER" != "$DATASET_FOLDER" ]]; then
        TRAIN_PATH="${DATA_ROOT}/${TRAIN_DATASET_FOLDER}"
    fi

    CENTROID_OUTPUT="${TRAIN_PATH}/train/class_centers.json"
    FAISS_INDEX_PATH="${TRAIN_PATH}/train/faiss_knn"

    RAW_JSON="top${TOP_K}_${PREDICTION_SOURCE}_predictions.json"
    CONVERTED_JSON="ntop${TOP_K}_${PREDICTION_SOURCE}_predictions.json"

    OOD_TRAIN_FOLDER=""
    if [[ -n "$TRAIN_DATASET_FOLDER" && "$TRAIN_DATASET_FOLDER" != "$DATASET_FOLDER" ]]; then
        OOD_TRAIN_FOLDER="$TRAIN_DATASET_FOLDER"
    fi

    N_COMPONENTS_EVAL=0
    [[ "$FEATURE_TYPE" == "embeddings" ]] && N_COMPONENTS_EVAL=$N_COMPONENTS

    USE_RAG_PY="False"; RAG_K_EVAL=0
    if [[ "$USE_RAG" == "true" ]]; then
        USE_RAG_PY="True"; RAG_K_EVAL=$RAG_K
    fi
}

# ─── Main ────────────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Batch Experiment Runner${NC}"
if $RESUME; then
    echo -e "${GREEN}  Mode: RESUME (reusing incomplete folders, skipping done)${NC}"
fi
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Count total
TOTAL=$(( ${#MODELS[@]} * ${#DATASETS[@]} * ${#PREDICTION_SOURCES[@]} * ${#RAG_OPTIONS[@]} * ${#FEATURE_TYPES[@]} ))
echo "  Total experiments: $TOTAL"

mkdir -p "$EXP_DIR"

# CSV results file
CSV_FILE="${EXP_DIR}/results_summary.csv"
if [[ ! -f "$CSV_FILE" ]]; then
    echo "Model,DATASET_FOLDER,TRAIN_DATASET_FOLDER,PREDICTION_SOURCE,USE_RAG,FEATURE_TYPE,1-pass,1-majority,acc,clean-acc,Number of unique prompts,exp_folder" \
        > "$CSV_FILE"
    echo "  Created $CSV_FILE"
else
    echo "  Appending to existing $CSV_FILE"
fi

COMPLETED=0
FAILED=0
SKIPPED=0
RUN_IDX=0

# ── Track which train datasets already have centroids/FAISS built ────────
declare -A CENTROIDS_BUILT=()
declare -A FAISS_BUILT=()

cd "$PROJECT_ROOT"

for ds_row in "${DATASETS[@]}"; do
    IFS='|' read -r DATASET_FOLDER TRAIN_DATASET_FOLDER <<< "$ds_row"

    # Determine which dataset centroids/FAISS are built on (the TRAIN dataset)
    EFFECTIVE_TRAIN="${TRAIN_DATASET_FOLDER:-$DATASET_FOLDER}"

    for PREDICTION_SOURCE in "${PREDICTION_SOURCES[@]}"; do
        for USE_RAG in "${RAG_OPTIONS[@]}"; do
            for FEATURE_TYPE in "${FEATURE_TYPES[@]}"; do

                _recompute_derived

                echo ""
                echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
                echo -e "${GREEN}  DATA PREP: dataset=${DATASET_FOLDER} train=${EFFECTIVE_TRAIN}${NC}"
                echo -e "${GREEN}             pred=${PREDICTION_SOURCE} rag=${USE_RAG} feat=${FEATURE_TYPE}${NC}"
                echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

                # ── Step 3: Centroids (once per train dataset) ────────
                if [[ -z "${CENTROIDS_BUILT[$EFFECTIVE_TRAIN]:-}" ]]; then
                    if [[ -f "$CENTROID_OUTPUT" ]]; then
                        echo -e "${YELLOW}  ⊘ Centroids exist: ${CENTROID_OUTPUT} (skipping)${NC}"
                    else
                        echo "  → Step 3: computing centroids for ${EFFECTIVE_TRAIN} …"
                        python -m src.representation_learning.compute_centroids \
                            --backbone "$BACKBONE" \
                            --weights "$CLASSIFIER_PATH" \
                            --dataset_path "${TRAIN_PATH}/train" \
                            --output "$CENTROID_OUTPUT" \
                            --image_size "$IMAGE_SIZE" \
                            --batch_size "$BATCH_SIZE" \
                            --num_workers "$NUM_WORKERS" \
                            --find_closest || { echo -e "${RED}  Step 3 FAILED${NC}"; }
                    fi
                    CENTROIDS_BUILT[$EFFECTIVE_TRAIN]=1
                fi

                # ── Step 3b: FAISS index (once per train dataset, only if needed) ──
                if [[ ("$PREDICTION_SOURCE" == "faiss" || "$PREDICTION_SOURCE" == "faiss_filled") && -z "${FAISS_BUILT[$EFFECTIVE_TRAIN]:-}" ]]; then
                    if [[ -d "$FAISS_INDEX_PATH" ]]; then
                        echo -e "${YELLOW}  ⊘ FAISS index exists: ${FAISS_INDEX_PATH} (skipping)${NC}"
                    else
                        echo "  → Step 3b: building FAISS index for ${EFFECTIVE_TRAIN} …"
                        python -m src.representation_learning.inference build_faiss \
                            --backbone "$BACKBONE" \
                            --weights "$CLASSIFIER_PATH" \
                            --train_path "${TRAIN_PATH}/train" \
                            --output "$FAISS_INDEX_PATH" \
                            --image_size "$IMAGE_SIZE" || { echo -e "${RED}  Step 3b FAILED${NC}"; }
                    fi
                    FAISS_BUILT[$EFFECTIVE_TRAIN]=1
                fi

                # ── Step 4b: Top-k predictions ───────────────────────
                RAW_JSON_PATH="${DATASET_PATH}/${RAW_JSON}"
                if [[ -f "$RAW_JSON_PATH" ]]; then
                    echo -e "${YELLOW}  ⊘ Raw predictions exist: ${RAW_JSON_PATH} (skipping step 4b)${NC}"
                else
                    echo "  → Step 4b: predict top-k (source=${PREDICTION_SOURCE}) …"
                    extra_flags=""
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
                        --output "$RAW_JSON_PATH" \
                        --image_size "$IMAGE_SIZE" \
                        $extra_flags || { echo -e "${RED}  Step 4b FAILED${NC}"; continue; }
                fi

                # ── Step 5: Convert keys ─────────────────────────────
                CONVERTED_JSON_PATH="${DATASET_PATH}/${CONVERTED_JSON}"
                if [[ -f "$CONVERTED_JSON_PATH" ]]; then
                    echo -e "${YELLOW}  ⊘ Converted predictions exist: ${CONVERTED_JSON_PATH} (skipping step 5)${NC}"
                else
                    echo "  → Step 5: convert prediction keys …"
                    python -m src.representation_learning.convert_predictions \
                        --input  "$RAW_JSON_PATH" \
                        --output "$CONVERTED_JSON_PATH" || { echo -e "${RED}  Step 5 FAILED${NC}"; continue; }
                fi

                # ── Step 6: Generate datasets ────────────────────────
                echo "  → Step 6: generate datasets (feature_type=${FEATURE_TYPE}) …"
                emb_flags=""
                if [[ "$FEATURE_TYPE" == "embeddings" ]]; then
                    emb_flags="--feature_type embeddings --encoder_weights $ENCODER_WEIGHTS --backbone $BACKBONE --n_components $N_COMPONENTS --batch_size $BATCH_SIZE"
                fi
                rag_flags=""
                [[ "$USE_RAG" == "true" ]] && rag_flags="--use_rag --rag_k $RAG_K --min_classes $MIN_CLASSES"

                ood_flag=""
                skip_train=false
                if [[ -n "$TRAIN_DATASET_FOLDER" && "$TRAIN_DATASET_FOLDER" != "$DATASET_FOLDER" ]]; then
                    ood_flag="--train_dataset_folder=$TRAIN_DATASET_FOLDER"
                    skip_train=true
                fi

                # Build pkl filenames to check for existence
                PKL_TAG="${PREDICTION_SOURCE}"
                [[ "$FEATURE_TYPE" == "embeddings" ]] && PKL_TAG+="_emb${N_COMPONENTS}"
                TRAIN_PKL_FILE="${TRAIN_PATH}/train_${PKL_TAG}_${NOISE_MODE}_${N_BINS}_${TOP_K}_data.pkl"
                TEST_PKL_FILE="${DATASET_PATH}/test_${PKL_TAG}_${NOISE_MODE}_${N_BINS}_${TOP_K}_data.pkl"

                if [[ "$skip_train" == "false" ]]; then
                    if [[ -f "$TRAIN_PKL_FILE" ]]; then
                        echo -e "${YELLOW}  ⊘ Train pkl exists: ${TRAIN_PKL_FILE} (skipping step 6a)${NC}"
                    else
                        echo "    6a. Building TRAIN data …"
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
                            --prompt_version "$PROMPT_VERSION" || { echo -e "${RED}  Step 6a FAILED${NC}"; continue; }
                    fi
                fi

                if [[ -f "$TEST_PKL_FILE" ]]; then
                    echo -e "${YELLOW}  ⊘ Test pkl exists: ${TEST_PKL_FILE} (skipping step 6b)${NC}"
                else
                    echo "    6b. Building TEST data …"
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
                        --prompt_version "$PROMPT_VERSION" || { echo -e "${RED}  Step 6 FAILED${NC}"; continue; }
                fi

                # ── Steps 7+8: per model ─────────────────────────────
                for UNSLOTH_MODEL in "${MODELS[@]}"; do
                    RUN_IDX=$((RUN_IDX + 1))

                    # ── Build exp folder name ────────────────────────
                    base_name="$(_build_exp_dir_name "$UNSLOTH_MODEL")"

                    # ── Skip / resume logic ──────────────────────────
                    latest_existing=""
                    v=1
                    while [[ -d "${EXP_DIR}/${base_name}_v$(printf '%02d' $v)" ]]; do
                        latest_existing="${EXP_DIR}/${base_name}_v$(printf '%02d' $v)"
                        v=$((v + 1))
                    done

                    if [[ -n "$latest_existing" ]] && ls "$latest_existing"/*_responses.json &>/dev/null; then
                        echo -e "${YELLOW}  [$RUN_IDX/$TOTAL] SKIP (results exist in ${latest_existing##*/})${NC}"
                        SKIPPED=$((SKIPPED + 1))
                        continue
                    fi

                    if $RESUME && [[ -n "$latest_existing" ]]; then
                        # Reuse the incomplete folder instead of creating a new version
                        EXP_RUN_DIR="$latest_existing"
                        echo -e "${YELLOW}  [$RUN_IDX/$TOTAL] RESUME (reusing incomplete ${latest_existing##*/})${NC}"
                    else
                        # Create new experiment folder
                        EXP_RUN_DIR="${EXP_DIR}/${base_name}_v$(printf '%02d' $v)"
                    fi
                    mkdir -p "$EXP_RUN_DIR"

                    git_hash=$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo "n/a")
                    model_short="$(_shorten_model "$UNSLOTH_MODEL")"
                    cat > "${EXP_RUN_DIR}/config.json" <<EOCFG
{
  "experiment_id": "${base_name}_v$(printf '%02d' $v)",
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
    "provider": "unsloth",
    "llm_model": "${model_short}",
    "llm_model_full": "${UNSLOTH_MODEL}"
  }
}
EOCFG

                    echo ""
                    echo -e "${GREEN}  [$RUN_IDX/$TOTAL] ${UNSLOTH_MODEL}${NC}"
                    echo "    exp → ${EXP_RUN_DIR}"

                    # ── Step 7: query LLM ────────────────────────────
                    echo "    → Step 7: querying ${UNSLOTH_MODEL} …"
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
)
" && {
                        # ── Step 8: metrics → terminal + CSV ─────────
                        echo "    → Step 8: computing metrics …"
                        CSV_ROW_TMP="${EXP_RUN_DIR}/.csv_row.tmp"
                        python -c "
import sys
from src.evaluation.unsloth_eval import read_results, CLASS_NAMES
from src.evaluation.utils import (
    sort_results_by_prompt, get_unique_prompts, print_metrics,
    acc, clean_acc, pass_acc, majority_acc,
)

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
)
sorted_results = sort_results_by_prompt(results)
n_unique = len(get_unique_prompts(results))

print(f'Unique prompts: {n_unique}')
print_metrics(sorted_results, CLASS_NAMES)

# Write CSV row to temp file (avoids stdout capture issues)
pass_val  = pass_acc(sorted_results)
maj_val   = majority_acc(sorted_results)
acc_val   = acc(sorted_results)
clean_val = clean_acc(sorted_results, class_names=CLASS_NAMES)

def fmt(t):
    return '\"' + str(t) + '\"'

with open('${CSV_ROW_TMP}', 'w') as f:
    f.write(','.join([
        '${UNSLOTH_MODEL}',
        '${DATASET_FOLDER}',
        '${TRAIN_DATASET_FOLDER}',
        '${PREDICTION_SOURCE}',
        '${USE_RAG}',
        '${FEATURE_TYPE}',
        fmt(pass_val),
        fmt(maj_val),
        fmt(acc_val),
        fmt(clean_val),
        str(n_unique),
        '${EXP_RUN_DIR}',
    ]) + '\n')
" && {
                            cat "$CSV_ROW_TMP" >> "$CSV_FILE"
                            rm -f "$CSV_ROW_TMP"
                            echo "    ✓ Row appended to results_summary.csv"
                        } || echo -e "${RED}    Step 8 FAILED${NC}"
                        COMPLETED=$((COMPLETED + 1))
                    } || {
                        echo -e "${RED}    Step 7 FAILED for ${UNSLOTH_MODEL}${NC}"
                        FAILED=$((FAILED + 1))
                    }

                done  # models
            done  # feature_types
        done  # rag_options
    done  # prediction_sources
done  # datasets

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Batch complete!${NC}"
echo -e "${GREEN}  Completed: ${COMPLETED}  Skipped: ${SKIPPED}  Failed: ${FAILED}  Total: ${TOTAL}${NC}"
echo -e "${GREEN}  CSV: ${CSV_FILE}${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
