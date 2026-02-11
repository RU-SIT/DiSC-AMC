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

# ─── Project paths ───────────────────────────────────────────────────────
PROJECT_ROOT="/mnt/d/Rowan/DiSC-AMC/"
SRC_REPR="${PROJECT_ROOT}/src/representation learning"
SRC_PROMPT="${PROJECT_ROOT}/src/prompt"
SRC_EVAL="${PROJECT_ROOT}/src/evaluation"
DATA_ROOT="${PROJECT_ROOT}/data/own"
EXP_DIR="${PROJECT_ROOT}/exp"

# ─── Dataset ─────────────────────────────────────────────────────────────
DATASET_FOLDER="-11_-15dB"                # folder name under DATA_ROOT
TRAIN_DATASET_FOLDER="unlabeled_10k"      # OOD: load train .pkl from this folder
                                          # set to "" to use DATASET_FOLDER for both
DATASET_PATH="${DATA_ROOT}/${DATASET_FOLDER}"

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
CENTROID_OUTPUT="${DATASET_PATH}/train/class_centers.json"
FIND_CLOSEST=true           # true → also print closest sample per class
if [[ -n "$TRAIN_DATASET_FOLDER" && "$TRAIN_DATASET_FOLDER" != "$DATASET_FOLDER" ]]; then
    CENTROID_OUTPUT="${DATA_ROOT}/${TRAIN_DATASET_FOLDER}/train/class_centers.json"
    echo "  OOD mode: reusing train centroids from ${TRAIN_DATASET_FOLDER}"
fi

# ─── Prediction & discretisation (Steps 4-6) ────────────────────────────
PREDICTION_SOURCE="centroid" # dnn | centroid | rf  (defined in src/naming.py)
TOP_K=5
NOISE_MODE="noisySignal"     # noisySignal | noiselessSignal
N_BINS=5

# ─── Feature type (Step 6) ──────────────────────────────────────────────
# RAG (Retrieval-Augmented Generation) — optional
USE_RAG=false                # true → build/use FAISS index for example selection
RAG_K=10                     # number of nearest neighbours per test signal
FEATURE_TYPE="stats"         # stats | embeddings
# If FEATURE_TYPE="embeddings", set these:
N_COMPONENTS=10              # PCA components to keep
ENCODER_WEIGHTS="${EXP_DIR}/dino_classifier.pth"

# Derived filenames — follow the convention in src/naming.py
RAW_JSON="top${TOP_K}_${PREDICTION_SOURCE}_predictions.json"
CONVERTED_JSON="ntop${TOP_K}_${PREDICTION_SOURCE}_predictions.json"

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

# ─── LLM evaluation (Step 7) ────────────────────────────────────────────
PROMPT_TYPE="discret_prompts"   # discret_prompts | old_discret_prompts | prompts | old_prompts
NUM_TRIES=1

# Provider-specific
GEMINI_MODEL="gemini-2.5-flash"
OPENAI_MODEL="o3-mini"
UNSLOTH_MODEL="unsloth/DeepSeek-R1-Distill-Qwen-7B" # Options: "unsloth/gpt-oss-20b-unsloth-bnb-4bit", "unsloth/gemma-3-27b-it-unsloth-bnb-4bit", "unsloth/DeepSeek-R1-Distill-Qwen-7B", "unsloth/DeepSeek-R1-Distill-Qwen-32B-unsloth-bnb-4bit"

# ═══════════════════════════════════════════════════════════════════════════
# Step functions
# ═══════════════════════════════════════════════════════════════════════════

step2_train_classifier() {
    echo "══════════════════════════════════════════════════════════════"
    echo "  STEP 2 — Train the classifier"
    echo "══════════════════════════════════════════════════════════════"
    local freeze_flag=""
    [[ "$FREEZE_ENCODER" == "true" ]] && freeze_flag="--freeze_encoder"

    cd "$SRC_REPR"
    python classifier_training.py \
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
    echo "══════════════════════════════════════════════════════════════"
    echo "  STEP 3 — Compute class centroids"
    echo "══════════════════════════════════════════════════════════════"
    local closest_flag=""
    [[ "$FIND_CLOSEST" == "true" ]] && closest_flag="--find_closest"

    cd "$SRC_REPR"
    python compute_centroids.py \
        --backbone "$BACKBONE" \
        --weights "$CLASSIFIER_PATH" \
        --dataset_path "${DATASET_PATH}/train" \
        --output "$CENTROID_OUTPUT" \
        --image_size "$IMAGE_SIZE" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        $closest_flag
}

step4a_evaluate_test() {
    echo "══════════════════════════════════════════════════════════════"
    echo "  STEP 4a — Evaluate on test set (accuracy report)"
    echo "══════════════════════════════════════════════════════════════"
    cd "$SRC_REPR"
    python inference.py evaluate \
        --backbone "$BACKBONE" \
        --weights "$CLASSIFIER_PATH" \
        --test_path "${DATASET_PATH}/test" \
        --centroid_path "$CENTROID_OUTPUT" \
        --topk "$TOP_K" \
        --batch_size "$BATCH_SIZE" \
        --image_size "$IMAGE_SIZE"
}

step4b_predict_topk() {
    echo "══════════════════════════════════════════════════════════════"
    echo "  STEP 4b — Top-k predictions (${PREDICTION_SOURCE})"
    echo "══════════════════════════════════════════════════════════════"
    cd "$SRC_REPR"

    local centroid_flag=""
    if [[ "$PREDICTION_SOURCE" == "centroid" ]]; then
        centroid_flag="--centroid_path $CENTROID_OUTPUT"
    fi

    python inference.py predict \
        --backbone "$BACKBONE" \
        --weights "$CLASSIFIER_PATH" \
        --dataset_path "${DATASET_PATH}/test" \
        --topk "$TOP_K" \
        --output "${DATASET_PATH}/${RAW_JSON}" \
        --image_size "$IMAGE_SIZE" \
        $centroid_flag

    echo "  → Saved ${RAW_JSON}"
}

step5_convert_keys() {
    echo "══════════════════════════════════════════════════════════════"
    echo "  STEP 5 — Convert .png keys → .npy keys"
    echo "══════════════════════════════════════════════════════════════"
    cd "$SRC_REPR"
    python convert_predictions.py \
        --input  "${DATASET_PATH}/${RAW_JSON}" \
        --output "${DATASET_PATH}/${CONVERTED_JSON}"

    echo "  → Saved ${CONVERTED_JSON}"
}

step6_generate_datasets() {
    echo "══════════════════════════════════════════════════════════════"
    echo "  STEP 6 — Generate LLM prompt datasets (.pkl)"
    echo "══════════════════════════════════════════════════════════════"
    cd "$SRC_PROMPT"

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
        rag_flags="--use_rag --rag_k $RAG_K"
    fi

    # OOD flag: when TRAIN_DATASET_FOLDER is set and differs from
    # DATASET_FOLDER, skip the train step (reuse existing train .pkl)
    # and pass --train_dataset_folder to the test step.
    local ood_flag=""
    local skip_train=false
    if [[ -n "$TRAIN_DATASET_FOLDER" && "$TRAIN_DATASET_FOLDER" != "$DATASET_FOLDER" ]]; then
        ood_flag="--train_dataset_folder=$TRAIN_DATASET_FOLDER"
        skip_train=true
        echo "  OOD mode: reusing train .pkl from ${TRAIN_DATASET_FOLDER}"
    fi

    if [[ "$skip_train" == "false" ]]; then
        echo "  6a. Building TRAIN data (feature_type=${FEATURE_TYPE}) …"
        python generated_dataset.py \
            --mode train \
            --dataset_folder="$DATASET_FOLDER" \
            --noise_mode "$NOISE_MODE" \
            --n_bins "$N_BINS" \
            --top_k "$TOP_K" \
            --prediction_source "$PREDICTION_SOURCE" \
            --data_root "$DATA_ROOT" \
            $emb_flags \
            $rag_flags
    fi

    echo "  6b. Building TEST data (feature_type=${FEATURE_TYPE}) …"
    python generated_dataset.py \
        --mode test \
        --dataset_folder="$DATASET_FOLDER" \
        --noise_mode "$NOISE_MODE" \
        --n_bins "$N_BINS" \
        --top_k "$TOP_K" \
        --prediction_source "$PREDICTION_SOURCE" \
        --data_root "$DATA_ROOT" \
        $ood_flag \
        $emb_flags \
        $rag_flags
}

step7_query_gemini() {
    echo "══════════════════════════════════════════════════════════════"
    echo "  STEP 7 — Query Gemini (${GEMINI_MODEL})"
    echo "══════════════════════════════════════════════════════════════"
    cd "$SRC_EVAL"
    python -c "
from gemini_googleai import main
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
)
"
}

step7_query_openai() {
    echo "══════════════════════════════════════════════════════════════"
    echo "  STEP 7 — Query OpenAI (${OPENAI_MODEL})"
    echo "══════════════════════════════════════════════════════════════"
    cd "$SRC_EVAL"
    python -c "
from gpt_openai import main
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
)
"
}

step7_query_unsloth() {
    echo "══════════════════════════════════════════════════════════════"
    echo "  STEP 7 — Query Unsloth (${UNSLOTH_MODEL})"
    echo "══════════════════════════════════════════════════════════════"
    cd "$SRC_EVAL"
    python -c "
from unsloth_eval import main
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
)
"
}

step8_metrics_gemini() {
    echo "══════════════════════════════════════════════════════════════"
    echo "  STEP 8 — Compute metrics (Gemini)"
    echo "══════════════════════════════════════════════════════════════"
    cd "$SRC_EVAL"
    python -c "
from gemini_googleai import read_results, CLASS_NAMES
from utils import sort_results_by_prompt, get_unique_prompts, print_metrics

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
)
sorted_results = sort_results_by_prompt(results)
print(f'Unique prompts: {len(get_unique_prompts(results))}')
print_metrics(sorted_results, CLASS_NAMES)
"
}

step8_metrics_openai() {
    echo "══════════════════════════════════════════════════════════════"
    echo "  STEP 8 — Compute metrics (OpenAI)"
    echo "══════════════════════════════════════════════════════════════"
    cd "$SRC_EVAL"
    python -c "
from gpt_openai import read_results, CLASS_NAMES
from utils import sort_results_by_prompt, get_unique_prompts, print_metrics

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
)
sorted_results = sort_results_by_prompt(results)
print(f'Unique prompts: {len(get_unique_prompts(results))}')
print_metrics(sorted_results, CLASS_NAMES)
"
}

step8_metrics_unsloth() {
    echo "══════════════════════════════════════════════════════════════"
    echo "  STEP 8 — Compute metrics (Unsloth)"
    echo "══════════════════════════════════════════════════════════════"
    cd "$SRC_EVAL"
    python -c "
from unsloth_eval import read_results, CLASS_NAMES
from utils import sort_results_by_prompt, get_unique_prompts, print_metrics

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
)
sorted_results = sort_results_by_prompt(results)
print(f'Unique prompts: {len(get_unique_prompts(results))}')
print_metrics(sorted_results, CLASS_NAMES)
"
}

# ═══════════════════════════════════════════════════════════════════════════
# RUN — Comment out any step you don't need
# ═══════════════════════════════════════════════════════════════════════════

# step2_train_classifier
# step3_compute_centroids
# step4a_evaluate_test
# step4b_predict_topk
# step5_convert_keys
# step6_generate_datasets

# Uncomment the provider(s) you want to run:
# step7_query_gemini
# step7_query_openai
step7_query_unsloth

# Uncomment to compute metrics from saved results:
# step8_metrics_gemini
# step8_metrics_openai
step8_metrics_unsloth

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Pipeline complete."
echo "═══════════════════════════════════════════════════════════════"
