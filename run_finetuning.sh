#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# DiSC-AMC — QLoRA Finetuning Pipeline
# ═══════════════════════════════════════════════════════════════════════════
#
# End-to-end pipeline for finetuning an LLM on signal classification data
# and evaluating the finetuned model. Builds on the same data pipeline as
# run_pipeline.sh, then adds QLoRA finetuning + finetuned-model evaluation.
#
# Usage:
#   1. Adjust the variables below to match your setup.
#   2. Comment out any STEP you don't want to run.
#   3. Run:  chmod +x run_finetuning.sh && ./run_finetuning.sh
#
# Each step is guarded by a function so you can comment / uncomment
# individual calls at the bottom of the file.
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ─── Colors ─────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_step() {
    echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  $1${NC}"
    echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}"
}

log_warn() {
    echo -e "${YELLOW}  ⚠  $1${NC}"
}

# ═══════════════════════════════════════════════════════════════════════════
# MANUAL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# ─── Project paths ───────────────────────────────────────────────────────
PROJECT_ROOT="/mnt/d/Rowan/DiSC-AMC/"
DATA_ROOT="/mnt/d/Rowan/discrete-llm-amc/data/own"
EXP_DIR="${PROJECT_ROOT}/exp"
MODEL_DIR="/mnt/d/Rowan/discrete-llm-amc/models"

# ─── Dataset ─────────────────────────────────────────────────────────────
DATASET_FOLDER="unlabeled_10k"                    # folder name under DATA_ROOT
TRAIN_DATASET_FOLDER=""      # OOD: load train .pkl from this folder
                                          # set to "" to use DATASET_FOLDER for both
DATASET_TYPE="own"           # "own" (flat dir, class in filename) or
                             # "radioml" (test/{Class}/*.npy, SNR in parent dir)

# ─── Model / backbone (for data generation steps) ───────────────────────
BACKBONE="dino"             # dino | resnet
IMAGE_SIZE=96
BATCH_SIZE=32
NUM_WORKERS=4

# ─── Data generation (from run_pipeline) ────────────────────────────────
CLASSIFIER_PATH="${EXP_DIR}/dino_classifier.pth"
PREDICTION_SOURCE="faiss_filled" # dnn | centroid | rf | faiss
TOP_K=5
NOISE_MODE="noisySignal"    # noisySignal | noiselessSignal
N_BINS=5
KNN_K=50

# Centroid / FAISS paths (used if running data generation steps)
CENTROID_OUTPUT="${DATA_ROOT}/${TRAIN_DATASET_FOLDER:-$DATASET_FOLDER}/train/class_centers.json"
FAISS_INDEX_PATH="${DATA_ROOT}/${TRAIN_DATASET_FOLDER:-$DATASET_FOLDER}/train/faiss_knn_${BACKBONE}"

# ─── Feature type ───────────────────────────────────────────────────────
USE_RAG=true               # true → build/use FAISS index for example selection
RAG_K=10                    # number of nearest neighbours per test signal
MIN_RAG_CLASSES=5           # min distinct classes among RAG *few-shot examples* (widens
                             # similarity search until N classes appear; 0 = no constraint)
                             # NOTE: this does NOT affect classification options in the prompt.
                             # For guaranteed TOP_K distinct options, regenerate predictions
                             # with PREDICTION_SOURCE=faiss_filled in run_pipeline.sh
FEATURE_TYPE="embeddings"        # stats | embeddings
PROMPT_VERSION="v2"         # v1 (original) | v2 (source-aware)
COMPLETION_VERSION=$PROMPT_VERSION       # v1 (generic reasoning) | v2 (feature-aware reasoning)
N_COMPONENTS=10             # PCA components (only for embeddings)
ENCODER_WEIGHTS="${EXP_DIR}/dino_classifier.pth"

# ═══════════════════════════════════════════════════════════════════════════
# FINETUNING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# ─── Base LLM to finetune ───────────────────────────────────────────────
UNSLOTH_MODEL="unsloth/DeepSeek-R1-Distill-Qwen-7B"
# Other options:
#   "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
#   "unsloth/gemma-3-27b-it-unsloth-bnb-4bit"
#   "unsloth/DeepSeek-R1-Distill-Qwen-32B-unsloth-bnb-4bit"

# ─── Prompt / data style ────────────────────────────────────────────────
PROMPT_STYLE="discret"       # discret | continuous
USE_THINKING=true            # include <think> reasoning in training targets

# ─── LoRA hyperparameters ───────────────────────────────────────────────
LORA_R=16                    # LoRA rank
LORA_ALPHA=16                # LoRA alpha (usually same as rank)

# ─── Training hyperparameters ───────────────────────────────────────────
EPOCHS=5
FT_BATCH_SIZE=2              # per-device batch size (VRAM-dependent)
GRAD_ACCUM=4                 # gradient accumulation steps
                             # effective batch = FT_BATCH_SIZE × GRAD_ACCUM
LR="5e-5"                   # learning rate
MAX_SEQ_LEN=1024             # max sequence length
WARMUP_STEPS=5               # warmup steps
SEED=3407                    # random seed
VAL_SPLIT=0.1                # fraction held out for validation (0.0 = none)

# ─── Output ─────────────────────────────────────────────────────────────
SAVE_MERGED=false            # also save merged 16-bit model (large!)
RUN_FINETUNE=false           # true → train new adapter (creates new version dir)
                             # false → evaluate only (reuses latest existing version)

# ─── Evaluation of finetuned model ──────────────────────────────────────
PROMPT_TYPE="discret_prompts"  # discret_prompts | old_discret_prompts | prompts | old_prompts
NUM_TRIES=1                    # number of inference attempts per prompt
INFERENCE_BATCH_SIZE=8         # prompts per model.generate() call (higher = faster, more VRAM)
MAX_NEW_TOKENS=512             # token budget per response (3000 is wasteful for classification)

# ─── OOD evaluation — test the finetuned model on other datasets ─────────
# Leave empty to skip OOD evaluation.  Each folder is used as the test set
# while TRAIN_DATASET_FOLDER (or DATASET_FOLDER) remains the training source.
OOD_TEST_FOLDERS=(             # e.g. ("-30dB" "-11_-15dB")
    "-30dB"
    "-11_-15dB"
)

# ═══════════════════════════════════════════════════════════════════════════
# AUTOMATIC / DERIVED VARIABLES (Do not edit below this line)
# ═══════════════════════════════════════════════════════════════════════════

DATASET_PATH="${DATA_ROOT}/${DATASET_FOLDER}"

# ─── OOD handling ────────────────────────────────────────────────────────
OOD_TRAIN_FOLDER=""
if [[ -n "$TRAIN_DATASET_FOLDER" && "$TRAIN_DATASET_FOLDER" != "$DATASET_FOLDER" ]]; then
    OOD_TRAIN_FOLDER="$TRAIN_DATASET_FOLDER"
    CENTROID_OUTPUT="${DATA_ROOT}/${TRAIN_DATASET_FOLDER}/train/class_centers.json"
    FAISS_INDEX_PATH="${DATA_ROOT}/${TRAIN_DATASET_FOLDER}/train/faiss_knn_${BACKBONE}"
fi

# ─── Derived prediction JSON filenames ───────────────────────────────────
_BB_TAG=""
[[ "$BACKBONE" != "dino" ]] && _BB_TAG="_${BACKBONE}"
RAW_JSON="top${TOP_K}_${PREDICTION_SOURCE}${_BB_TAG}_predictions.json"
CONVERTED_JSON="ntop${TOP_K}_${PREDICTION_SOURCE}${_BB_TAG}_predictions.json"

# ─── Embedding / RAG vars for evaluation ─────────────────────────────────
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

# ─── Build experiment tag for train .pkl path ────────────────────────────
# Mirrors src/naming.py ExperimentConfig.build_tag() EXCEPT for rag:
# Python's build_train() uses the legacy train_pkl_name() interface which
# does NOT accept use_rag, so the train pkl filename never contains a rag
# tag.  The rag tag only appears in test pkl and eval result names, which
# are handled by ExperimentConfig inside the Python evaluation code.
_build_pkl_tag() {
    local parts=()
    if [[ "$PREDICTION_SOURCE" != "dnn" ]]; then
        parts+=("$PREDICTION_SOURCE")
    fi
    if [[ -n "$OOD_TRAIN_FOLDER" ]]; then
        parts+=("ood")
    fi
    if [[ "$BACKBONE" != "dino" ]]; then
        parts+=("$BACKBONE")
    fi
    if [[ "$FEATURE_TYPE" == "embeddings" && "$N_COMPONENTS" -gt 0 ]]; then
        parts+=("emb${N_COMPONENTS}")
    fi
    # NOTE: rag is intentionally excluded — Python never puts rag in the
    # train pkl filename (build_train uses the legacy train_pkl_name call).
    local IFS="_"
    echo "${parts[*]}"
}

PKL_TAG="$(_build_pkl_tag)"
if [[ -n "$PKL_TAG" ]]; then
    TRAIN_PKL_NAME="train_${PKL_TAG}_${NOISE_MODE}_${N_BINS}_${TOP_K}_data.pkl"
else
    TRAIN_PKL_NAME="train_${NOISE_MODE}_${N_BINS}_${TOP_K}_data.pkl"
fi

# .pkl resides directly in the dataset folder (no train/ subdir)
TRAIN_PKL_DIR="${DATA_ROOT}/${TRAIN_DATASET_FOLDER:-$DATASET_FOLDER}"
TRAIN_PKL_PATH="${TRAIN_PKL_DIR}/${TRAIN_PKL_NAME}"

# ─── Build TEST pkl tag (must match naming used by generated_dataset.py) ──
# NOTE: generated_dataset.py calls test_pkl_name() with legacy positional args
# which only includes prediction_source + feature_tag (emb).  RAG and
# prompt_version are NOT encoded in the pkl filename — only in the
# *responses* JSON name created by the evaluation scripts.
_build_test_pkl_tag() {
    local parts=()
    if [[ "$PREDICTION_SOURCE" != "dnn" ]]; then
        parts+=("$PREDICTION_SOURCE")
    fi
    if [[ -n "$OOD_TRAIN_FOLDER" ]]; then
        parts+=("ood")
    fi
    if [[ "$BACKBONE" != "dino" ]]; then
        parts+=("$BACKBONE")
    fi
    if [[ "$FEATURE_TYPE" == "embeddings" && "$N_COMPONENTS" -gt 0 ]]; then
        parts+=("emb${N_COMPONENTS}")
    fi
    local IFS="_"
    echo "${parts[*]}"
}

TEST_PKL_TAG="$(_build_test_pkl_tag)"
if [[ -n "$TEST_PKL_TAG" ]]; then
    TEST_PKL_NAME="test_${TEST_PKL_TAG}_${NOISE_MODE}_${N_BINS}_${TOP_K}_data.pkl"
else
    TEST_PKL_NAME="test_${NOISE_MODE}_${N_BINS}_${TOP_K}_data.pkl"
fi
TEST_PKL_DIR="${DATA_ROOT}/${DATASET_FOLDER}"
TEST_PKL_PATH="${TEST_PKL_DIR}/${TEST_PKL_NAME}"

# ─── Shorten model name for folder naming ────────────────────────────────
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

MODEL_SHORT="$(_shorten_model "$UNSLOTH_MODEL")"
PROMPT_SHORT="$(_shorten_prompt_type "$PROMPT_TYPE")"

# ─── Build finetuning output directory ───────────────────────────────────
# Pattern: ft_{dataset}_{tag}_{prompt_style}_{model}_ep{N}_lr{LR}_r{R}_v{VV}
_build_ft_exp_name() {
    local name="ft_${DATASET_FOLDER}"
    if [[ -n "$PKL_TAG" ]]; then
        name+="_${PKL_TAG}"
    fi
    name+="_${PROMPT_STYLE}_${MODEL_SHORT}"
    name+="_ep${EPOCHS}_r${LORA_R}"
    echo "$name"
}

FT_BASE_NAME="$(_build_ft_exp_name)"

# Auto-increment version suffix
_FT_VERSION=1
while [[ -d "${EXP_DIR}/${FT_BASE_NAME}_v$(printf '%02d' $_FT_VERSION)" ]]; do
    _FT_VERSION=$((_FT_VERSION + 1))
done

# ═══════════════════════════════════════════════════════════════════════════
# Setup / find experiment helpers
# ═══════════════════════════════════════════════════════════════════════════

setup_ft_experiment() {
    local exp_name="${FT_BASE_NAME}_v$(printf '%02d' $_FT_VERSION)"
    FT_OUTPUT_DIR="${EXP_DIR}/${exp_name}"
    ADAPTER_DIR="${FT_OUTPUT_DIR}/lora_adapter"
    mkdir -p "$FT_OUTPUT_DIR"
    echo -e "${GREEN}  NEW finetuning experiment: ${FT_OUTPUT_DIR}${NC}"

    # Write config.json
    local git_hash=""
    git_hash=$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo "n/a")

    cat > "${FT_OUTPUT_DIR}/config.json" <<EOCFG
{
  "experiment_id": "${exp_name}",
  "experiment_type": "finetuning",
  "timestamp": "$(date -Iseconds)",
  "git_hash": "${git_hash}",
  "dataset": {
    "dataset_folder": "${DATASET_FOLDER}",
    "train_dataset_folder": "${TRAIN_DATASET_FOLDER}",
    "data_root": "${DATA_ROOT}",
    "noise_mode": "${NOISE_MODE}",
    "train_pkl_path": "${TRAIN_PKL_PATH}"
  },
  "model": {
    "base_model": "${UNSLOTH_MODEL}",
    "model_short": "${MODEL_SHORT}",
    "backbone": "${BACKBONE}",
    "prediction_source": "${PREDICTION_SOURCE}",
    "top_k": ${TOP_K}
  },
  "features": {
    "feature_type": "${FEATURE_TYPE}",
    "n_components": ${N_COMPONENTS},
    "n_bins": ${N_BINS},
    "prompt_style": "${PROMPT_STYLE}",
    "use_thinking": ${USE_THINKING}
  },
  "rag": {
    "use_rag": ${USE_RAG},
    "rag_k": ${RAG_K}
  },
  "lora": {
    "lora_r": ${LORA_R},
    "lora_alpha": ${LORA_ALPHA}
  },
  "training": {
    "epochs": ${EPOCHS},
    "batch_size": ${FT_BATCH_SIZE},
    "gradient_accumulation_steps": ${GRAD_ACCUM},
    "effective_batch_size": $((FT_BATCH_SIZE * GRAD_ACCUM)),
    "learning_rate": "${LR}",
    "max_seq_length": ${MAX_SEQ_LEN},
    "warmup_steps": ${WARMUP_STEPS},
    "seed": ${SEED},
    "val_split": ${VAL_SPLIT},
    "completion_version": "${COMPLETION_VERSION}",
    "save_merged": ${SAVE_MERGED}
  },
  "evaluation": {
    "prompt_type": "${PROMPT_TYPE}",
    "num_tries": ${NUM_TRIES}
  },
  "paths": {
    "project_root": "${PROJECT_ROOT}",
    "exp_dir": "${EXP_DIR}",
    "ft_output_dir": "${FT_OUTPUT_DIR}",
    "adapter_dir": "${FT_OUTPUT_DIR}/lora_adapter",
    "cache_dir": "${MODEL_DIR}"
  }
}
EOCFG
    echo "  config.json written."

    # Start logging
    FT_LOG="${FT_OUTPUT_DIR}/pipeline.log"
    exec > >(tee -a "$FT_LOG") 2>&1
    echo "═══ Finetuning pipeline log started at $(date -Iseconds) ═══"
}

find_latest_ft_experiment() {
    local latest=""
    local version=1
    while [[ -d "${EXP_DIR}/${FT_BASE_NAME}_v$(printf '%02d' $version)" ]]; do
        latest="${EXP_DIR}/${FT_BASE_NAME}_v$(printf '%02d' $version)"
        version=$((version + 1))
    done
    if [[ -z "$latest" ]]; then
        echo "ERROR: No existing finetuning experiment found for ${FT_BASE_NAME}_v*" >&2
        exit 1
    fi
    FT_OUTPUT_DIR="$latest"
    ADAPTER_DIR="${FT_OUTPUT_DIR}/lora_adapter"
    echo -e "${GREEN}  Using existing experiment: ${FT_OUTPUT_DIR}${NC}"

    FT_LOG="${FT_OUTPUT_DIR}/pipeline.log"
    exec > >(tee -a "$FT_LOG") 2>&1
    echo "═══ Finetuning pipeline log resumed at $(date -Iseconds) ═══"
}

# ═══════════════════════════════════════════════════════════════════════════
# Step functions
# ═══════════════════════════════════════════════════════════════════════════

# ─── STEP A: Generate train .pkl (reuse from run_pipeline) ──────────────
step_generate_train_pkl() {
    log_step "STEP A — Generate training .pkl dataset"

    if [[ -f "$TRAIN_PKL_PATH" ]]; then
        echo "  Train .pkl already exists: ${TRAIN_PKL_PATH}"
        echo "  Skipping generation. Delete it to regenerate."
        return 0
    fi

    cd "$PROJECT_ROOT"

    local emb_flags=""
    if [[ "$FEATURE_TYPE" == "embeddings" ]]; then
        emb_flags="--feature_type embeddings"
        emb_flags+=" --encoder_weights $ENCODER_WEIGHTS"
        emb_flags+=" --backbone $BACKBONE"
        emb_flags+=" --n_components $N_COMPONENTS"
        emb_flags+=" --batch_size $BATCH_SIZE"
    fi

    local rag_flags=""
    if [[ "$USE_RAG" == "true" ]]; then
        rag_flags="--use_rag --rag_k $RAG_K --min_classes $MIN_RAG_CLASSES"
    fi

    local target_folder="${TRAIN_DATASET_FOLDER:-$DATASET_FOLDER}"

    echo "  Building train .pkl for folder=${target_folder} …"
    python -m src.prompt.generated_dataset \
        --mode train \
        --dataset_folder="$target_folder" \
        --noise_mode "$NOISE_MODE" \
        --n_bins "$N_BINS" \
        --top_k "$TOP_K" \
        --prediction_source "$PREDICTION_SOURCE" \
        --data_root "$DATA_ROOT" \
        $emb_flags \
        $rag_flags \
        --prompt_version "$PROMPT_VERSION" \
        --dataset_type "$DATASET_TYPE"

    echo "  → Train .pkl generated: ${TRAIN_PKL_PATH}"
}

# ─── STEP B: Preview finetuning dataset ─────────────────────────────────
step_preview_dataset() {
    log_step "STEP B — Preview finetuning dataset"
    cd "$PROJECT_ROOT"

    local think_flag=""
    [[ "$USE_THINKING" != "true" ]] && think_flag="--no_thinking"

    python -m src.finetuning.dataset \
        --pkl_path "$TRAIN_PKL_PATH" \
        --prompt_style "$PROMPT_STYLE" \
        --num_preview 3 \
        $think_flag
}

# ─── STEP C: Run QLoRA finetuning ───────────────────────────────────────
step_finetune() {
    log_step "STEP C — QLoRA finetuning (${MODEL_SHORT}, LoRA r=${LORA_R}, ${EPOCHS} epochs)"
    cd "$PROJECT_ROOT"

    local think_flag=""
    [[ "$USE_THINKING" != "true" ]] && think_flag="--no_thinking"

    local merged_flag=""
    [[ "$SAVE_MERGED" == "true" ]] && merged_flag="--save_merged"

    python -m src.finetuning.train \
        --model_name "$UNSLOTH_MODEL" \
        --pkl_path "$TRAIN_PKL_PATH" \
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
    [[ "$SAVE_MERGED" == "true" ]] && echo "  Merged model  → ${FT_OUTPUT_DIR}/merged_model"
}

# ─── STEP D: Generate test .pkl for evaluation ──────────────────────────
step_generate_test_pkl() {
    log_step "STEP D — Generate test .pkl dataset"

    if [[ -f "$TEST_PKL_PATH" ]]; then
        echo "  Test .pkl already exists: ${TEST_PKL_PATH}"
        echo "  Skipping generation. Delete it to regenerate."
        return 0
    fi

    cd "$PROJECT_ROOT"

    local emb_flags=""
    if [[ "$FEATURE_TYPE" == "embeddings" ]]; then
        emb_flags="--feature_type embeddings"
        emb_flags+=" --encoder_weights $ENCODER_WEIGHTS"
        emb_flags+=" --backbone $BACKBONE"
        emb_flags+=" --n_components $N_COMPONENTS"
        emb_flags+=" --batch_size $BATCH_SIZE"
    fi

    local rag_flags=""
    if [[ "$USE_RAG" == "true" ]]; then
        rag_flags="--use_rag --rag_k $RAG_K --min_classes $MIN_RAG_CLASSES"
    fi

    local ood_flag=""
    if [[ -n "$OOD_TRAIN_FOLDER" ]]; then
        ood_flag="--train_dataset_folder=$OOD_TRAIN_FOLDER"
    fi

    echo "  Building test .pkl for folder=${DATASET_FOLDER} …"
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

    echo "  → Test .pkl generated."
}

# ─── STEP E: Evaluate finetuned model ───────────────────────────────────
step_evaluate_finetuned() {
    log_step "STEP E — Evaluate finetuned model"
    cd "$PROJECT_ROOT"

    if [[ ! -d "$ADAPTER_DIR" ]]; then
        echo "ERROR: Adapter not found at ${ADAPTER_DIR}" >&2
        echo "  Run step_finetune first, or point ADAPTER_DIR to your adapter." >&2
        return 1
    fi

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
    output_dir='${FT_OUTPUT_DIR}',
    adapter_path='${ADAPTER_DIR}',
    inference_batch_size=${INFERENCE_BATCH_SIZE},
    max_new_tokens=${MAX_NEW_TOKENS},
    prompt_version='${PROMPT_VERSION}',
    backbone='${BACKBONE}',
)
"
}

# ─── STEP E-OOD: Evaluate finetuned model on multiple OOD test sets ────
# For each folder in OOD_TEST_FOLDERS this will:
#   1. Generate the test pkl (skips if already present)
#   2. Run inference with the finetuned adapter
#   3. Print accuracy metrics
# The ood_train_folder argument tells ExperimentConfig which dataset was used
# for training so that output filenames follow the *_ood_* naming convention.
step_evaluate_ood_all() {
    if [[ ${#OOD_TEST_FOLDERS[@]} -eq 0 ]]; then
        echo "  OOD_TEST_FOLDERS is empty — skipping OOD evaluation."
        return 0
    fi

    # The folder that provided the training data (used as ood_train_folder in naming)
    local train_src="${TRAIN_DATASET_FOLDER:-$DATASET_FOLDER}"

    local emb_flags=""
    if [[ "$FEATURE_TYPE" == "embeddings" ]]; then
        emb_flags="--feature_type embeddings"
        emb_flags+=" --encoder_weights $ENCODER_WEIGHTS"
        emb_flags+=" --backbone $BACKBONE"
        emb_flags+=" --n_components $N_COMPONENTS"
        emb_flags+=" --batch_size $BATCH_SIZE"
    fi

    local rag_flags=""
    if [[ "$USE_RAG" == "true" ]]; then
        rag_flags="--use_rag --rag_k $RAG_K --min_classes $MIN_RAG_CLASSES"
    fi

    for test_folder in "${OOD_TEST_FOLDERS[@]}"; do
        log_step "OOD EVAL — test=${test_folder}  train=${train_src}"
        cd "$PROJECT_ROOT"

        # ── 1. Generate test pkl ─────────────────────────────────────
        # Build the expected filename (must match generated_dataset.py naming)
        local ood_parts=()
        [[ "$PREDICTION_SOURCE" != "dnn" ]] && ood_parts+=("$PREDICTION_SOURCE")
        [[ "$BACKBONE" != "dino" ]] && ood_parts+=("$BACKBONE")
        [[ "$FEATURE_TYPE" == "embeddings" && "$N_COMPONENTS" -gt 0 ]] && ood_parts+=("emb${N_COMPONENTS}")
        local ood_tag=""
        if [[ ${#ood_parts[@]} -gt 0 ]]; then
            local IFS="_"; ood_tag="${ood_parts[*]}"; unset IFS
        fi
        local ood_pkl_name
        if [[ -n "$ood_tag" ]]; then
            ood_pkl_name="test_${ood_tag}_${NOISE_MODE}_${N_BINS}_${TOP_K}_data.pkl"
        else
            ood_pkl_name="test_${NOISE_MODE}_${N_BINS}_${TOP_K}_data.pkl"
        fi
        local ood_pkl_path="${DATA_ROOT}/${test_folder}/${ood_pkl_name}"

        if [[ -f "$ood_pkl_path" ]]; then
            echo "  Test .pkl already exists: ${ood_pkl_path} — skipping."
        else
            echo "  Generating test pkl for ${test_folder} …"
            python -m src.prompt.generated_dataset \
                --mode test \
                --dataset_folder="$test_folder" \
                --noise_mode "$NOISE_MODE" \
                --n_bins "$N_BINS" \
                --top_k "$TOP_K" \
                --prediction_source "$PREDICTION_SOURCE" \
                --data_root "$DATA_ROOT" \
                --train_dataset_folder="$train_src" \
                $emb_flags \
                $rag_flags \
                --prompt_version "$PROMPT_VERSION" \
                --dataset_type "$DATASET_TYPE"
        fi

        # ── 2. Run inference ─────────────────────────────────────────
        echo "  Running inference …"
        python -c "
from src.evaluation.unsloth_eval import main
main(
    dataset_folder='${test_folder}',
    prompt_type='${PROMPT_TYPE}',
    model_name='${UNSLOTH_MODEL}',
    noise_mode='${NOISE_MODE}',
    n_bins=${N_BINS},
    top_k=${TOP_K},
    num_tries=${NUM_TRIES},
    prediction_source='${PREDICTION_SOURCE}',
    feature_type='${FEATURE_TYPE}',
    n_components=${N_COMPONENTS_EVAL},
    ood_train_folder='${train_src}',
    use_rag=${USE_RAG_PY},
    rag_k=${RAG_K_EVAL},
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

        # ── 3. Print metrics and save to CSV ─────────────────────────
        python -c "
import os
import csv
from src.evaluation.unsloth_eval import read_results, get_class_names
from src.evaluation.utils import sort_results_by_prompt, get_unique_prompts, print_metrics, pass_acc, majority_acc, acc, clean_acc

_CLASS_NAMES = get_class_names('${DATASET_TYPE}')

results = read_results(
    dataset_folder='${test_folder}',
    prompt_type='${PROMPT_TYPE}',
    model_name='${UNSLOTH_MODEL}',
    noise_mode='${NOISE_MODE}',
    n_bins=${N_BINS},
    top_k=${TOP_K},
    prediction_source='${PREDICTION_SOURCE}',
    feature_type='${FEATURE_TYPE}',
    n_components=${N_COMPONENTS_EVAL},
    ood_train_folder='${train_src}',
    use_rag=${USE_RAG_PY},
    rag_k=${RAG_K_EVAL},
    output_dir='${FT_OUTPUT_DIR}',
    prompt_version='${PROMPT_VERSION}',
    backbone='${BACKBONE}',
)
sorted_results = sort_results_by_prompt(results)
n_unique = len(get_unique_prompts(results))
print(f'[${test_folder}]  Unique prompts: {n_unique}')
print_metrics(sorted_results, _CLASS_NAMES)

# Append to CSV
csv_path = os.path.join('${EXP_DIR}', 'results_summary.csv')
file_exists = os.path.isfile(csv_path)

with open(csv_path, 'a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow([
            'Model', 'DATASET_FOLDER', 'TRAIN_DATASET_FOLDER', 'PREDICTION_SOURCE',
            'USE_RAG', 'FEATURE_TYPE', 'prompt_version', 'min_classes', 'backbone',
            'knn_k', '1-pass', '1-majority', 'acc', 'clean-acc', 'Number of unique prompts', 'exp_folder'
        ])
    
    writer.writerow([
        '${UNSLOTH_MODEL}',
        '${test_folder}',
        '${train_src}',
        '${PREDICTION_SOURCE}',
        '${USE_RAG_PY}'.upper(),
        '${FEATURE_TYPE}',
        '${PROMPT_VERSION}',
        '${MIN_RAG_CLASSES}',
        '${BACKBONE}',
        '${KNN_K}',
        str(pass_acc(sorted_results)),
        str(majority_acc(sorted_results)),
        str(acc(sorted_results)),
        str(clean_acc(sorted_results, class_names=_CLASS_NAMES)),
        str(n_unique),
        '${FT_OUTPUT_DIR}'
    ])
"
    done
}

# ─── STEP F: Compute metrics ────────────────────────────────────────────
step_metrics() {
    log_step "STEP F — Compute metrics for finetuned model"
    cd "$PROJECT_ROOT"
    python -c "
import os
import csv
from src.evaluation.unsloth_eval import read_results, get_class_names
from src.evaluation.utils import sort_results_by_prompt, get_unique_prompts, print_metrics, pass_acc, majority_acc, acc, clean_acc

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
    output_dir='${FT_OUTPUT_DIR}',
    prompt_version='${PROMPT_VERSION}',
    backbone='${BACKBONE}',
)
sorted_results = sort_results_by_prompt(results)
n_unique = len(get_unique_prompts(results))
print(f'Unique prompts: {n_unique}')
print_metrics(sorted_results, _CLASS_NAMES)

# Append to CSV
csv_path = os.path.join('${EXP_DIR}', 'results_summary.csv')
file_exists = os.path.isfile(csv_path)

with open(csv_path, 'a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow([
            'Model', 'DATASET_FOLDER', 'TRAIN_DATASET_FOLDER', 'PREDICTION_SOURCE',
            'USE_RAG', 'FEATURE_TYPE', 'prompt_version', 'min_classes', 'backbone',
            'knn_k', '1-pass', '1-majority', 'acc', 'clean-acc', 'Number of unique prompts', 'exp_folder'
        ])
    
    writer.writerow([
        '${UNSLOTH_MODEL}',
        '${DATASET_FOLDER}',
        '${OOD_TRAIN_FOLDER}',
        '${PREDICTION_SOURCE}',
        '${USE_RAG_PY}'.upper(),
        '${FEATURE_TYPE}',
        '${PROMPT_VERSION}',
        '${MIN_RAG_CLASSES}',
        '${BACKBONE}',
        '${KNN_K}',
        str(pass_acc(sorted_results)),
        str(majority_acc(sorted_results)),
        str(acc(sorted_results)),
        str(clean_acc(sorted_results, class_names=_CLASS_NAMES)),
        str(n_unique),
        '${FT_OUTPUT_DIR}'
    ])
"
}

# ═══════════════════════════════════════════════════════════════════════════
# RUN — Comment out any step you don't need
# ═══════════════════════════════════════════════════════════════════════════

# ─── Auto-select experiment directory based on RUN_FINETUNE ────────────
if [[ "$RUN_FINETUNE" == "true" ]]; then
    setup_ft_experiment        # creates new versioned directory
else
    find_latest_ft_experiment  # reuses latest existing directory
fi
# Or override manually:
# FT_OUTPUT_DIR="..."        # ← set to a specific folder path
# ADAPTER_DIR="${FT_OUTPUT_DIR}/lora_adapter"

echo ""
echo "──────────────────────────────────────────────────────────"
echo "  Configuration Summary"
echo "──────────────────────────────────────────────────────────"
echo "  MODEL:            ${UNSLOTH_MODEL}"
echo "  DATASET:          ${DATASET_FOLDER}"
echo "  TRAIN DATA:       ${TRAIN_DATASET_FOLDER:-$DATASET_FOLDER}"
echo "  TRAIN PKL:        ${TRAIN_PKL_PATH}"
echo "  PREDICTION_SRC:   ${PREDICTION_SOURCE}"
echo "  PROMPT_STYLE:     ${PROMPT_STYLE}"
echo "  USE_THINKING:     ${USE_THINKING}"
echo "  LoRA:             r=${LORA_R}, alpha=${LORA_ALPHA}"
echo "  TRAINING:         ${EPOCHS} epochs, bs=${FT_BATCH_SIZE}×${GRAD_ACCUM}, lr=${LR}"
echo "  MAX_SEQ_LEN:      ${MAX_SEQ_LEN}"
echo "  VAL_SPLIT:        ${VAL_SPLIT}"
echo "  COMPLETION_VER:    ${COMPLETION_VERSION}"
echo "  FT_OUTPUT_DIR:    ${FT_OUTPUT_DIR}"
echo "  ADAPTER_DIR:      ${ADAPTER_DIR}"
echo "──────────────────────────────────────────────────────────"
echo ""

# ─── Data preparation ───────────────────────────────────────────────────
step_generate_train_pkl       # Generate train .pkl if not exists
# step_preview_dataset        # Preview a few training samples (optional)

# ─── Finetuning ─────────────────────────────────────────────────────────
[[ "$RUN_FINETUNE" == "true" ]] && step_finetune

# ─── Evaluation (uses the same inference pipeline as run_pipeline) ──────
step_generate_test_pkl      # Uncomment if test .pkl not yet generated
step_evaluate_finetuned       # Run inference with finetuned model
step_metrics                  # Compute and print accuracy metrics

# ─── OOD evaluation — loop over OOD_TEST_FOLDERS ────────────────────────
step_evaluate_ood_all         # Evaluate on each folder in OOD_TEST_FOLDERS

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Finetuning pipeline complete.  Results → ${FT_OUTPUT_DIR}${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
