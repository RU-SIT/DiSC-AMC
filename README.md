# DiSC-AMC

Implementation code for the paper [DiSC-AMC: Token- and Parameter-Efficient Discretized Statistics In-Context Automatic Modulation Classification](https://arxiv.org/abs/2510.00316) — a framework for Automatic Modulation Classification (AMC) using Discrete Large Language Models (LLMs) and Vision Transformers.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start — `run_pipeline.sh`](#quick-start--run_pipelinesh)
4. [RAG — Retrieval-Augmented Few-Shot Selection](#rag--retrieval-augmented-few-shot-selection)
5. [Project Structure](#project-structure)
6. [End-to-End Pipeline](#end-to-end-pipeline)
7. [Module Reference](#module-reference)
   - [Representation Learning (`src/representation learning/`)](#representation-learning)
   - [Prompt Engineering (`src/prompt/`)](#prompt-engineering)
   - [LLM Evaluation (`src/evaluation/`)](#llm-evaluation)
8. [Summary](#summary)
9. [Key Libraries](#key-libraries)
10. [License](#license)

---

## Overview

The pipeline follows these high-level stages:

```
Raw Signals (.npy)
    │
    ▼
┌──────────────────────────────────┐
│ 1. Representation Learning       │  src/representation learning/
│    • Spectrogram / Constellation │
│    • DINO / ResNet Autoencoder   │
│    • Classifier Training         │
│    • Centroid Computation        │
│    • Inference & Evaluation      │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ 2. Prompt Engineering            │  src/prompt/
│    • Feature Extraction (stats)  │
│    • Discretization (KBins)      │
│    • LLM Prompt Generation       │
│    • Dataset Preparation (.pkl)  │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ 3. LLM Evaluation               │  src/evaluation/
│    • Gemini API                  │
│    • OpenAI API                  │
│    • Local Unsloth Models        │
│    • Metrics & Reporting         │
└──────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Linux environment
- NVIDIA GPU with CUDA support (assumes CUDA 12.8)
- Python 3.10 – 3.12

### Setup

```bash
git clone <repository_url>
cd DiSC-AMC
pip install poetry
```

**Core only** (representation learning + prompt engineering):

```bash
poetry install
```

**With cloud API evaluation** (Gemini / OpenAI):

```bash
poetry install -E api
```

**With local LLM inference** (Unsloth + vLLM):

```bash
poetry install -E llm
```

**Everything**:

```bash
poetry install -E all
```

| Extra | What it adds |
|-------|-------------|
| `llm` | `unsloth`, `vllm`, `xformers`, `triton`, `transformers` |
| `api` | `google-generativeai`, `openai`, `python-dotenv` |
| `notebook` | `ipywidgets`, `ipykernel` |
| `tracking` | `wandb` |
| `all` | All of the above |

> **Note:** For custom builds of vLLM / Triton / xformers from source, see
> `install_dep.sh` or `unsloth_requirements.sh`.

---

## Quick Start — `run_pipeline.sh`

The entire pipeline (Steps 2–8) can be run from a single script.
All tuneable settings are declared as variables at the top of the file.

### 1. Configure

Open [run_pipeline.sh](run_pipeline.sh) and edit the variable blocks:

| Section | Key variables | Default |
|---------|--------------|---------|
| **Dataset** | `DATASET_FOLDER`, `TRAIN_DATASET_FOLDER` | `unlabeled_10k`, `""` |
| **Model / backbone** | `BACKBONE`, `IMAGE_SIZE`, `BATCH_SIZE` | `dino`, `96`, `32` |
| **Training** | `PRETRAINED_PATH`, `CLASSIFIER_PATH`, `NUM_EPOCHS`, `LEARNING_RATE` | see file |
| **Centroids** | `CENTROID_OUTPUT`, `FIND_CLOSEST` | `…/class_centers.json`, `true` |
| **Prediction** | `PREDICTION_SOURCE`, `TOP_K`, `NOISE_MODE`, `N_BINS` | `centroid`, `5`, `noisySignal`, `5` |
| **Feature type** | `FEATURE_TYPE`, `N_COMPONENTS`, `ENCODER_WEIGHTS` | `stats`, `10`, `…/dino_classifier.pth` |
| **LLM evaluation** | `PROMPT_TYPE`, `NUM_TRIES`, `GEMINI_MODEL`, … | `discret_prompts`, `3` |

Set `FEATURE_TYPE="embeddings"` to use encoder embeddings (PCA → discretize → letter-encode)
instead of statistical features. When doing so, also set `N_COMPONENTS` and `ENCODER_WEIGHTS`.

### 2. Choose which steps to run

At the bottom of the file, comment out any step you don't need:

```bash
step2_train_classifier
step3_compute_centroids
step4a_evaluate_test
step4b_predict_topk
step5_convert_keys
step6_generate_datasets

# Uncomment the provider(s) you want to run:
# step7_query_gemini
# step7_query_openai
# step7_query_unsloth

# Uncomment to compute metrics from saved results:
# step8_metrics_gemini
# step8_metrics_openai
# step8_metrics_unsloth
```

### 3. Run

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

The script uses `set -euo pipefail`, so it stops on the first error.

### In-Distribution (ID) vs Out-of-Distribution (OOD) Experiments

The pipeline supports two experimental modes controlled by `TRAIN_DATASET_FOLDER`:

#### In-Distribution (ID) — Default

Train and test on the **same** dataset. The pipeline fits a scaler and discretizers
on the training split, then applies them to the test split of the same dataset.

```bash
# run_pipeline.sh
DATASET_FOLDER="unlabeled_10k"
TRAIN_DATASET_FOLDER=""          # empty → uses DATASET_FOLDER for both
```

This runs the full flow: step 6a builds the train `.pkl` (fitting transformers),
then step 6b builds the test `.pkl` (reusing those transformers + loading top-k
predictions from the same folder).

#### Out-of-Distribution (OOD)

Test on a **different** dataset than the one used for training. The scaler,
discretizers, and few-shot examples come from the training dataset, while the
test signals come from a separate folder that may have different SNR ranges,
channel conditions, or other distribution shifts.

```bash
# run_pipeline.sh
DATASET_FOLDER="-11_-15dB"                # test signals (OOD target)
TRAIN_DATASET_FOLDER="unlabeled_10k"      # training data (source of scaler/examples)
```

When `TRAIN_DATASET_FOLDER` differs from `DATASET_FOLDER`, the pipeline:

1. **Skips step 6a** — the train `.pkl` must already exist in `TRAIN_DATASET_FOLDER`.
2. **Loads transformers** (scaler, discretizers, and optionally PCA) from the
   train `.pkl` in `data/own/<TRAIN_DATASET_FOLDER>/`.
3. **Loads few-shot examples** from the training dataset's signal files.
4. **Processes test signals** from `data/own/<DATASET_FOLDER>/test/`.
5. **Top-k predictions are optional** — if `ntop<k>_<source>_predictions.json`
   exists in the test dataset folder, engineered prompts with narrowed options
   are generated; otherwise, only old-style prompts (all 10 classes) are produced.
6. **Saves the output** `.pkl` in the test dataset folder.

You can also run this directly via the CLI without `run_pipeline.sh`:

```bash
cd src/prompt/

python generated_dataset.py \
    --mode test \
    --dataset_folder "-11_-15dB" \
    --train_dataset_folder unlabeled_10k \
    --noise_mode noisySignal \
    --n_bins 5 --top_k 5 \
    --prediction_source centroid
```

##### OOD Data Layout

The OOD test dataset only requires a `test/` directory:

```
data/own/-11_-15dB/
├── test/
│   ├── noisySignal/          # .npy raw signals
│   ├── noiselessSignal/      # (optional)
│   ├── noisyImg/             # (optional, needed for embedding features)
│   └── noiseLessImg/         # (optional)
│
│  ── Generated by the pipeline ──
└── test_centroid_noisySignal_5_5_data.pkl   # Step 6b output
```

The training dataset (`unlabeled_10k`) must already have a train `.pkl`:

```
data/own/unlabeled_10k/
├── train/                    # Signal files + few-shot examples
├── train_centroid_noisySignal_5_5_data.pkl   # Pre-fitted transformers
└── ...
```

---

## RAG — Retrieval-Augmented Few-Shot Selection

The pipeline includes an **optional RAG (Retrieval-Augmented Generation)** module
that replaces the default random/diversity-based few-shot example selection with
**similarity-based retrieval**.  The implementation is inspired by the general RAG
paradigm introduced in *"Retrieval-Augmented Generation for Knowledge-Intensive
NLP Tasks"* (Lewis et al., NeurIPS 2020), adapted here for signal-level
few-shot prompt construction rather than open-domain text generation.

### Motivation

In the default pipeline, few-shot examples included in each LLM prompt are drawn
from a static pool using `reduce_example_dict()`, which selects examples with a
diversity heuristic (no more than a fixed number per class, shuffled randomly).
While this ensures broad class coverage, it does **not** consider the similarity
between the test signal and the examples — a prompt asking about a fading OQPSK
signal at −5 dB may receive examples from completely unrelated conditions.

RAG addresses this by retrieving the **k most similar training signals** for each
test query, so the LLM sees examples whose statistical fingerprints are closest
to the signal it needs to classify.

### How It Works in This Repo

The RAG system operates in two phases, both integrated into
`generated_dataset.py` (Step 6 of the pipeline):

#### Phase 1 — Index Construction (Train Time)

When `--use_rag` is passed during `--mode train`:

1. The pipeline computes scaled statistical feature vectors for every training
   signal (the same 19-dimensional vectors produced by `StandardScaler` —
   SNR, skewness, kurtosis, 10 statistical moments, 4 k-statistics, and
   2 k-statistic variances).
2. These vectors are inserted into a **FAISS** `IndexFlatL2` index for exact
   L2 nearest-neighbour search.  For datasets larger than ~100k signals, an
   `IndexIVFFlat` (approximate search with Voronoi partitioning) is used
   instead.
3. The index and associated metadata (signal paths, labels, SNRs) are persisted
   to disk next to the train `.pkl`:

```
data/own/unlabeled_10k/
├── train_centroid_noisySignal_5_5_data.pkl       # existing train data
├── train_centroid_noisySignal_5_5_rag.index      # FAISS binary index
└── train_centroid_noisySignal_5_5_rag_meta.pkl   # metadata (paths, labels, SNRs)
```

#### Phase 2 — Retrieval (Test Time)

When `--use_rag` is passed during `--mode test`:

1. The FAISS index and metadata are loaded from disk.
2. For **each test signal**, its scaled feature vector is used as a query
   against the index to retrieve the `k` nearest training signals
   (controlled by `--rag_k`, default 10).
3. The retrieved signals are loaded and formatted into the `example_dict`
   structure expected by `generate_prompt()`, replacing the static pool.
4. Self-retrieval is prevented — if the test signal also exists in the
   training set, it is excluded from its own results.

```
Test signal (feature vector)
        │
        ▼
┌──────────────────────────────┐
│   FAISS Index (L2 search)    │  Trained on N training feature vectors
│   → k nearest neighbours     │
└──────────┬───────────────────┘
           │
           ▼
  Retrieved signals + labels + SNRs
           │
           ▼
  example_dict  →  generate_prompt()  →  LLM prompt
```

### Usage

**Install FAISS** (required only when RAG is enabled):

```bash
pip install faiss-cpu    # CPU-only
# or
pip install faiss-gpu    # CUDA-accelerated search
```

**Via `run_pipeline.sh`:**

```bash
# In run_pipeline.sh, set:
USE_RAG=true
RAG_K=10      # neighbours per test signal
```

**Via CLI directly:**

```bash
cd src/prompt/

# 1. Build train data + RAG index
python generated_dataset.py \
  --mode train \
  --dataset_folder unlabeled_10k \
  --noise_mode noisySignal \
  --n_bins 5 --top_k 5 \
  --prediction_source centroid \
  --use_rag --rag_k 10

# 2. Build test data with RAG retrieval
python generated_dataset.py \
  --mode test \
  --dataset_folder unlabeled_10k \
  --noise_mode noisySignal \
  --n_bins 5 --top_k 5 \
  --prediction_source centroid \
  --use_rag --rag_k 10
```

**OOD + RAG** — works with out-of-distribution mode as well:

```bash
python generated_dataset.py \
  --mode test \
  --dataset_folder "-11_-15dB" \
  --train_dataset_folder unlabeled_10k \
  --noise_mode noisySignal \
  --n_bins 5 --top_k 5 \
  --prediction_source centroid \
  --use_rag --rag_k 10
```

### Backward Compatibility

RAG is **fully optional**.  When `--use_rag` is not passed (or `USE_RAG=false`
in the shell script), the pipeline behaves exactly as before — FAISS is never
imported, no index is built, and few-shot examples are selected using the
original `reduce_example_dict()` diversity heuristic.

| Flag | Example selection method |
|------|-------------------------|
| *(default)* | Random diversity pool via `reduce_example_dict()` |
| `--use_rag` | FAISS L2 nearest-neighbour retrieval via `rag.py` |

---

## Project Structure

```
DiSC-AMC/
├── src/
│   ├── representation learning/   # Vision models, training, inference
│   │   ├── autoencoder_vit.py     # DINO ViT autoencoder + decoders
│   │   ├── autoencoders.py        # ResNet autoencoder
│   │   ├── classifier_training.py # ImageClassifier model + training loop
│   │   ├── compute_centroids.py   # CLI to compute per-class centroids
│   │   ├── convert_predictions.py # CLI to convert .png keys to .npy in prediction JSONs
│   │   ├── data_loader.py         # SpectogramDataset, ConstilationDataset, DatasetWithPath
│   │   ├── embedding_pipeline.py  # DINO features → PCA → discretize → centroids
│   │   ├── inference.py           # Test evaluation, per-image predictions
│   │   ├── processing.py          # Spectrogram helpers (power dB, colormap)
│   │   └── constants.py           # FFT_SIZE, SAMPLING_RATE, etc.
│   │
│   ├── naming.py                  # Centralized prediction-source naming conventions
│   │
│   ├── prompt/                    # Signal → statistical features → LLM prompts
│   │   ├── data_processing.py     # Feature extraction, scaling, discretisation, prompt gen
│   │   ├── embedding_features.py  # Encoder embeddings → PCA → feature dicts for prompts
│   │   ├── generated_dataset.py   # Full dataset builder (train/test .pkl files)
│   │   ├── rag.py                 # Optional FAISS-based RAG for few-shot example retrieval
│   │   ├── templates.py           # Prompt templates, modulation families, class names
│   │   ├── visualization.py       # t-SNE, confusion matrix, Plotly helpers
│   │   ├── baseline.py            # Traditional ML baselines (SVM, RF, KNN, etc.)
│   │   ├── radioml.py             # RadioML-specific dataset processing
│   │   ├── sfa.py                 # Symbolic Fourier Approximation experiments
│   │   └── __init__.py            # Package exports
│   │
│   └── evaluation/                # LLM response collection & metrics
│       ├── utils.py               # Shared I/O, sampling, accuracy metrics
│       ├── gemini_googleai.py     # Gemini API provider
│       ├── gpt_openai.py          # OpenAI GPT provider
│       └── unsloth_eval.py       # Local Unsloth model provider
│
├── notebooks/                     # Jupyter notebooks for experiments
├── data/                          # Datasets (RadioML, own signals)
├── models/                        # Downloaded model checkpoints
└── exp/                           # Trained weights & experiment logs
```

### Expected Data Layout

Each dataset folder under `data/own/` should have the following structure:

```
data/own/unlabeled_10k/
├── train/
│   ├── noisySignal/            # .npy raw signals (noisy)
│   ├── noiselessSignal/        # .npy raw signals (clean)
│   ├── noisyImg/               # .png constellation images (noisy)
│   ├── noiseLessImg/           # .png constellation images (clean)
│   └── class_centers.json      # Per-class centroids (Step 3)
├── test/
│   ├── noisySignal/
│   ├── noiselessSignal/
│   ├── noisyImg/
│   └── noiseLessImg/
│
│  ── Generated by the pipeline ──
├── top5_dnn_predictions.json           # Step 4b — .png keys, classifier head
├── top5_centroid_predictions.json      # Step 4c — .png keys, centroids
├── ntop5_dnn_predictions.json          # Step 5  — .npy keys, classifier head
├── ntop5_centroid_predictions.json     # Step 5  — .npy keys, centroids
├── train_noisySignal_5_5_data.pkl      # Step 6a — train data (DNN source)
├── train_centroid_noisySignal_5_5_data.pkl  # Step 6a — train data (centroid source)
├── test_noisySignal_5_5_data.pkl       # Step 6b — test data (DNN source)
├── test_centroid_noisySignal_5_5_data.pkl   # Step 6b — test data (centroid source)
├── train_centroid_noisySignal_5_5_emb10_data.pkl  # Step 6a — embedding features (10 PCA)
└── test_centroid_noisySignal_5_5_emb10_data.pkl   # Step 6b — embedding features (10 PCA)
```

---

## End-to-End Pipeline

Every step below is a self-contained CLI script — no notebook cells needed.

### Step 1 — Train the Autoencoder

Train a DINO ViT or ResNet autoencoder for self-supervised feature learning on constellation/spectrogram images. Use the notebooks (`autoencoder.ipynb`) or the training scripts in `src/representation learning/`.

### Step 2 — Train the Classifier

```bash
cd src/representation\ learning/

python classifier_training.py \
  --model dino \
  --base_data_path ../../data/own/unlabeled_10k/train \
  --pretrained_path ../../exp/dino_autoencoder.pth \
  --save_path ../../exp/dino_classifier.pth \
  --num_epochs 100 \
  --learning_rate 1e-4 \
  --image_size 96
```

This trains an `ImageClassifier` (DINO encoder + linear head) and saves the weights.

### Step 3 — Compute Class Centroids

```bash
cd src/representation\ learning/

python compute_centroids.py \
  --backbone dino \
  --weights ../../exp/dino_classifier.pth \
  --dataset_path ../../data/own/unlabeled_10k/train \
  --output ../../data/own/unlabeled_10k/train/class_centers.json \
  --find_closest
```

Computes per-class centroids in the encoder's feature space.  These are used both for centroid-based predictions and for narrowing the LLM's option set (top-k neighbors).

### Step 4 — Generate Top-k Predictions

**4a.  Evaluate on a test set** (optional, prints top-1/top-k accuracy):

```bash
cd src/representation\ learning/

python inference.py evaluate \
  --backbone dino \
  --weights ../../exp/dino_classifier.pth \
  --test_path ../../data/own/unlabeled_10k/test \
  --centroid_path ../../data/own/unlabeled_10k/train/class_centers.json \
  --topk 5
```

**4b.  Per-image top-k predictions** (classifier head):

```bash
python inference.py predict \
  --backbone dino \
  --weights ../../exp/dino_classifier.pth \
  --dataset_path ../../data/own/unlabeled_10k/test \
  --topk 5 \
  --output ../../data/own/unlabeled_10k/top5_dnn_predictions.json
```

**4c.  Per-image top-k predictions** (centroid-based):

```bash
python inference.py predict \
  --backbone dino \
  --weights ../../exp/dino_classifier.pth \
  --dataset_path ../../data/own/unlabeled_10k/test \
  --centroid_path ../../data/own/unlabeled_10k/train/class_centers.json \
  --topk 5 \
  --output ../../data/own/unlabeled_10k/top5_centroid_predictions.json
```

### Step 5 — Convert `.png` Keys to `.npy`

The predictions JSON keys reference `.png` constellation images, but the prompt pipeline works with `.npy` signal files.  Convert them:

```bash
cd src/representation\ learning/

# For classifier-head predictions
python convert_predictions.py \
  --input ../../data/own/unlabeled_10k/top5_dnn_predictions.json \
  --output ../../data/own/unlabeled_10k/ntop5_dnn_predictions.json

# For centroid-based predictions
python convert_predictions.py \
  --input ../../data/own/unlabeled_10k/top5_centroid_predictions.json \
  --output ../../data/own/unlabeled_10k/ntop5_centroid_predictions.json
```

### Step 6 — Generate LLM Prompt Datasets

Build the `.pkl` dataset files containing signal features + formatted prompts.
The `--prediction_source` flag controls which top-k JSON to load:

| Source | Reads JSON | Output naming |
|--------|-----------|---------------|
| `dnn` | `ntop<k>_dnn_predictions.json` | `test_noisySignal_5_5_data.pkl` |
| `centroid` | `ntop<k>_centroid_predictions.json` | `test_centroid_noisySignal_5_5_data.pkl` |
| `rf` | `ntop<k>_rf_predictions.json` | `test_rf_noisySignal_5_5_data.pkl` |

All naming patterns are defined centrally in [`src/naming.py`](src/naming.py) —
adding a new source requires editing only that file.

```bash
cd src/prompt/

# 6a. Build TRAIN data (fits scaler & discretizers)
python generated_dataset.py \
  --mode train \
  --dataset_folder unlabeled_10k \
  --noise_mode noisySignal \
  --n_bins 5 --top_k 5 \
  --prediction_source centroid

# 6b. Build TEST data (loads train scaler + top-k predictions)
python generated_dataset.py \
  --mode test \
  --dataset_folder unlabeled_10k \
  --noise_mode noisySignal \
  --n_bins 5 --top_k 5 \
  --prediction_source centroid
```

**OOD test** — test on a different dataset, reusing training transformers:

```bash
cd src/prompt/

python generated_dataset.py \
  --mode test \
  --dataset_folder "-11_-15dB" \
  --train_dataset_folder unlabeled_10k \
  --noise_mode noisySignal \
  --n_bins 5 --top_k 5 \
  --prediction_source centroid
```

**Embedding features** — use encoder embeddings (DINO or ResNet) instead of statistical features:

```bash
cd src/prompt/

# 6a. TRAIN — fits PCA + scaler + discretizers on training embeddings
python generated_dataset.py \
  --mode train \
  --dataset_folder unlabeled_10k \
  --noise_mode noisySignal \
  --n_bins 5 --top_k 5 \
  --prediction_source centroid \
  --feature_type embeddings \
  --encoder_weights ../../exp/dino_classifier.pth \
  --backbone dino \
  --n_components 10

# 6b. TEST — reuses PCA/scaler/discretizers + top-k predictions
python generated_dataset.py \
  --mode test \
  --dataset_folder unlabeled_10k \
  --noise_mode noisySignal \
  --n_bins 5 --top_k 5 \
  --prediction_source centroid \
  --feature_type embeddings \
  --encoder_weights ../../exp/dino_classifier.pth \
  --backbone dino \
  --n_components 10
```

Embedding-mode pkl files include an `emb{N}` tag in the filename (e.g.
`test_centroid_noisySignal_5_5_emb10_data.pkl`). The output dict additionally
contains a `'pca'` key with the fitted PCA transformer.

This produces files like `test_centroid_noisySignal_5_5_data.pkl` containing:
- Raw signal data, labels, SNRs
- Scaled & discretized statistical features
- `old_prompts` and `old_discret_prompts` (basic template)
- `prompts` and `discret_prompts` (engineered template with top-k narrowed options)

### Step 7 — Query LLMs

```bash
cd src/evaluation/

# Gemini
python gemini_googleai.py

# OpenAI
python gpt_openai.py

# Local Unsloth model
python unsloth_eval.py
```

Each script loads the `.pkl` data, sends prompts to the respective API, and saves structured JSON results.

### Step 8 — Compute Metrics

The evaluation scripts print metrics at the end automatically.  To re-compute from saved results:

```bash
cd src/evaluation/
python -c "
from gemini_googleai import read_results
from utils import sort_results_by_prompt, print_metrics

CLASS_NAMES = ['4ASK','4PAM','8ASK','16PAM','CPFSK','DQPSK','GFSK','GMSK','OQPSK','OOK']
results = read_results('discret_prompts', 'gemini-2.5-flash', 'noisySignal', 5, 5)
sorted_results = sort_results_by_prompt(results)
print_metrics(sorted_results, CLASS_NAMES)
"
```

---

## Module Reference

### Representation Learning

**`src/representation learning/`**

#### `constants.py`

| Constant | Value | Description |
|----------|-------|-------------|
| `FFT_SIZE` | 64 | FFT window size for spectrogram generation |
| `SAMPLING_RATE` | 20e6 | Signal sampling rate (Hz) |
| `CENTER_FREQ` | 2.447e9 | Center frequency (Hz) |

#### `processing.py`

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `visualize_signal(signal, Fs)` | Complex signal `np.ndarray`, sampling rate | `matplotlib.Figure` | Plots signal magnitude over time |
| `get_power_spectrogram_db(stft_matrix, ...)` | Complex STFT matrix | `np.ndarray` (dB-scaled power) | Converts STFT to power spectrogram in dB |
| `get_color_img(spectrum_db, colormap)` | Power spectrogram `np.ndarray` | RGB `np.ndarray` in [0,1] | Applies colormap to spectrogram for image representation |

#### `data_loader.py`

| Class | Input | Output (`__getitem__`) | Description |
|-------|-------|------------------------|-------------|
| `SpectogramDataset` | `dataset_path`, `classes`, `fft_size`, `transform` | `(image_tensor, label_int)` | Loads `.npy` signals, converts to spectrogram images on-the-fly |
| `ConstilationDataset` | `dataset_path`, `classes`, `transform` | `(image_tensor, label_int)` | Loads pre-rendered constellation `.png` images from `noisyImg/` and `noiseLessImg/` |
| `DatasetWithPath` | `dataset_path`, `classes`, `transform` | `(image_tensor, label_int, file_path)` | Wraps `ConstilationDataset` to also return the source file path |

#### `autoencoder_vit.py`

| Class | Description |
|-------|-------------|
| `DinoV2Autoencoder` | DINO ViT-B/8 encoder (frozen or trainable) + lightweight decoder. Latent dim = 768 |
| `ResNetDecoder` | Transpose-conv decoder from 768-dim latent to 96x96 RGB |
| `ShallowResNetDecoder` | Parameter-efficient variant of `ResNetDecoder` |
| `LightViTDecoder` | ViT-inspired decoder with progressive upsampling |

#### `autoencoders.py`

| Class | Description |
|-------|-------------|
| `ResNetAutoEncoder` | ResNet-34/50 encoder + symmetric decoder for self-supervised image reconstruction |
| `ResNetEncoder` | Configurable ResNet encoder with residual/bottleneck blocks |
| `ResNetDecoder` | Symmetric decoder with transposed convolutions |

#### `classifier_training.py`

| Symbol | Type | Input / Output | Description |
|--------|------|-----------------|-------------|
| `ImageClassifier` | `nn.Module` | `(B, 3, H, W)` -> `(B, num_classes)` | Generalized classifier: DINO or ResNet backbone + classification head |
| `topk_accuracy(output, target, k)` | function | Logits tensor, labels tensor, int -> `float` | Counts correct top-k predictions in a batch |
| `topk_centroid_accuracy(features, centroid_tensor, target, k)` | function | Feature tensor `(B, D)`, centroids `(C, D)`, labels, int -> `float` | Top-k accuracy by nearest centroid distance |
| `TopKLoss` | `nn.Module` | Logits, targets -> scalar loss | Cross-entropy only on samples where the target is **not** in the top-k |
| `main(args)` | function | `argparse.Namespace` | Full training loop with validation, test evaluation, and model saving |

**CLI arguments for `classifier_training.py`:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | (required) | `dino` or `resnet` |
| `--base_data_path` | `../../data/own/unlabeled_10k/train` | Training data directory |
| `--pretrained_path` | `../../exp/dino_autoencoder.pth` | Path to pretrained autoencoder |
| `--save_path` | `../../exp/dino_classifier.pth` | Where to save trained classifier |
| `--freeze_encoder` | `False` | Freeze encoder weights during training |
| `--num_epochs` | 100 | Training epochs |
| `--learning_rate` | 1e-4 | Learning rate |
| `--batch_size` | 32 | Batch size |
| `--image_size` | 96 | Input image resolution |

#### `embedding_pipeline.py`

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `extract_label(filename)` | Image filename string | Modulation class string | Parses label from filename (e.g. `"16PAM_0.01dB__..."` -> `"16PAM"`) |
| `load_encoder(model_class, weights_path, device)` | Model class, weights path | `(encoder, device)` | Loads model, extracts `.encoder`, sets to eval |
| `extract_embeddings(encoder, device, image_dir, batch_size)` | Encoder, directory of `.png` images | `(filenames, embeddings)` where embeddings is `(N, latent_dim)` | Batch-encodes all images in a directory |
| `process_split(encoder, device, split_path, ...)` | Encoder, data split root | `(results_dict, pca_dict, disc_dict)` | Full PCA -> discretize pipeline per feature type |
| `compute_class_centroids(encoder, device, dataset, classes, ...)` | Encoder, dataset with paths, class list | `{class_name: mean_feature_vector}` | Computes mean encoder feature for each class |
| `find_closest_to_centroids(encoder, device, dataset, classes, centroids, ...)` | Encoder, dataset, centroids dict | `{class_name: closest_sample_path}` | Finds the sample nearest to each centroid |
| `save_centroids(centroids, output_path)` | Centroids dict | JSON file on disk | Saves centroids as JSON (numpy -> list) |
| `load_centroids(path, device)` | JSON path | `(centroid_tensor, class_names)` | Loads centroids as a ready-to-use `torch.Tensor` |
| `save_results(results, output_path)` | Results dict | JSON file | Persists discretized features |
| `load_results(path)` | JSON path | Results dict | Loads previously saved features |
| `run_pipeline(...)` | Dataset folder, weights, etc. | Merged results dict | End-to-end: load -> train/test split -> save JSON |

#### `inference.py`

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `load_classifier(backbone, weights_path, num_classes, ...)` | Backbone type, weights path | `(ImageClassifier, device)` | Loads a trained classifier in eval mode |
| `evaluate_test_set(model, device, test_dataset_path, ...)` | Model, test path, optional centroid path | `dict` with `top1_accuracy`, `topk_accuracy`, `centroid_topk_accuracy` | Full test-set evaluation with multiple metrics |
| `predict_topk_classifier(model, device, dataset_path, ...)` | Model, image directory | `{filename: {img_type: [class1, ...]}}` | Per-image top-k predictions via classifier head |
| `predict_topk_centroids(model, device, dataset_path, centroid_path, ...)` | Model, image directory, centroids JSON | `{filename: {img_type: [class1, ...]}}` | Per-image top-k predictions via nearest centroid |

**CLI subcommands for `inference.py`:**

```
python inference.py evaluate --backbone dino --weights PATH --test_path PATH [--centroid_path PATH] [--topk 5]
python inference.py predict  --backbone dino --weights PATH --dataset_path PATH --output PATH [--centroid_path PATH] [--topk 5]
```

#### `compute_centroids.py`

Standalone CLI script to compute per-class centroids from a trained classifier's encoder features.

```
python compute_centroids.py \
  --backbone dino \
  --weights ../../exp/dino_classifier.pth \
  --dataset_path ../../data/own/unlabeled_10k/train \
  --output ../../data/own/unlabeled_10k/train/class_centers.json \
  [--find_closest] [--image_size 96] [--batch_size 32]
```

#### `convert_predictions.py`

Standalone CLI script to convert `.png` filenames to `.npy` in prediction JSON keys.

```
python convert_predictions.py \
  --input ../../data/own/unlabeled_10k/top5_centroid_predictions.json \
  --output ../../data/own/unlabeled_10k/ntop5_centroid_predictions.json
```

---

### Shared Utilities

#### `src/naming.py`

Centralized naming conventions for prediction sources.  Every pipeline step imports from this module, so adding a new source only requires editing this file.

| Symbol | Type | Description |
|--------|------|-------------|
| `PredictionSource` | `dataclass` | Holds `key`, `label`, `raw_json`, `converted_json`, `pkl_tag` for one source |
| `SOURCES` | `dict[str, PredictionSource]` | Registry: `"dnn"`, `"centroid"`, `"rf"` |
| `VALID_SOURCES` | `list[str]` | Keys of `SOURCES` — used as CLI `choices` |
| `get_source(name)` | function | Look up by key, raises `ValueError` if unknown |
| `raw_json_name(source, top_k)` | function | `.png`-keyed JSON filename (inference output) |
| `converted_json_name(source, top_k)` | function | `.npy`-keyed JSON filename (after conversion) |
| `train_pkl_name(source, noise_mode, n_bins, top_k, feature_tag="")` | function | Train `.pkl` filename (tag adds e.g. `_emb10`) |
| `test_pkl_name(source, noise_mode, n_bins, top_k, feature_tag="")` | function | Test `.pkl` filename (tag adds e.g. `_emb10`) |

To add a new prediction source (e.g. XGBoost):

```python
# In src/naming.py
SOURCES["xgb"] = PredictionSource(
    key="xgb",
    label="XGBoost",
    raw_json="top{topk}_xgb_predictions.json",
    converted_json="ntop{topk}_xgb_predictions.json",
    pkl_tag="xgb",
)
```

All downstream scripts (`generated_dataset.py`, `utils.py`, CLI `--help`) automatically pick up the new source.

---

### Prompt Engineering

**`src/prompt/`**

#### `templates.py`

Defines all prompt templates and modulation family constants:

| Symbol | Type | Description |
|--------|------|-------------|
| `PROMPT_TEMPLATE` | `str` | Basic instruction template for LLM classification |
| `NEW_PROMPT_TEMPLATE` | `str` | Instruction with cumulant-based domain context |
| `PROMPT_ENGINEERED_TEMPLATE` | `str` | Detailed role/objective/context template for engineered prompts |
| `INPUT_TEXT` / `INPUT_ENGINEERED_TEXT` | `str` | Question formatting templates (`"### Question: {} OPTIONS: {} ### Answer: {}"`) |
| `MODULATION_FAMILIES` | `dict` | Hierarchical grouping of modulation schemes |
| `CLASS_NAMES` | `list[str]` | Full list of 24 modulation classes (RadioML) |
| `FEATURE_NAMES` | `list[str]` | Default statistical feature names (23 features) |

#### `data_processing.py`

Core signal processing and prompt generation functions:

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `load_npy_file(path)` | File path | `np.ndarray` | Load raw signal data |
| `get_features(signal, feature_names, snr)` | Signal array, feature list | `dict[str, float or ndarray]` | Compute statistical features (moments, cumulants, kurtosis, etc.) |
| `dict_to_np(summary, feature_names)` | Feature dict | `np.ndarray` (1D) | Flatten feature dict to array |
| `discretize_features(stats, n_bins, strategy, discretizers)` | 2D feature array | `(discretized_array, discretizers_dict)` | Per-column KBins discretization (fit or transform) |
| `get_discrete_info(info, discretizers)` | Feature dict, fitted discretizers | `dict[str, int]` | Discretize a single sample's features |
| `get_scaled_info(info, scaler)` | Feature dict, fitted StandardScaler | `dict[str, float]` | Scale a single sample's features |
| `get_text_info(info, decimal_precision)` | Feature dict | `str` | Format features as text: `"kurtosis: 3.142, ..."` |
| `get_discrete_text_info(info)` | Discretized feature dict | `str` | Format bin indices as letters: `"kurtosis: C, ..."` |
| `create_options(class_names)` | List of class names | `str` | Format as `"[A: OOK, B: 4ASK, ...]"` |
| `get_question_answer(signal, options, template, ...)` | Signal/features, options, template | `str` | Generate a single Q&A formatted string |
| `generate_prompt(signal_data, ...)` | Signal, templates, examples | `str` | Full prompt: instruction + few-shot context + question |
| `get_processed_data(signal_paths, labels, ...)` | File paths, labels, SNRs | `dict` with signals, stats, prompts, etc. | Process a batch of signals into a complete dataset dict |
| `reduce_example_dict(example_dict, label, max_examples)` | Example dict, target label | Reduced example dict | Select diverse few-shot examples ensuring label coverage |
| `save_processed_data(data, path)` | Any data, file path | `.pkl` file | Pickle serialization |
| `load_processed_data(path)` | File path | Loaded data | Pickle deserialization |
| `convert_signal_to_complex(signal)` | 2D array (N, 2) | 1D complex array | Convert I/Q to complex representation |

> **Note:** `generate_prompt()` accepts an `examples_processed` flag (default `False`).
> Set it to `True` when the few-shot examples are already pre-processed feature dicts
> (e.g. from the embedding pipeline) rather than raw signals.

#### `embedding_features.py`

Bridges encoder embeddings with the prompt generation pipeline
(Encoder → PCA → Discretize → Letter-encode):

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `signal_path_to_image_path(signal_path, noise_mode)` | `.npy` signal path | `.png` image path | Map `noisySignal/foo.npy` → `noisyImg/foo.png` |
| `pca_feature_names(n_components)` | Int | `['pc_0', …, 'pc_N']` | Canonical PCA feature names |
| `embedding_to_feature_dict(vector, feature_names)` | 1D array, names | `dict[str, float]` | Wrap a PCA-reduced vector in a feature dict |
| `extract_embeddings_from_paths(encoder, device, image_paths, batch_size)` | Encoder, image paths | `np.ndarray (N, latent_dim)` | Batch-extract encoder embeddings from `.png` files |
| `compute_embedding_features(embeddings, n_components, n_bins, …)` | Raw embeddings | `(feat_dicts, disc_dicts, scaled_dicts, names, pca, disc, scaler)` | Full PCA → discretize → scale pipeline |
| `prepare_example_embedding_dicts(encoder, device, example_paths, …)` | Example signal paths | `(scaled_ex_dict, discret_ex_dict)` | Pre-process few-shot examples for `examples_processed=True` |
| `load_encoder_for_embeddings(backbone, weights_path, …)` | Backbone type, `.pth` path | `(encoder, device)` | Load a trained classifier's encoder sub-module |

#### `generated_dataset.py`

End-to-end dataset builder that combines all prompt engineering steps:

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `get_dataset_label(path)` | Signal file path | `str` (e.g. `"4ASK"`) | Extract label from filename |
| `get_dataset_snr(path)` | Signal file path | `str` (e.g. `"-5.57"`) | Extract SNR from filename |
| `create_dataset_example_paths(base_dir, noise_mode)` | Base directory | `dict[class: [path]]` | Hardcoded few-shot example paths per class |
| `dataset_example_maker(base_dir, mode, noise_mode)` | Base directory, mode | `(signal_paths, example_dict, labels, snrs)` | Prepare signal paths and few-shot examples |
| `get_processed_data(...)` | Signal paths, labels, example paths, scalers, ktop_info | Complete `dict` | Generates all prompt variants and saves as `.pkl` |
| `get_embedding_processed_data(...)` | Signal paths, encoder, noise_mode, PCA params | Complete `dict` | Embedding variant: images → encoder → PCA → prompts |

**Running the dataset builder:**

```bash
cd src/prompt/

# Build TRAIN dataset (fits scaler & discretizers)
python generated_dataset.py --mode train --dataset_folder unlabeled_10k \
    --noise_mode noisySignal --n_bins 5 --top_k 5 --prediction_source centroid

# Build TEST dataset (requires train .pkl + matching predictions JSON)
python generated_dataset.py --mode test --dataset_folder unlabeled_10k \
    --noise_mode noisySignal --n_bins 5 --top_k 5 --prediction_source centroid
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | (required) | `train` or `test` |
| `--dataset_folder` | `unlabeled_10k` | Folder name under `--data_root` |
| `--train_dataset_folder` | `None` | If set, load the train `.pkl` and few-shot examples from this folder (OOD mode) |
| `--noise_mode` | `noisySignal` | `noisySignal` or `noiselessSignal` |
| `--n_bins` | `5` | Number of KBinsDiscretizer bins |
| `--top_k` | `5` | Top-k value matching predictions file |
| `--prediction_source` | `centroid` | `dnn`, `centroid`, or `rf` — controls which JSON to load (see [`naming.py`](src/naming.py)) |
| `--data_root` | `../../data/own` | Root data directory |
| `--use_rag` | `False` | Enable FAISS-based RAG for few-shot example retrieval |
| `--rag_k` | `10` | Number of nearest neighbours to retrieve when RAG is enabled |

**Output `.pkl` structure:**

| Key | Type | Description |
|-----|------|-------------|
| `signal_paths` | `list[str]` | Absolute paths to source `.npy` files |
| `signals` | `list[np.ndarray]` | Raw loaded signals |
| `stats` | `list[dict]` | Scaled continuous features per signal |
| `discret_stats` | `list[dict]` | Discretized feature bin indices per signal |
| `labels` | `list[str]` | Ground truth modulation labels |
| `snrs` | `list[str]` | Signal-to-noise ratios |
| `prompts` | `list[str]` | Engineered prompts with continuous features |
| `discret_prompts` | `list[str]` | Engineered prompts with discretized features |
| `old_prompts` | `list[str]` | Basic template prompts (continuous) |
| `old_discret_prompts` | `list[str]` | Basic template prompts (discretized) |
| `scaler` | `StandardScaler` | Fitted scaler (reuse for test set) |
| `discretizers` | `dict[int, KBinsDiscretizer]` | Fitted discretizers (reuse for test set) |
| `k-top` | `list or None` | Top-k classes per signal from centroid predictions |

#### `rag.py`

Optional FAISS-based Retrieval-Augmented Generation module for similarity-based
few-shot example selection. Only imported when `--use_rag` is enabled.

| Symbol | Type | Description |
|--------|------|-------------|
| `RAGRetriever` | `dataclass` | Wraps a FAISS index + metadata (signal paths, labels, SNRs, feature vectors) |
| `build_rag_index(signal_paths, labels, snrs, feature_vectors, train_pkl_path)` | function | Builds a FAISS `IndexFlatL2` (or `IndexIVFFlat` for large datasets) and persists to disk as `*_rag.index` + `*_rag_meta.pkl` |
| `load_rag_index(train_pkl_path)` | function | Loads a previously built index and metadata from disk |
| `retrieve_examples(retriever, query_vector, rag_k)` | function | Returns the `k` nearest training signals as `{label: [(path, snr), ...]}` |
| `rag_example_dict_from_paths(retrieved)` | function | Converts retrieved paths to loaded signal arrays matching the `example_dict` format |
| `retrieve_example_dict_for_signal(retriever, query_feature_vector, rag_k)` | function | End-to-end helper: retrieve → load → return `example_dict` for `generate_prompt()` |

**Dependencies:** `faiss-cpu` or `faiss-gpu` (lazy-imported, not required when RAG is disabled).

#### `visualization.py`

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `visualize_tsne(data, labels, ...)` | Feature array, labels | `(fig_2d, fig_3d)` Plotly figures | t-SNE visualization with PCA pre-reduction |
| `plot_confusion_matrix(y_true, y_pred, class_names)` | True/predicted labels | Plotly heatmap figure | Annotated confusion matrix |
| `generate_distinct_colors(n)` | Integer | `list[str]` hex colors | HSV-based distinct color generation |
| `save_figure_as_html(fig, path)` | Plotly figure, path | `.html` file | Save interactive plot |

#### `baseline.py`

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `train_and_evaluate_models(train_data, test_data)` | Processed data dicts | `dict[model_name: {accuracy, predictions, time}]` | Trains SVM, Random Forest, Gradient Boosting, KNN, Logistic Regression |

#### `radioml.py`

Same structure as `generated_dataset.py` but adapted for the RadioML 2018 dataset format where labels and SNRs are encoded in the directory structure.

#### `sfa.py`

Experimental module for Symbolic Fourier Approximation (SFA) of I/Q signals using `pyts`.

---

### LLM Evaluation

**`src/evaluation/`**

#### `utils.py` — Shared Utilities

**Data Loading & Sampling:**

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `load_data(data_path, noise_mode, n_bins, top_k)` | Base path, noise mode, bin/topk params | `(sampled_data, indices, rng)` | Loads `.pkl`, samples 20 per label |
| `sample_per_label(data, per_label, seed, shuffle)` | Data dict, samples per class | `(sampled_data, selected_indices, rng)` | Stratified sampling with reproducible RNG |
| `build_prompts_data(data, prompt_type, limit)` | Data dict, prompt key | `list[{prompt, filename}]` | Extracts prompts with filenames |

**Result I/O:**

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `load_existing_results(filepath, num_tries, key_field)` | JSON path | `(results, prompts_done, completed_set)` | Resume-friendly result loading |
| `get_prompts_to_process(all_prompts, completed, prompts_done)` | Prompt list, completed set | `list[(prompt_data, num_done)]` | Filter out already-completed prompts |
| `save_results_atomic(results, filepath)` | Results list, path | JSON file (atomic write) | Safe save with tmp file + rename |
| `build_result_entry(filename, try_idx, prompt, raw_response, ...)` | Response data | Standardized result `dict` | Extracts `<think>` reasoning, response label, true label |

**Accuracy Metrics:**

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `acc(sorted_results)` | `dict[int, list[dict]]` | `(correct, total, accuracy)` | Raw accuracy across all tries |
| `clean_acc(sorted_results, class_names)` | Results, valid class list | `(correct, total, accuracy)` | Accuracy only on responses containing a valid class name |
| `pass_acc(sorted_results)` | Results dict | `(passed, total, accuracy)` | pass@k: at least one correct in k tries |
| `majority_acc(sorted_results)` | Results dict | `(passed, total, accuracy)` | Majority vote across k tries |
| `per_class_acc(sorted_results, class_names)` | Results, class list | `dict[class: accuracy]` | Per-class breakdown |
| `print_metrics(sorted_results, class_names)` | Results, class list | Printed output | Prints all metrics at once |

**Text Extraction:**

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `extract_think_text(response)` | Raw LLM response string | `(end_index, reasoning_text)` | Extracts content between `<think>` tags |
| `extract_tag(response, start_tag, end_tag)` | Response, tag pair | `(end_index, extracted_text)` | Generic tag extraction |
| `find_classes_in_text(text, class_names)` | Text, valid classes | `list[str]` | Finds which class names appear in text |

#### `gemini_googleai.py` — Gemini Provider

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `get_gemini_response(prompt, model, temperature)` | Prompt string, Gemini model | Response string | Single Gemini API call |
| `clean_up_response_label(results)` | Results list | `(cleaned_results, problematic)` | Fix Gemini Flash artifact (overly long labels) |
| `main(prompt_type, model_name, noise_mode, n_bins, top_k, num_tries)` | Config params | JSON file on disk | Full evaluation loop: load data -> query API -> save |
| `read_results(prompt_type, model_name, ...)` | Config params | `list[dict]` | Load saved results |

**Environment:** Requires `GEMINI_API_KEY` in `.env`.

#### `gpt_openai.py` — OpenAI Provider

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `get_openai_response(client, prompt, model)` | OpenAI client, prompt | Response string | Single OpenAI API call with 3 retries |
| `main(...)` | Config params | JSON file on disk | Full evaluation loop |
| `read_results(...)` | Config params | `list[dict]` | Load saved results |

**Environment:** Requires `OPENAI_API_KEY` in `.env`.

#### `unsloth_eval.py` — Local Model Provider

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `load_model_and_tokenizer(model_name)` | Unsloth model name (e.g. `"unsloth/DeepSeek-R1-Distill-Qwen-32B-unsloth-bnb-4bit"`) | `(model, tokenizer)` | Loads 4-bit quantized model via Unsloth |
| `get_model_response(prompt, model, tokenizer, temperature, chat_template)` | Prompt, model objects | Response string | Local inference with configurable chat template |
| `main(dataset_folder, prompt_type, model_name, ...)` | Config params | JSON file on disk | Full evaluation loop |
| `read_results(...)` | Config params | `list[dict]` | Load saved results |

---

## Summary

Here is where each piece now lives:

| Cell | New Location | Function / Class |
|---------------------|-------------|------------------|
| `DinoClassifier` model definition | `classifier_training.py` | `ImageClassifier(backbone='dino', ...)` |
| `topk_accuracy()` | `classifier_training.py` | `topk_accuracy()` |
| `topk_centriod_accuracy()` | `classifier_training.py` | `topk_centroid_accuracy()` |
| Training loop (epochs, val, test) | `classifier_training.py` | `main(args)` (CLI) |
| Test evaluation with centroids | `inference.py` | `evaluate_test_set()` |
| Per-image classifier-head predictions -> JSON | `inference.py` | `predict_topk_classifier()` |
| Per-image centroid-based predictions -> JSON | `inference.py` | `predict_topk_centroids()` |
| `DatasetWithPath` wrapper | `data_loader.py` | `DatasetWithPath` |
| Centroid computation + closest sample finder | `embedding_pipeline.py` | `compute_class_centroids()`, `find_closest_to_centroids()` |
| Centroid save/load | `embedding_pipeline.py` | `save_centroids()`, `load_centroids()` |
| Pickle data exploration | N/A (exploratory) | Use `data_processing.load_processed_data()` |

---

## Key Libraries

- **[Unsloth](https://github.com/unslothai/unsloth)** — Memory-efficient LLM finetuning & inference
- **[vLLM](https://github.com/vllm-project/vllm)** — High-performance LLM serving
- **PyTorch / Torchvision** — Deep learning framework
- **scikit-learn** — PCA, KBinsDiscretizer, StandardScaler, ML baselines
- **[FAISS](https://github.com/facebookresearch/faiss)** — Vector similarity search for RAG-based example retrieval (optional)
- **Plotly** — Interactive visualizations (t-SNE, confusion matrices)

## License

[Insert License Here]
