# DiSC-AMC Repository Summary

> **DiSC-AMC** — Discrete Signal Classification for Automatic Modulation Classification

This repository implements an end-to-end pipeline for automatic modulation
classification (AMC) that bridges traditional signal processing with Large
Language Model (LLM) reasoning. Raw I/Q signals are converted to constellation
diagram images, encoded via self-supervised vision models (DINO, ResNet,
DenoMAE2), and then classified by LLMs using carefully engineered text prompts
built from the extracted signal features.

---

## Repository Structure

```
DiSC-AMC/
├── src/
│   ├── representation_learning/   # Vision models, training, inference
│   │   ├── autoencoder_vit.py     # DINO ViT autoencoder + decoders
│   │   ├── autoencoders.py        # ResNet autoencoder
│   │   ├── autoencoder_training.py # CLI to train DINO/ResNet autoencoders
│   │   ├── classifier_training.py # ImageClassifier model + training loop
│   │   ├── compute_centroids.py   # CLI to compute per-class centroids
│   │   ├── convert_predictions.py # .png keys → .npy keys in prediction JSONs
│   │   ├── data_loader.py         # SpectogramDataset, ConstilationDataset
│   │   ├── embedding_pipeline.py  # DINO features → PCA → discretize → centroids
│   │   ├── inference.py           # Test evaluation, DNN/centroid/FAISS predictions
│   │   ├── processing.py          # Spectrogram helpers (power dB, colormap)
│   │   └── constants.py           # FFT_SIZE, SAMPLING_RATE, etc.
│   │
│   ├── denoMAE2/                  # DenoMAE2 masked autoencoder (git submodule)
│   │   ├── main.py                # DenoMAE2 ViT model architecture
│   │   ├── pretrain.py            # Self-supervised pretraining loop
│   │   ├── finetune.py            # Downstream classification finetuning
│   │   ├── datagen.py             # Data generation utilities
│   │   └── util/                  # Positional embeddings, helpers
│   │
│   ├── finetuning/                # QLoRA finetuning for LLMs
│   │   ├── dataset.py             # SFT dataset builder from train .pkl files
│   │   ├── train.py               # QLoRA training with Unsloth + TRL SFTTrainer
│   │   └── __init__.py
│   │
│   ├── naming.py                  # Centralized naming conventions + ExperimentConfig
│   │
│   ├── prompt/                    # Signal → statistical features → LLM prompts
│   │   ├── data_processing.py     # Feature extraction, scaling, discretisation
│   │   ├── embedding_features.py  # Encoder embeddings → PCA → feature dicts
│   │   ├── generated_dataset.py   # Full dataset builder (train/test .pkl files)
│   │   ├── naming.py              # Prompt-level naming (PredictionSource registry)
│   │   ├── rag.py                 # FAISS-based RAG for few-shot example retrieval
│   │   ├── templates.py           # V1/V2 prompt templates, modulation families
│   │   ├── visualization.py       # t-SNE, confusion matrix, Plotly helpers
│   │   ├── baseline.py            # Traditional ML baselines (SVM, RF, KNN, etc.)
│   │   ├── radioml.py             # RadioML-specific dataset processing
│   │   ├── sfa.py                 # Symbolic Fourier Approximation experiments
│   │   └── __init__.py
│   │
│   └── evaluation/                # LLM response collection & metrics
│       ├── utils.py               # Shared I/O, sampling, accuracy metrics
│       ├── gemini_googleai.py     # Gemini API provider
│       ├── gpt_openai.py          # OpenAI GPT provider
│       ├── unsloth_eval.py        # Local Unsloth model provider (batched inference)
│       └── audit_experiments.py   # Experiment auditing & CSV regeneration
│
├── run_pipeline.sh                # Single-run pipeline (Steps 1–8)
├── run_experiments.sh             # Batch experiment grid runner
├── run_finetuning.sh              # QLoRA finetuning orchestration
├── notebooks/                     # Jupyter notebooks & plotting scripts
│   ├── plot.py                    # Detailed experiment visualization
│   └── plot_results.py            # Publication-quality bar charts (5 figures)
├── data/                          # Datasets (RadioML, own signals)
├── models/                        # Downloaded model checkpoints
└── exp/                           # Experiment folders, weights & CSV results
```

---

## Pipeline Overview

The system follows a 4-stage pipeline:

### Stage 1: Representation Learning (`src/representation_learning/`)

Converts raw I/Q signals into visual representations and trains neural encoders:

- **Image generation** — Converts `.npy` signals to constellation diagrams or spectrograms
- **Autoencoder training** — Trains DINO ViT or ResNet autoencoders for unsupervised feature learning (`autoencoder_training.py`)
- **Classifier training** — Trains an `ImageClassifier` with a DNN head (`classifier_training.py`)
- **Centroid computation** — Computes per-class mean embeddings for nearest-centroid classification (`compute_centroids.py`)
- **FAISS indexing** — Builds kNN indices for FAISS-based predictions (`inference.py build_faiss`)
- **Inference** — Generates top-k predictions via DNN head, centroids, Random Forest, or FAISS kNN voting (`inference.py predict`)

**Alternative backbone:** The `src/denoMAE2/` submodule provides the DenoMAE 2.0 masked autoencoder (ViT-based), which combines image reconstruction with patch location classification for stronger feature representations. See: [arxiv.org/abs/2502.18202](https://arxiv.org/abs/2502.18202)

### Stage 2: Prompt Engineering (`src/prompt/`)

Transforms encoder outputs into structured text prompts for LLMs:

- **Feature extraction** — Computes statistical features (moments, cumulants, kurtosis) from raw signals (`data_processing.py`) or PCA-compressed encoder embeddings (`embedding_features.py`)
- **Discretization** — Per-feature KBins discretization with letter-encoded bins (A, B, C, …)
- **Prompt templates** — V1 (original) and V2 (source-aware with `SOURCE_CONTEXT` / `FEATURE_CONTEXT` blocks) (`templates.py`)
- **RAG** — Optional FAISS-based retrieval of few-shot examples from the training set (`rag.py`)
- **Dataset generation** — Builds train/test `.pkl` files containing prompts, labels, signal paths, and metadata (`generated_dataset.py`)

**Prediction sources** (centralized in `src/naming.py`):

| Source         | Method              | Description                                                           |
| -------------- | ------------------- | --------------------------------------------------------------------- |
| `dnn`          | DNN classifier head | Softmax probabilities from the trained classifier                     |
| `centroid`     | Nearest centroid    | Euclidean distance to per-class prototypes                            |
| `rf`           | Random Forest       | Trained on the encoder's feature space                                |
| `faiss`        | FAISS kNN voting    | k-nearest-neighbour voting in the embedding space                     |
| `faiss_filled` | FAISS kNN (filled)  | Same as `faiss` but pads to full top-k if fewer classes are voted for |

### Stage 3: LLM Evaluation (`src/evaluation/`)

Queries LLMs with constructed prompts and computes classification metrics:

- **Providers** — Gemini (`gemini_googleai.py`), OpenAI (`gpt_openai.py`), local Unsloth models (`unsloth_eval.py`)
- **Batched inference** — `unsloth_eval.py` supports batched generation with left-padding for efficiency
- **Metrics** — 1-pass accuracy, majority-vote accuracy, per-class clean accuracy (`utils.py`)
- **Experiment auditing** — `audit_experiments.py` scans `exp/` folders and regenerates a clean summary CSV

### Stage 4: QLoRA Finetuning (`src/finetuning/`, optional)

Adapts pre-trained LLMs to the classification task using quantized LoRA:

- **Dataset construction** — `dataset.py` converts train `.pkl` files into HF `Dataset` objects formatted as chat conversations for `SFTTrainer`
- **Training** — `train.py` runs QLoRA finetuning with Unsloth (4-bit quantized) and TRL's `SFTTrainer`
- **Evaluation** — Finetuned adapters are loaded via `unsloth_eval.py --adapter_path` for standard evaluation
- **Orchestration** — `run_finetuning.sh` automates the full workflow with versioned output directories

---

## Shell Scripts

| Script               | Purpose                                                                                                                                                                              |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `run_pipeline.sh`    | Single-run execution of the full pipeline (Steps 1–8) for one configuration                                                                                                          |
| `run_experiments.sh` | Batch grid runner — iterates over models × datasets × prediction sources × RAG × feature types. Supports `--resume` to skip completed experiments. Outputs `exp/results_summary.csv` |
| `run_finetuning.sh`  | QLoRA finetuning orchestration — data generation, training, and evaluation with auto-versioned output directories                                                                    |

---

## Naming Conventions

All output filenames are managed by `src/naming.py` through two key abstractions:

- **`PredictionSource`** — dataclass defining filenames for each prediction method (`raw_json`, `converted_json`, `pkl_tag`)
- **`ExperimentConfig`** — dataclass encoding all experiment dimensions (prediction source, OOD status, feature type, RAG, embedding components). The `build_tag()` method produces a compact filename tag from non-default dimensions.

Adding a new prediction source only requires registering it in the `SOURCES` dict; all downstream pipeline steps pick it up automatically.

---

## Supported Modulation Classes

The system supports 24 modulation schemes from the RadioML dataset:

```
128APSK, 128QAM, 16APSK, 16PSK, 16QAM, 256QAM, 32APSK, 32PSK, 32QAM,
4ASK, 64APSK, 64QAM, 8ASK, 8PSK, AM-DSB-SC, AM-DSB-WC, AM-SSB-SC,
AM-SSB-WC, BPSK, CPFSK, FM, GFSK, GMSK, OOK, OQPSK, QPSK
```

A 10-class subset is used for the custom dataset:
`4ASK, 4PAM, 8ASK, 16PAM, CPFSK, DQPSK, GFSK, GMSK, OQPSK, OOK`

---

## Data Layout

```
data/own/
├── unlabeled_10k/              # In-distribution dataset
│   ├── train/                  # Training images (per-class subdirectories)
│   │   ├── class_centers.json  # Computed centroids
│   │   └── faiss_knn/          # FAISS kNN index
│   └── test/                   # Test images
├── -11_-15dB/                  # OOD: SNR range −11 to −15 dB
│   └── test/
└── -30dB/                      # OOD: SNR −30 dB
    └── test/
```

---

## Key Libraries

- **[Unsloth](https://github.com/unslothai/unsloth)** — Memory-efficient LLM finetuning & inference (4-bit quantized)
- **[TRL](https://github.com/huggingface/trl)** — Transformer Reinforcement Learning (`SFTTrainer` for QLoRA)
- **[Hugging Face Datasets](https://github.com/huggingface/datasets)** — Dataset formatting
- **PyTorch / Torchvision** — Deep learning framework
- **scikit-learn** — PCA, KBinsDiscretizer, StandardScaler, ML baselines
- **[FAISS](https://github.com/facebookresearch/faiss)** — Vector similarity search for kNN predictions & RAG
- **Plotly** — Interactive visualizations & publication-quality charts

---

## Quick Start

```bash
# 1. Run the full pipeline for a single configuration
bash run_pipeline.sh

# 2. Run all experiment combinations
bash run_experiments.sh --resume

# 3. Run QLoRA finetuning
bash run_finetuning.sh

# 4. Generate result plots
python notebooks/plot_results.py
```

See `README.md` for detailed installation instructions, CLI arguments, and module reference.
