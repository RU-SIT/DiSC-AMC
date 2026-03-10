#!/usr/bin/env python3
"""
Audit all experiment folders, report completion status, and regenerate
results_summary_audit.csv from the response JSONs.

Usage:
    python audit_experiments.py          # from project root
"""

import os
import sys
import csv
import json
from pathlib import Path

# ── project imports (no GPU / unsloth needed) ───────────────────────────
from src.naming import ExperimentConfig, eval_result_name
from src.evaluation.utils import (
    load_existing_results,
    sort_results_by_prompt,
    get_unique_prompts,
    acc,
    clean_acc,
    pass_acc,
    majority_acc,
)

# ── same grid as run_experiments.sh ─────────────────────────────────────
MODELS = [
    "unsloth/DeepSeek-R1-Distill-Qwen-7B",
    "unsloth/GLM-4.6V-Flash",
]

# (DATASET_FOLDER, TRAIN_DATASET_FOLDER)
DATASETS = [
    ("unlabeled_10k", ""),
    ("-11_-15dB", "unlabeled_10k"),
    ("-30dB", "unlabeled_10k"),
]

PREDICTION_SOURCES = ["centroid", "faiss"]
RAG_OPTIONS = [("true", True), ("false", False)]
FEATURE_TYPES = ["stats", "embeddings"]

# ── fixed config (mirrors run_experiments.sh) ───────────────────────────
PROJECT_ROOT = Path("/mnt/d/Rowan/DiSC-AMC")
EXP_DIR = PROJECT_ROOT / "exp"
PROMPT_TYPE = "discret_prompts"
NOISE_MODE = "noisySignal"
N_BINS = 5
TOP_K = 5
NUM_TRIES = 1
N_COMPONENTS = 10
RAG_K = 10

CLASS_NAMES = [
    "4ASK", "4PAM", "8ASK", "16PAM", "CPFSK",
    "DQPSK", "GFSK", "GMSK", "OQPSK", "OOK",
]

RADIOML_CLASS_NAMES = [
    '128APSK', '128QAM', '16APSK', '16PSK', '16QAM', '256QAM',
    '32APSK', '32PSK', '32QAM', '4ASK', '64APSK', '64QAM',
    '8ASK', '8PSK', 'AM-DSB-SC', 'AM-DSB-WC', 'AM-SSB-SC', 'AM-SSB-WC',
    'BPSK', 'FM', 'GMSK', 'OOK', 'OQPSK', 'QPSK',
]

def get_class_names(dataset_type: str = 'own') -> list:
    """Return class names for the given dataset type."""
    if dataset_type == 'radioml':
        return RADIOML_CLASS_NAMES
    return CLASS_NAMES


def _shorten_model(model: str) -> str:
    m = model.replace("unsloth/", "")
    if "-unsloth" in m:
        m = m[: m.index("-unsloth")]
    return m


def _shorten_prompt_type(pt: str) -> str:
    return {
        "discret_prompts": "disc",
        "old_discret_prompts": "old_disc",
        "prompts": "cont",
        "old_prompts": "old_cont",
    }.get(pt, pt)


def _build_exp_dir_name(
    model: str,
    dataset_folder: str,
    prediction_source: str,
    feature_type: str,
    n_components: int,
    use_rag: bool,
    rag_k: int,
    ood_train_folder: str,
    backbone: str = "dino",
) -> str:
    model_short = _shorten_model(model)
    feat_tag = ""
    if prediction_source != "dnn":
        feat_tag += prediction_source
    if ood_train_folder:
        if feat_tag:
            feat_tag += "_"
        feat_tag += "ood"
    if backbone != "dino":
        if feat_tag:
            feat_tag += "_"
        feat_tag += backbone
    if feature_type == "embeddings" and n_components > 0:
        if feat_tag:
            feat_tag += "_"
        feat_tag += f"emb{n_components}"
    if use_rag and rag_k > 0:
        if feat_tag:
            feat_tag += "_"
        feat_tag += f"rag{rag_k}"

    prompt_short = _shorten_prompt_type(PROMPT_TYPE)
    base = dataset_folder
    if feat_tag:
        base += f"_{feat_tag}"
    base += f"_{prompt_short}_unsloth_{model_short}"
    return base


def main():
    csv_path = EXP_DIR / "results_summary_all.csv"
    header = [
        "Model",
        "DATASET_FOLDER",
        "TRAIN_DATASET_FOLDER",
        "PREDICTION_SOURCE",
        "USE_RAG",
        "FEATURE_TYPE",
        "1-pass",
        "1-majority",
        "acc",
        "clean-acc",
        "Number of unique prompts",
        "exp_folder",
    ]

    rows = []
    total = 0
    completed_count = 0
    incomplete_list = []
    missing_list = []

    for dataset_folder, train_dataset_folder in DATASETS:
        for prediction_source in PREDICTION_SOURCES:
            for rag_label, use_rag in RAG_OPTIONS:
                for feature_type in FEATURE_TYPES:
                    for model in MODELS:
                        total += 1

                        ood_train = ""
                        if train_dataset_folder and train_dataset_folder != dataset_folder:
                            ood_train = train_dataset_folder

                        n_comp = N_COMPONENTS if feature_type == "embeddings" else 0

                        base = _build_exp_dir_name(
                            model, dataset_folder, prediction_source,
                            feature_type, N_COMPONENTS, use_rag, RAG_K, ood_train,
                        )

                        # find latest version
                        v = 1
                        latest = None
                        while True:
                            folder = EXP_DIR / f"{base}_v{v:02d}"
                            if folder.is_dir():
                                latest = folder
                                v += 1
                            else:
                                break

                        label = (
                            f"{dataset_folder} | {_shorten_model(model)} | "
                            f"{prediction_source} | rag={rag_label} | {feature_type}"
                        )

                        if latest is None:
                            missing_list.append(label)
                            print(f"  ✗ MISSING  {label}")
                            continue

                        # Build ExperimentConfig to resolve the JSON filename
                        cfg = ExperimentConfig(
                            dataset_folder=dataset_folder,
                            prediction_source=prediction_source,
                            noise_mode=NOISE_MODE,
                            n_bins=N_BINS,
                            top_k=TOP_K,
                            feature_type=feature_type,
                            n_components=n_comp,
                            ood_train_folder=ood_train,
                            use_rag=use_rag,
                            rag_k=RAG_K if use_rag else 0,
                        )
                        clean_model = model.replace("unsloth/", "")
                        json_name = eval_result_name(cfg, PROMPT_TYPE, clean_model, "custom")
                        json_path = latest / json_name

                        if not json_path.exists():
                            missing_list.append(f"{label}  (folder exists, no JSON)")
                            print(f"  ✗ NO JSON  {label}")
                            continue

                        # Use existing resume logic to count completed prompts
                        results, prompts_done, completed_set = load_existing_results(
                            str(json_path), NUM_TRIES
                        )

                        n_completed = len(completed_set)
                        n_total_prompts = len(prompts_done)

                        if n_completed < 200:
                            incomplete_list.append(
                                f"{label}  →  {n_completed}/200 completed"
                            )
                            print(
                                f"  ⚠ INCOMPLETE  {label}  "
                                f"({n_completed}/{n_total_prompts} prompts done)"
                            )
                        else:
                            print(f"  ✓ COMPLETE   {label}  ({n_completed} prompts)")

                        # Compute metrics if any results
                        if results:
                            completed_count += 1
                            sorted_results = sort_results_by_prompt(results)
                            n_unique = len(get_unique_prompts(results))
                            pass_val = pass_acc(sorted_results)
                            maj_val = majority_acc(sorted_results)
                            acc_val = acc(sorted_results)
                            clean_val = clean_acc(sorted_results, class_names=CLASS_NAMES)

                            rows.append([
                                model,
                                dataset_folder,
                                train_dataset_folder,
                                prediction_source,
                                rag_label.upper() if rag_label in ("true", "false") else rag_label,
                                feature_type,
                                str(pass_val),
                                str(maj_val),
                                str(acc_val),
                                str(clean_val),
                                str(n_unique),
                                str(latest),
                            ])

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)

    # Summary
    print("\n" + "=" * 65)
    print(f"  Total experiments in grid : {total}")
    print(f"  Completed (with metrics)  : {completed_count}")
    print(f"  Incomplete                : {len(incomplete_list)}")
    print(f"  Missing (no folder/JSON)  : {len(missing_list)}")
    print(f"  CSV written to            : {csv_path}")
    print("=" * 65)

    if incomplete_list:
        print("\n── Incomplete experiments ──")
        for item in incomplete_list:
            print(f"  {item}")

    if missing_list:
        print("\n── Missing experiments ──")
        for item in missing_list:
            print(f"  {item}")


if __name__ == "__main__":
    main()
