"""
Generate LLM prompt datasets (.pkl) from raw signals and top-k predictions.

This is the main dataset builder for the DiSC-AMC pipeline.  It:

1.  Loads raw ``.npy`` signal files from train/test splits.
2.  Computes features — either statistical (moments, cumulants, kurtosis, …)
    or **encoder embeddings** (DINO / ResNet → PCA).
3.  Fits (train) or applies (test) a ``StandardScaler`` and ``KBinsDiscretizer``
    (and ``PCA`` for embedding mode).
4.  Generates four prompt variants per signal:
    - ``old_prompts`` / ``old_discret_prompts``  – basic template, all classes
    - ``prompts`` / ``discret_prompts``           – engineered template, top-k narrowed options
5.  Saves the result as a ``.pkl`` file consumed by the evaluation scripts.

Usage
-----
**Statistical features (default)**::

    cd src/prompt/

    # Step 1:  Build TRAIN data (fits scaler & discretizers, no top-k needed)
    python generated_dataset.py \\
        --mode train \\
        --dataset_folder unlabeled_10k \\
        --noise_mode noisySignal \\
        --n_bins 5

    # Step 2:  Build TEST data (reuses train scaler/discretizers + top-k)
    python generated_dataset.py \\
        --mode test \\
        --dataset_folder unlabeled_10k \\
        --noise_mode noisySignal \\
        --n_bins 5 \\
        --top_k 5

**Encoder embedding features**::

    # Step 1:  Build TRAIN data (fits PCA + scaler + discretizers)
    python generated_dataset.py \\
        --mode train \\
        --dataset_folder unlabeled_10k \\
        --noise_mode noisySignal \\
        --n_bins 5 \\
        --feature_type embeddings \\
        --encoder_weights ../../exp/dino_classifier.pth \\
        --backbone dino \\
        --n_components 10

    # Step 2:  Build TEST data (reuses PCA/scaler/discretizers + top-k)
    python generated_dataset.py \\
        --mode test \\
        --dataset_folder unlabeled_10k \\
        --noise_mode noisySignal \\
        --n_bins 5 \\
        --top_k 5 \\
        --feature_type embeddings \\
        --encoder_weights ../../exp/dino_classifier.pth \\
        --backbone dino \\
        --n_components 10

The test step expects two files to already exist:
  - The train ``.pkl`` produced by the train step.
  - The converted ``.npy``-keyed predictions JSON
    (produced by ``inference.py predict`` + ``convert_predictions.py``).

Filename patterns are defined centrally in :mod:`naming`.
"""
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
from glob import glob
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from naming import (
    VALID_SOURCES,
    converted_json_name,
    get_source,
    test_pkl_name,
    train_pkl_name,
)

from data_processing import (
    create_options,
    dict_to_np,
    discretize_features,
    generate_prompt,
    get_discrete_info,
    get_family_example,
    get_family_label,
    get_features,
    get_scaled_info,
    ktop_example,
    load_from_json,
    load_npy_file,
    reduce_example_dict,
    save_processed_data,
    split_real_imaginary,
)
from templates import (
    CASCADE_PROMPT,
    END_ENGINEERED_TEXT,
    INPUT_ENGINEERED_TEXT,
    INPUT_TEXT,
    MODULATION_FAMILIES,
    NEW_PROMPT_TEMPLATE,
    PROMPT_ENGINEERED_TEMPLATE,
    PROMPT_TEMPLATE,
    QUESTION_TEMPLATE,
)

# Embedding imports — lazy-loaded via _get_embedding_helpers() to avoid
# hard dependency on torch/torchvision when only using statistical features.

def get_dataset_label(signal_path):
    class_names = os.path.basename(signal_path).split('_')[0]
    return class_names

def get_dataset_snr(signal_path):
    snrs = os.path.basename(signal_path).split('_')[1].replace('db','').replace('dB','')
    return snrs

def get_dataset_random_example_paths(example_paths, classes, label, max_examples=10):
    new_example_paths = {}
    for class_name in classes:
            new_example_paths[f'{class_name}'] = [random.choice(example_paths[f'{class_name}'])]
    
    example_dict = reduce_example_dict(new_example_paths, label, max_examples)

    return example_dict

def create_dataset_example_paths(base_dir, noise_mode='noiselessSignal'):
    # example_paths = {
    #     'OOK': [f'{base_dir}/train/{noise_mode}/OOK_2.17dB__0379_20250627_150438.npy'],
    #     '4ASK': [f'{base_dir}/train/{noise_mode}/4ASK_6.58dB__0216_20250627_150737.npy'],
    #     '8ASK': [f'{base_dir}/train/{noise_mode}/8ASK_7.10dB__0265_20250627_151057.npy'],
    #     'OQPSK': [f'{base_dir}/train/{noise_mode}/OQPSK_6.17dB__0942_20250627_151545.npy'],
    #     'CPFSK': [f'{base_dir}/train/{noise_mode}/CPFSK_5.11dB__0829_20250627_151906.npy'],
    #     'GFSK': [f'{base_dir}/train/{noise_mode}/GFSK_3.57dB__0396_20250627_152112.npy'],
    #     '4PAM': [f'{base_dir}/train/{noise_mode}/4PAM_9.86dB__0954_20250627_152654.npy'],
    #     'DQPSK': [f'{base_dir}/train/{noise_mode}/DQPSK_8.61dB__0739_20250627_152953.npy'],
    #     '16PAM': [f'{base_dir}/train/{noise_mode}/16PAM_-9.62dB__0681_20250627_153144.npy'],
    #     'GMSK': [f'{base_dir}/train/{noise_mode}/GMSK_7.45dB__0839_20250627_155308.npy'],
    # }
    example_paths = {
        '4ASK': [f'{base_dir}/test/{noise_mode}/4ASK_-0.17dB__081_20250127_164342.npy'],
        '4PAM': [f'{base_dir}/test/{noise_mode}/4PAM_-0.00dB__031_20250127_164618.npy'],
        '8ASK': [f'{base_dir}/test/{noise_mode}/8ASK_-0.11dB__016_20250127_164352.npy'],
        '16PAM': [f'{base_dir}/test/{noise_mode}/16PAM_-0.08dB__058_20250127_145951.npy'],
        'CPFSK': [f'{base_dir}/test/{noise_mode}/CPFSK_-0.03dB__088_20250127_164523.npy'],
        'DQPSK': [f'{base_dir}/test/{noise_mode}/DQPSK_-0.01dB__036_20250127_164655.npy'],
        'GFSK': [f'{base_dir}/test/{noise_mode}/GFSK_-0.05dB__042_20250127_164545.npy'],
        'GMSK': [f'{base_dir}/test/{noise_mode}/GMSK_-0.12dB__059_20250127_164925.npy'],
        'OQPSK': [f'{base_dir}/test/{noise_mode}/OQPSK_-0.24dB__006_20250127_145655.npy'],
        'OOK': [f'{base_dir}/test/{noise_mode}/OOK_-0.17dB__091_20250127_164311.npy'],
    }

    # -30dB
    # example_paths = {
    #     '4ASK': [f'{base_dir}/train/{noise_mode}/4ASK_-30.01dB__050_20250924_121717.npy'],
    #     '4PAM': [f'{base_dir}/train/{noise_mode}/4PAM_-30.06dB__011_20250924_121815.npy'],
    #     '8ASK': [f'{base_dir}/train/{noise_mode}/8ASK_-30.44dB__090_20250924_121732.npy'],
    #     '16PAM': [f'{base_dir}/train/{noise_mode}/16PAM_-30.05dB__050_20250924_121843.npy'],
    #     'CPFSK': [f'{base_dir}/train/{noise_mode}/CPFSK_-30.65dB__077_20250924_121757.npy'],
    #     'DQPSK': [f'{base_dir}/train/{noise_mode}/DQPSK_-32.18dB__055_20250924_121835.npy'],
    #     'GFSK': [f'{base_dir}/train/{noise_mode}/GFSK_-31.48dB__008_20250924_121801.npy'],
    #     'GMSK': [f'{base_dir}/train/{noise_mode}/GMSK_-32.47dB__040_20250924_121917.npy'],
    #     'OQPSK': [f'{base_dir}/train/{noise_mode}/OQPSK_-33.82dB__095_20250924_121746.npy'],
    #     'OOK': [f'{base_dir}/train/{noise_mode}/OOK_-31.72dB__095_20250924_121710.npy'],
    # }
    
    return example_paths

def dataset_example_maker(base_dir="../../data/own/unlabeled_10k/", mode='train', noise_mode='noiselessSignal'):
    # Load signal paths
    pattern = f"{base_dir}/{mode}/{noise_mode}/*.npy"
    signal_paths = glob(pattern)
    
    label_list = [get_dataset_label(sig) for sig in signal_paths]
    snr_list = [get_dataset_snr(sig) for sig in signal_paths]

    unique_classes = list(set(label_list))
    unique_snrs = list(set(snr_list))

    # example_paths = load_from_json('../../data/RadioML/example_paths.json')
    example_paths = create_dataset_example_paths(base_dir, noise_mode=noise_mode)
    example_dict = get_dataset_random_example_paths(example_paths, unique_classes, unique_classes[0], max_examples=len(unique_classes))
    signal_paths = [signal_path for signal_path in signal_paths if signal_path not in example_paths.values()]

    return signal_paths, example_dict, label_list, snr_list

def get_processed_data(
    signal_paths: List[str], 
    signal_labels: List[str], 
    signal_snr: List[Union[int, float]], 
    feature_names: Optional[List[str]], 
    example_paths: List[str], 
    scaler: Optional[StandardScaler] = None,
    discretizers: Optional[Dict[int, KBinsDiscretizer]] = None,
    decimal_precision: int = 3, 
    add_context: bool = True,
    ktop_info: Optional[Dict[str, Any]] = None,
    n_bins: int = 5
) -> Dict[str, Any]:
    """
    Processes signal data from file paths to generate features, prompts, and metadata.

    This function loads signals, calculates statistical features, scales and discretizes 
    these features, and generates textual prompts (both continuous and discretized) 
    with optional few-shot context.

    Args:
        signal_paths (List[str]): List of file paths to the signal data (.npy files).
        signal_labels (List[str]): List of corresponding labels for each signal.
        signal_snr (List[Union[int, float]]): List of corresponding Signal-to-Noise Ratios.
        feature_names (Optional[List[str]]): List of feature names to calculate. 
                                             If None, a default list is used.
        scaler (Optional[StandardScaler], optional): Pre-fitted StandardScaler. If None, a new one 
                                                     is fitted to the data. Defaults to None.
        discretizers (Optional[Dict[int, KBinsDiscretizer]], optional):
            Dictionary mapping feature indices to pre-fitted KBinsDiscretizer objects.
            If None, new discretizers are created and fitted to the data. Defaults to None.
        decimal_precision (int, optional): Number of decimal places for formatting continuous features 
                                           in prompts. Defaults to 3.
        add_context (bool, optional): Whether to add few-shot examples to the prompts. Defaults to True.

    Returns:
        Dict[str, Any]: A dictionary containing the processed data:
            - 'signals': List of loaded signal data (np.ndarray).
            - 'stats': List of dictionaries containing scaled features for each signal.
            - 'discret_stats': List of dictionaries containing discretized features (bin indices).
            - 'labels': Original list of signal labels.
            - 'snrs': Original list of signal SNRs.
            - 'prompts': List of generated prompts using scaled continuous features.
            - 'discret_prompts': List of generated prompts using discretized features.
            - 'feature_names': List of feature names used.
            - 'scaler': The StandardScaler object used (fitted or provided).
            - 'discretizers': Dictionary of KBinsDiscretizer objects used for each feature index.
            - 'num_samples': Total number of signals processed.
            - 'num_features': Number of features extracted per signal (after flattening).
            - '#classes': Number of unique classes found in labels.
            - '#snr': Number of unique SNR values found.
    """
    # Define default feature names if none are provided
    if not feature_names:
        feature_names = ['snr','min', 'max', 'mean', 'variance', 'skewness', 'kurtosis', 
                         'moment_0', 'moment_1', 'moment_2', 'moment_3', 'moment_4', 
                         'moment_5', 'moment_6', 'moment_7', 'moment_8', 'moment_9',
                         'kstat_1', 'kstat_2', 'kstat_3', 'kstat_4',
                         'kstatvar_1', 'kstatvar_2']

    # Load signal data from the specified paths
    signals_data: List[np.ndarray] = [split_real_imaginary(load_npy_file(path)) for path in signal_paths]
    signals_snr: List[Union[int, float]] = [get_dataset_snr(path) for path in signal_paths]

    # Calculate statistical features for each loaded signal
    signal_summaries: List[Dict[str, Union[float, np.ndarray, int]]] = [
        get_features(sig, feature_names, snr=snr) 
        for sig, snr in tqdm(zip(signals_data, signals_snr), desc="Calculating features", total=len(signals_data))
    ]

    # Convert the feature dictionaries into NumPy arrays
    signal_features: List[np.ndarray] = [dict_to_np(sig, feature_names) for sig in tqdm(signal_summaries, desc="Converting features to array")]
    
    # Discretize the features using KBinsDiscretizer
    # Note: This fits discretizers based on the current batch of signal_features
    if discretizers is None:
        # If no discretizers are provided, create new ones and fit them to the feature data
        discretizers: Dict[int, KBinsDiscretizer] = {}
        _, discretizers = discretize_features(np.array(signal_features), n_bins=n_bins, strategy='uniform')
    # Apply the fitted discretizers to get the discrete representation for each signal summary
    signal_discretized_feature: List[Dict[str, Union[int, np.ndarray]]] = [get_discrete_info(sig, discretizers) for sig in tqdm(signal_summaries, desc="Discretizing features")]

    # Normalize features using standardization
    if scaler is None:
        # If no scaler is provided, create a new one and fit it to the feature data
        scaler = StandardScaler()
        _ = scaler.fit(signal_features) # Fit the scaler
    
    # Apply the scaler (either pre-fitted or newly fitted) to get scaled features
    signal_stats: List[Dict[str, Union[float, np.ndarray]]] = [get_scaled_info(sig, scaler) for sig in tqdm(signal_summaries, desc="Scaling features")]


    ######################## OLD PROMPTS ########################
    # Determine the unique options (classes) from the labels
    options: List[str] = list(set(signal_labels))

    # Generate the multiple-choice options string (e.g., "[A: opt1, B: opt2]").
    options_str: str = create_options(options)
    question_template = INPUT_TEXT
    instruction_template = PROMPT_TEMPLATE
    question_template_format: List[str] = [options_str]
    instruction_template_format: List[str] = []
    all_example_dict = {
        k: [(split_real_imaginary(load_npy_file(p)), get_dataset_snr(p)) for p in v]
        for k, v in example_paths.items()
    }

    # print(all_example_dict)
    # Generate prompts using the scaled continuous features
    # These prompts might include few-shot examples if add_context is True
    old_context_prompts: List[str] = [
        generate_prompt(
            sig_info, question_template, question_template_format, instruction_template, instruction_template_format, feature_names, 
            processed=True, add_context=add_context, example_dict=reduce_example_dict(all_example_dict, get_dataset_label(signal_paths[i]), max_examples=10),
            decimal_precision=decimal_precision, options=options, 
            discretizers=None, scaler=scaler, discretized=False
        ) for i, sig_info in tqdm(enumerate(signal_stats), desc="Generating continuous prompts")
    ]
    
    # Generate prompts using the discretized features (represented by letters)
    # These prompts might also include few-shot examples if add_context is True
    old_discret_context_prompts: List[str] = [
        generate_prompt(
            sig_info, question_template, question_template_format, instruction_template, instruction_template_format, feature_names, 
            processed=True, add_context=add_context, example_dict=reduce_example_dict(all_example_dict, get_dataset_label(signal_paths[i]), max_examples=10), 
            decimal_precision=decimal_precision, options=options, 
            discretizers=discretizers, scaler=scaler, discretized=True
        ) for i, sig_info in tqdm(enumerate(signal_discretized_feature), desc="Generating discrete prompts")
    ]


    ######################## K-TOP PROMPTS ########################
    context_prompts, discret_context_prompts = [], []
    if ktop_info is not None:
        question_template = INPUT_ENGINEERED_TEXT
        instruction_template = PROMPT_ENGINEERED_TEMPLATE
        # Generate prompts using the scaled continuous features
        # These prompts might include few-shot examples if add_context is True
        context_prompts: List[str] = [
            generate_prompt(
                sig_info, question_template, [ktop_info[i]], instruction_template, instruction_template_format, feature_names, 
                processed=True, add_context=add_context, example_dict=ktop_example(ktop_info[i], example_dict=all_example_dict),
                decimal_precision=decimal_precision, options=ktop_info[i], 
                discretizers=None, scaler=scaler, discretized=False
            ) + END_ENGINEERED_TEXT
            for i, sig_info in tqdm(enumerate(signal_stats), desc="Generating continuous prompts")
        ]
        
        # Generate prompts using the discretized features (represented by letters)
        # These prompts might also include few-shot examples if add_context is True
        discret_context_prompts: List[str] = [
            generate_prompt(
                sig_info, question_template, [ktop_info[i]], instruction_template, instruction_template_format, feature_names, 
                processed=True, add_context=add_context, example_dict=ktop_example(ktop_info[i], example_dict=all_example_dict),
                decimal_precision=decimal_precision, options=ktop_info[i], 
                discretizers=discretizers, scaler=scaler, discretized=True
            ) + END_ENGINEERED_TEXT
            for i, sig_info in tqdm(enumerate(signal_discretized_feature), desc="Generating discrete prompts")
        ]


    # ##TODO: For Cascade Prompting
    # ######################## NEW PROMPTS ########################
    # # Determine the unique options (classes) from the labels
    # options: List[str] = list(MODULATION_FAMILIES.keys())

    # # Generate the multiple-choice options string (e.g., "[A: opt1, B: opt2]").
    # options_str: str = create_options(options)

    # all_example_paths = get_family_example(MODULATION_FAMILIES, example_paths)

    # question_template = QUESTION_TEMPLATE
    # instruction_template = CASCADE_PROMPT
    # question_template_format: List[str] = []
    # instruction_template_format: List[str] = [str(len(MODULATION_FAMILIES.keys())), 'Wireless', str(MODULATION_FAMILIES).replace("'", ''), options_str]
    # all_example_dict = {k:[load_npy_file(p) for p in v] for k,v in all_example_paths.items()}

    # # Generate prompts using the scaled continuous features
    # # These prompts might include few-shot examples if add_context is True
    # context_prompts: List[str] = [
    #     generate_prompt(
    #         sig_info, question_template, question_template_format, instruction_template, instruction_template_format, feature_names, 
    #         processed=True, add_context=add_context, example_dict=reduce_example_dict(all_example_dict, get_family_label(get_dataset_label(signal_paths[i]), MODULATION_FAMILIES), max_examples=2*len(options)),
    #         decimal_precision=decimal_precision, options=options, 
    #         discretizers=None, scaler=scaler, discretized=False
    #     ) for i, sig_info in tqdm(enumerate(signal_stats), desc="Generating continuous prompts")
    # ]
    
    # # Generate prompts using the discretized features (represented by letters)
    # # These prompts might also include few-shot examples if add_context is True
    # discret_context_prompts: List[str] = [
    #     generate_prompt(
    #         sig_info, question_template, question_template_format, instruction_template, instruction_template_format, feature_names, 
    #         processed=True, add_context=add_context, example_dict=reduce_example_dict(all_example_dict, get_family_label(get_dataset_label(signal_paths[i]), MODULATION_FAMILIES), max_examples=2*len(options)), 
    #         decimal_precision=decimal_precision, options=options, 
    #         discretizers=discretizers, scaler=scaler, discretized=True
    #     ) for i, sig_info in tqdm(enumerate(signal_discretized_feature), desc="Generating discrete prompts")
    # ]

    # Compile all processed data and metadata into a dictionary
    data: Dict[str, Any] = {
        'signal_paths': [os.path.abspath(p) for p in signal_paths],               # Original signal paths
        'signals': signals_data,                     # Raw signal data
        'stats': signal_stats,                       # Scaled continuous features (list of dicts)
        'discret_stats': signal_discretized_feature, # Discretized features (list of dicts)
        'labels': signal_labels,                     # Original labels
        'snrs': signal_snr,                          # Original SNRs
        'prompts': context_prompts,                  # Prompts with continuous features
        'discret_prompts': discret_context_prompts,  # Prompts with discrete features
        'old_prompts': old_context_prompts,                  # Prompts with continuous features
        'old_discret_prompts': old_discret_context_prompts,  # Prompts with discrete features
        'feature_names': feature_names,              # List of feature names used
        'scaler': scaler,                            # Scaler object used
        'discretizers': discretizers,                # Discretizer objects used
        'num_samples': len(signal_labels),           # Total number of samples
        'num_features': signal_features[0].shape[0], # Number of features per sample
        '#classes': len(options),                    # Number of unique classes
        '#snr': len(set(signal_snr)),              # Number of unique SNR values,
        'k-top': ktop_info,              # Number of unique SNRs
    }
    
    # Return the dictionary containing all processed information
    return data

DEFAULT_FEATURE_NAMES: List[str] = [
    "snr", "skewness", "kurtosis",
    "moment_0", "moment_1", "moment_2", "moment_3", "moment_4",
    "moment_5", "moment_6", "moment_7", "moment_8", "moment_9",
    "kstat_1", "kstat_2", "kstat_3", "kstat_4",
    "kstatvar_1", "kstatvar_2",
]


# ── Embedding-based feature pipeline ────────────────────────────────────────

def _feature_tag(n_components: int) -> str:
    """Build a short tag for embedding pkl filenames, e.g. ``'emb10'``."""
    return f"emb{n_components}"


def get_embedding_processed_data(
    signal_paths: List[str],
    signal_labels: List[str],
    signal_snr: List[Union[int, float]],
    noise_mode: str,
    encoder,                       # nn.Module — not typed to avoid top-level torch import
    device,                        # torch.device
    example_paths: Dict[str, List[str]],
    n_components: int = 10,
    n_bins: int = 5,
    pca=None,
    discretizers=None,
    scaler=None,
    batch_size: int = 32,
    decimal_precision: int = 3,
    add_context: bool = True,
    ktop_info: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Process signals using encoder embeddings instead of statistical features.

    This is the embedding counterpart of :func:`get_processed_data`.  Instead
    of computing moments/cumulants, it:

    1. Maps ``.npy`` signal paths to ``.png`` constellation images.
    2. Extracts encoder embeddings (DINO / ResNet).
    3. PCA-reduces to *n_components* dimensions.
    4. Discretizes and scales the PCA components.
    5. Generates the same four prompt variants as the statistical pipeline.

    Parameters
    ----------
    signal_paths
        Paths to ``.npy`` signal files.
    noise_mode
        ``"noisySignal"`` or ``"noiselessSignal"`` — determines which image
        directory to use.
    encoder, device
        Pre-loaded encoder module and torch device.
    example_paths
        ``{class: [path, …]}`` for few-shot prompt context.
    n_components
        Number of PCA components to keep.
    pca, discretizers, scaler
        Pre-fitted transformers for test mode.  Pass ``None`` for train.

    Returns
    -------
    dict
        Same structure as :func:`get_processed_data`, plus ``'pca'`` and
        ``'feature_type'`` keys.
    """
    from embedding_features import (
        compute_embedding_features,
        extract_embeddings_from_paths,
        prepare_example_embedding_dicts,
        signal_path_to_image_path,
    )

    # ── 1. Load raw signals (for compatibility with downstream pkl readers) ──
    signals_data = [
        split_real_imaginary(load_npy_file(p)) for p in signal_paths
    ]

    # ── 2. Map to images & extract embeddings ────────────────────────────
    image_paths = [
        signal_path_to_image_path(p, noise_mode) for p in signal_paths
    ]
    embeddings = extract_embeddings_from_paths(
        encoder, device, image_paths, batch_size,
    )

    # ── 3. PCA → discretize → scale ─────────────────────────────────────
    (
        feature_dicts, discretized_dicts, scaled_dicts,
        feature_names, pca, discretizers, scaler,
    ) = compute_embedding_features(
        embeddings, n_components, n_bins,
        pca=pca, discretizers=discretizers, scaler=scaler,
    )

    # ── 4. Pre-process few-shot examples ─────────────────────────────────
    scaled_ex, discret_ex = prepare_example_embedding_dicts(
        encoder, device, example_paths, noise_mode,
        n_components, pca, discretizers, scaler, batch_size,
    )

    # ── 5. OLD PROMPTS ───────────────────────────────────────────────────
    options: List[str] = list(set(signal_labels))
    options_str: str = create_options(options)

    old_context_prompts: List[str] = [
        generate_prompt(
            sig_info, INPUT_TEXT, [options_str],
            PROMPT_TEMPLATE, [], feature_names,
            processed=True, add_context=add_context,
            example_dict=reduce_example_dict(
                scaled_ex, get_dataset_label(signal_paths[i]), max_examples=10,
            ),
            decimal_precision=decimal_precision, options=options,
            discretizers=None, scaler=scaler, discretized=False,
            examples_processed=True,
        )
        for i, sig_info in tqdm(
            enumerate(scaled_dicts), desc="Generating old continuous prompts",
        )
    ]

    old_discret_context_prompts: List[str] = [
        generate_prompt(
            sig_info, INPUT_TEXT, [options_str],
            PROMPT_TEMPLATE, [], feature_names,
            processed=True, add_context=add_context,
            example_dict=reduce_example_dict(
                discret_ex, get_dataset_label(signal_paths[i]), max_examples=10,
            ),
            decimal_precision=decimal_precision, options=options,
            discretizers=discretizers, scaler=scaler, discretized=True,
            examples_processed=True,
        )
        for i, sig_info in tqdm(
            enumerate(discretized_dicts), desc="Generating old discrete prompts",
        )
    ]

    # ── 6. K-TOP PROMPTS ─────────────────────────────────────────────────
    context_prompts: List[str] = []
    discret_context_prompts: List[str] = []
    if ktop_info is not None:
        context_prompts = [
            generate_prompt(
                sig_info, INPUT_ENGINEERED_TEXT, [ktop_info[i]],
                PROMPT_ENGINEERED_TEMPLATE, [], feature_names,
                processed=True, add_context=add_context,
                example_dict=ktop_example(ktop_info[i], example_dict=scaled_ex),
                decimal_precision=decimal_precision, options=ktop_info[i],
                discretizers=None, scaler=scaler, discretized=False,
                examples_processed=True,
            ) + END_ENGINEERED_TEXT
            for i, sig_info in tqdm(
                enumerate(scaled_dicts), desc="Generating k-top continuous prompts",
            )
        ]

        discret_context_prompts = [
            generate_prompt(
                sig_info, INPUT_ENGINEERED_TEXT, [ktop_info[i]],
                PROMPT_ENGINEERED_TEMPLATE, [], feature_names,
                processed=True, add_context=add_context,
                example_dict=ktop_example(ktop_info[i], example_dict=discret_ex),
                decimal_precision=decimal_precision, options=ktop_info[i],
                discretizers=discretizers, scaler=scaler, discretized=True,
                examples_processed=True,
            ) + END_ENGINEERED_TEXT
            for i, sig_info in tqdm(
                enumerate(discretized_dicts), desc="Generating k-top discrete prompts",
            )
        ]

    # ── 7. Build output dict ─────────────────────────────────────────────

    data: Dict[str, Any] = {
        "signal_paths": [os.path.abspath(p) for p in signal_paths],
        "signals": signals_data,
        "stats": scaled_dicts,
        "discret_stats": discretized_dicts,
        "labels": signal_labels,
        "snrs": signal_snr,
        "prompts": context_prompts,
        "discret_prompts": discret_context_prompts,
        "old_prompts": old_context_prompts,
        "old_discret_prompts": old_discret_context_prompts,
        "feature_names": feature_names,
        "scaler": scaler,
        "discretizers": discretizers,
        "pca": pca,
        "num_samples": len(signal_labels),
        "num_features": n_components,
        "#classes": len(options),
        "#snr": len(set(signal_snr)),
        "k-top": ktop_info,
        "feature_type": "embeddings",
    }
    return data


def build_train(
    data_root: str,
    dataset_folder: str,
    noise_mode: str,
    n_bins: int,
    top_k: int,
    prediction_source: str = "centroid",
    feature_names: Optional[List[str]] = None,
    feature_type: str = "stats",
    encoder_weights: Optional[str] = None,
    backbone: str = "dino",
    n_components: int = 10,
    batch_size: int = 32,
) -> None:
    """Build and save the TRAIN dataset ``.pkl``.

    Fits a new ``StandardScaler`` and ``KBinsDiscretizer`` on the training
    signals.  Only basic (old) prompts are generated because top-k
    predictions are not available for the training set.

    When ``feature_type="embeddings"``, the encoder is loaded and PCA is
    fit on the training embeddings.  The resulting ``.pkl`` includes a
    ``'pca'`` key.
    """
    base_dir = os.path.join(data_root, dataset_folder)
    feat_tag = _feature_tag(n_components) if feature_type == "embeddings" else ""

    # Gather signal paths & metadata
    train_signal_paths, example_paths, train_labels, train_snrs = dataset_example_maker(
        base_dir=base_dir + "/",
        mode="train",
        noise_mode=noise_mode,
    )

    print(f"Train signals: {len(train_signal_paths)} | Classes: {len(set(train_labels))}")

    if feature_type == "embeddings":
        from embedding_features import load_encoder_for_embeddings

        if encoder_weights is None:
            raise ValueError("--encoder_weights is required for feature_type=embeddings")

        encoder, device = load_encoder_for_embeddings(backbone, encoder_weights)
        print(f"Encoder ({backbone}) loaded on {device}")

        train_data = get_embedding_processed_data(
            signal_paths=train_signal_paths,
            signal_labels=train_labels,
            signal_snr=train_snrs,
            noise_mode=noise_mode,
            encoder=encoder,
            device=device,
            example_paths=example_paths,
            n_components=n_components,
            n_bins=n_bins,
            batch_size=batch_size,
            decimal_precision=3,
            add_context=True,
        )
    else:
        feature_names = feature_names or DEFAULT_FEATURE_NAMES
        train_data = get_processed_data(
            signal_paths=train_signal_paths,
            signal_labels=train_labels,
            signal_snr=train_snrs,
            feature_names=feature_names,
            example_paths=example_paths,
            scaler=None,
            discretizers=None,
            decimal_precision=3,
            add_context=True,
            n_bins=n_bins,
        )

    out_path = os.path.join(
        base_dir,
        train_pkl_name(prediction_source, noise_mode, n_bins, top_k, feature_tag=feat_tag),
    )
    save_processed_data(train_data, out_path)
    print(f"Train data saved → {out_path}")


def build_test(
    data_root: str,
    dataset_folder: str,
    noise_mode: str,
    n_bins: int,
    top_k: int,
    prediction_source: str = "centroid",
    feature_names: Optional[List[str]] = None,
    feature_type: str = "stats",
    encoder_weights: Optional[str] = None,
    backbone: str = "dino",
    n_components: int = 10,
    batch_size: int = 32,
    train_dataset_folder: Optional[str] = None,
) -> None:
    """Build and save the TEST dataset ``.pkl``.

    Reuses the scaler and discretizers from the previously saved train
    ``.pkl``.  Loads the top-k predictions JSON to generate the engineered
    prompts with narrowed classification options.

    When ``train_dataset_folder`` is set, the train ``.pkl`` and few-shot
    examples are loaded from that folder instead of ``dataset_folder``.
    This enables **out-of-distribution (OOD)** experiments where
    training statistics come from one dataset (e.g. ``unlabeled_10k``)
    and test signals come from another (e.g. ``-11_-15dB``).

    When ``feature_type="embeddings"``, the PCA transformer is also reused
    from the train pkl.
    """
    base_dir = os.path.join(data_root, dataset_folder)
    train_folder = train_dataset_folder or dataset_folder
    train_base_dir = os.path.join(data_root, train_folder)
    feat_tag = _feature_tag(n_components) if feature_type == "embeddings" else ""

    if train_folder != dataset_folder:
        print(f"OOD mode: train data from '{train_folder}', test data from '{dataset_folder}'")

    # ── Load train data (for scaler, discretizers, and optionally PCA) ───
    train_pkl = os.path.join(
        train_base_dir,
        train_pkl_name(prediction_source, noise_mode, n_bins, top_k, feature_tag=feat_tag),
    )
    if not os.path.isfile(train_pkl):
        raise FileNotFoundError(
            f"Train data not found: {train_pkl}\n"
            "Run with --mode train on the train dataset first."
        )
    with open(train_pkl, "rb") as f:
        train_data = pickle.load(f)
    print(f"Loaded train transformers from {train_pkl}")

    # ── Load example paths from train dataset ────────────────────────────
    _, example_paths, _, _ = dataset_example_maker(
        base_dir=train_base_dir + "/",
        mode="train",
        noise_mode=noise_mode,
    )

    # ── Gather test signal paths ─────────────────────────────────────────
    test_signal_paths = glob(
        os.path.join(base_dir, "test", noise_mode, "*.npy")
    )
    if not test_signal_paths:
        raise FileNotFoundError(
            f"No .npy files in {os.path.join(base_dir, 'test', noise_mode)}"
        )
    test_labels = [get_dataset_label(p) for p in test_signal_paths]
    test_snrs = [get_dataset_snr(p) for p in test_signal_paths]
    print(f"Test signals: {len(test_signal_paths)} | Classes: {len(set(test_labels))}")

    # ── Load top-k predictions (optional for OOD) ──────────────────────
    ktop_json = os.path.join(base_dir, converted_json_name(prediction_source, top_k))
    ktop_info = None
    if os.path.isfile(ktop_json):
        ktop_raw = load_from_json(ktop_json)
        print(f"Loaded top-{top_k} predictions from {ktop_json}")
        # Map signal_path → top-k class list
        img_key = "noiseLessImg" if noise_mode == "noiselessSignal" else "noisyImg"
        ktop_info = [
            ktop_raw[os.path.basename(p)][img_key]
            for p in test_signal_paths
        ]
    else:
        print(f"Top-k predictions not found at {ktop_json}")
        print("  → Generating old-style prompts only (all classes as options)")

    # ── Process ──────────────────────────────────────────────────────────
    if feature_type == "embeddings":
        from embedding_features import load_encoder_for_embeddings

        if encoder_weights is None:
            raise ValueError("--encoder_weights is required for feature_type=embeddings")

        encoder, device = load_encoder_for_embeddings(backbone, encoder_weights)
        print(f"Encoder ({backbone}) loaded on {device}")

        test_data = get_embedding_processed_data(
            signal_paths=test_signal_paths,
            signal_labels=test_labels,
            signal_snr=test_snrs,
            noise_mode=noise_mode,
            encoder=encoder,
            device=device,
            example_paths=example_paths,
            n_components=n_components,
            n_bins=n_bins,
            pca=train_data["pca"],
            discretizers=train_data["discretizers"],
            scaler=train_data["scaler"],
            batch_size=batch_size,
            decimal_precision=3,
            add_context=True,
            ktop_info=ktop_info,
        )
    else:
        feature_names = feature_names or DEFAULT_FEATURE_NAMES
        test_data = get_processed_data(
            signal_paths=test_signal_paths,
            signal_labels=test_labels,
            signal_snr=test_snrs,
            feature_names=feature_names,
            example_paths=example_paths,
            scaler=train_data["scaler"],
            discretizers=train_data["discretizers"],
            decimal_precision=3,
            add_context=True,
            ktop_info=ktop_info,
            n_bins=n_bins,
        )

    out_path = os.path.join(
        base_dir,
        test_pkl_name(prediction_source, noise_mode, n_bins, top_k, feature_tag=feat_tag),
    )
    save_processed_data(test_data, out_path)
    print(f"Test data saved → {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def _build_epilog() -> str:
    """Generate the ``--help`` epilog dynamically from :data:`naming.SOURCES`."""
    from naming import SOURCES

    lines = ["Prediction sources & output naming:"]
    for src in SOURCES.values():
        test_pkl = test_pkl_name(src.key, "noisySignal", 5, 5)
        lines.append(
            f"  --prediction_source {src.key:<10s} → reads {src.converted_json.format(topk=5)}"
        )
        lines.append(f"  {'':<30s}   writes {test_pkl}")
    lines.append("")
    lines.append("Examples:")
    lines.append("  python generated_dataset.py --mode train \\")
    lines.append("      --dataset_folder unlabeled_10k --noise_mode noisySignal --n_bins 5 --top_k 5")
    lines.append("")
    lines.append("  python generated_dataset.py --mode test \\")
    lines.append("      --dataset_folder unlabeled_10k --noise_mode noisySignal --n_bins 5 --top_k 5 \\")
    lines.append("      --prediction_source centroid")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate LLM prompt datasets (.pkl) for DiSC-AMC.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_build_epilog(),
    )
    parser.add_argument(
        "--mode", type=str, required=True, choices=["train", "test"],
        help="Which split to build: 'train' or 'test'.",
    )
    parser.add_argument(
        "--dataset_folder", type=str, default="unlabeled_10k",
        help="Name of the dataset folder under data_root (default: unlabeled_10k).",
    )
    parser.add_argument(
        "--noise_mode", type=str, default="noisySignal",
        choices=["noisySignal", "noiselessSignal"],
        help="Signal variant to use (default: noisySignal).",
    )
    parser.add_argument(
        "--n_bins", type=int, default=5,
        help="Number of KBinsDiscretizer bins (default: 5).",
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Top-k value matching the predictions file (default: 5).",
    )
    parser.add_argument(
        "--prediction_source", type=str, default="centroid",
        choices=VALID_SOURCES,
        help="Which top-k prediction source to use (default: centroid).",
    )
    parser.add_argument(
        "--data_root", type=str, default="../../data/own",
        help="Root data directory (default: ../../data/own).",
    )
    parser.add_argument(
        "--train_dataset_folder", type=str, default=None,
        help="If set, load the train .pkl (scaler, discretizers, examples) "
             "from this folder instead of --dataset_folder. Enables OOD "
             "experiments (e.g. train on unlabeled_10k, test on -11_-15dB).",
    )
    parser.add_argument(
        "--feature_type", type=str, default="stats",
        choices=["stats", "embeddings"],
        help="Feature source: 'stats' (moments/cumulants) or 'embeddings' "
             "(encoder → PCA → discretize).  Default: stats.",
    )
    parser.add_argument(
        "--encoder_weights", type=str, default=None,
        help="Path to encoder checkpoint (.pth).  Required when "
             "--feature_type=embeddings.",
    )
    parser.add_argument(
        "--backbone", type=str, default="dino",
        choices=["dino", "resnet"],
        help="Encoder backbone type (default: dino).  Only used when "
             "--feature_type=embeddings.",
    )
    parser.add_argument(
        "--n_components", type=int, default=10,
        help="Number of PCA components (default: 10).  Only used when "
             "--feature_type=embeddings.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for embedding extraction (default: 32).  Only "
             "used when --feature_type=embeddings.",
    )

    args = parser.parse_args()

    if args.mode == "train":
        build_train(
            data_root=args.data_root,
            dataset_folder=args.dataset_folder,
            noise_mode=args.noise_mode,
            n_bins=args.n_bins,
            top_k=args.top_k,
            prediction_source=args.prediction_source,
            feature_type=args.feature_type,
            encoder_weights=args.encoder_weights,
            backbone=args.backbone,
            n_components=args.n_components,
            batch_size=args.batch_size,
        )
    elif args.mode == "test":
        build_test(
            data_root=args.data_root,
            dataset_folder=args.dataset_folder,
            noise_mode=args.noise_mode,
            n_bins=args.n_bins,
            top_k=args.top_k,
            prediction_source=args.prediction_source,
            feature_type=args.feature_type,
            encoder_weights=args.encoder_weights,
            backbone=args.backbone,
            n_components=args.n_components,
            batch_size=args.batch_size,
            train_dataset_folder=args.train_dataset_folder,
        )
