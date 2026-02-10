# -*- coding: utf-8 -*-
# %%
import os
import random
from glob import glob

from sre_constants import IN
from typing import List, Dict, Any, Optional, Union
import numpy as np
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from tqdm import tqdm
from sklearn.preprocessing import KBinsDiscretizer
from data_processing import (
    load_npy_file, get_features, dict_to_np, get_discrete_info, get_scaled_info,
    discretize_features, create_options, generate_prompt, save_processed_data, reduce_example_dict, get_family_example,
    get_family_label
    )

from templates import (QUESTION_TEMPLATE, INPUT_TEXT, PROMPT_TEMPLATE, CASCADE_PROMPT, MODULATION_FAMILIES, AMPLITUDE_FAMILY, ANANLOG_FAMILY, QAM_FAMILY, PHASE_FAMILY, FREQUENCY_FAMILY)


def get_radioml_label(signal_path):
    class_names = os.path.dirname(signal_path).split(os.sep)[-1]
    return class_names

def get_radioml_snr(signal_path):
    snrs = os.path.dirname(signal_path).split(os.sep)[-3].replace('snr_','').replace('db','')
    return snrs

def get_radionml_random_example_paths(example_paths, snrs, classes, label, max_examples=10):
    new_example_paths = {}
    for class_name in classes:
            snr=random.choice(snrs)
            new_example_paths[f'{class_name}'] = example_paths[f'{class_name} with {snr}db SNR']
    
    example_dict = reduce_example_dict(new_example_paths, label, max_examples)

    return example_dict

def create_radioml_example_paths(signal_paths, snrs, classes):
    example_paths = {}
    for snr in snrs:
        for class_name in classes:
            acceptable_paths = [path for path in signal_paths if f'snr_{snr}db' in path and class_name in path]
            example_paths[f'{class_name} with {snr}db SNR'] = [random.choice(acceptable_paths)]
    
    return example_paths

def radioml_example_maker(pattern="../../data/RadioML/*/train/*/*.npy"):
    # Load signal paths
    signal_paths = glob(pattern)
    
    label_list = [get_radioml_label(sig) for sig in signal_paths]
    snr_list = [get_radioml_snr(sig) for sig in signal_paths]

    unique_classes = list(set(label_list))
    unique_snrs = list(set(snr_list))

    # example_paths = load_from_json('../../data/RadioML/example_paths.json')
    example_paths = create_radioml_example_paths(signal_paths, unique_snrs, unique_classes)
    example_dict = get_radionml_random_example_paths(example_paths, snr_list, unique_classes, unique_classes[0], max_examples=len(unique_classes))
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
    add_context: bool = True
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
        feature_names = ['min', 'max', 'mean', 'variance', 'skewness', 'kurtosis', 
                         'moment_0', 'moment_1', 'moment_2', 'moment_3', 'moment_4', 
                         'moment_5', 'moment_6', 'moment_7', 'moment_8', 'moment_9',
                         'kstat_1', 'kstat_2', 'kstat_3', 'kstat_4',
                         'kstatvar_1', 'kstatvar_2']

    # Load signal data from the specified paths
    signals_data: List[np.ndarray] = [load_npy_file(path) for path in signal_paths]
    # Calculate statistical features for each loaded signal
    signal_summaries: List[Dict[str, Union[float, np.ndarray, int]]] = [get_features(sig, feature_names) for sig in tqdm(signals_data, desc="Calculating features")]
    # Convert the feature dictionaries into NumPy arrays
    signal_features: List[np.ndarray] = [dict_to_np(sig, feature_names) for sig in tqdm(signal_summaries, desc="Converting features to array")]
    
    # Discretize the features using KBinsDiscretizer
    # Note: This fits discretizers based on the current batch of signal_features
    if discretizers is None:
        # If no discretizers are provided, create new ones and fit them to the feature data
        discretizers: Dict[int, KBinsDiscretizer] = {}
        _, discretizers = discretize_features(np.array(signal_features), n_bins=10, strategy='uniform')
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
    all_example_dict = {k:[load_npy_file(p) for p in v] for k,v in example_paths.items()}

    # Generate prompts using the scaled continuous features
    # These prompts might include few-shot examples if add_context is True
    old_context_prompts: List[str] = [
        generate_prompt(
            sig_info, question_template, question_template_format, instruction_template, instruction_template_format, feature_names, 
            processed=True, add_context=add_context, example_dict=reduce_example_dict(all_example_dict, get_radioml_label(signal_paths[i]), max_examples=10),
            decimal_precision=decimal_precision, options=options, 
            discretizers=None, scaler=scaler, discretized=False
        ) for i, sig_info in tqdm(enumerate(signal_stats), desc="Generating continuous prompts")
    ]
    
    # Generate prompts using the discretized features (represented by letters)
    # These prompts might also include few-shot examples if add_context is True
    old_discret_context_prompts: List[str] = [
        generate_prompt(
            sig_info, question_template, question_template_format, instruction_template, instruction_template_format, feature_names, 
            processed=True, add_context=add_context, example_dict=reduce_example_dict(all_example_dict, get_radioml_label(signal_paths[i]), max_examples=10), 
            decimal_precision=decimal_precision, options=options, 
            discretizers=discretizers, scaler=scaler, discretized=True
        ) for i, sig_info in tqdm(enumerate(signal_discretized_feature), desc="Generating discrete prompts")
    ]

    ######################## NEW PROMPTS ########################
    # Determine the unique options (classes) from the labels
    options: List[str] = list(MODULATION_FAMILIES.keys())

    # Generate the multiple-choice options string (e.g., "[A: opt1, B: opt2]").
    options_str: str = create_options(options)

    all_example_paths = get_family_example(MODULATION_FAMILIES, example_paths)

    question_template = QUESTION_TEMPLATE
    instruction_template = CASCADE_PROMPT
    question_template_format: List[str] = []
    instruction_template_format: List[str] = [str(len(MODULATION_FAMILIES.keys())), 'Wireless', str(MODULATION_FAMILIES).replace("'", ''), options_str]
    all_example_dict = {k:[load_npy_file(p) for p in v] for k,v in all_example_paths.items()}

    # Generate prompts using the scaled continuous features
    # These prompts might include few-shot examples if add_context is True
    context_prompts: List[str] = [
        generate_prompt(
            sig_info, question_template, question_template_format, instruction_template, instruction_template_format, feature_names, 
            processed=True, add_context=add_context, example_dict=reduce_example_dict(all_example_dict, get_family_label(get_radioml_label(signal_paths[i]), MODULATION_FAMILIES), max_examples=2*len(options)),
            decimal_precision=decimal_precision, options=options, 
            discretizers=None, scaler=scaler, discretized=False
        ) for i, sig_info in tqdm(enumerate(signal_stats), desc="Generating continuous prompts")
    ]
    
    # Generate prompts using the discretized features (represented by letters)
    # These prompts might also include few-shot examples if add_context is True
    discret_context_prompts: List[str] = [
        generate_prompt(
            sig_info, question_template, question_template_format, instruction_template, instruction_template_format, feature_names, 
            processed=True, add_context=add_context, example_dict=reduce_example_dict(all_example_dict, get_family_label(get_radioml_label(signal_paths[i]), MODULATION_FAMILIES), max_examples=2*len(options)), 
            decimal_precision=decimal_precision, options=options, 
            discretizers=discretizers, scaler=scaler, discretized=True
        ) for i, sig_info in tqdm(enumerate(signal_discretized_feature), desc="Generating discrete prompts")
    ]

    # Compile all processed data and metadata into a dictionary
    data: Dict[str, Any] = {
        'signal_paths': signal_paths,               # Original signal paths
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
        '#snr': len(set(signal_snr))                 # Number of unique SNRs
    }
    
    # Return the dictionary containing all processed information
    return data

# %%

if __name__ == "__main__":
    
    train_signal_paths, example_paths, train_signal_labels, train_signal_snr = radioml_example_maker(pattern="../../data/RadioML/*/train/*/*.npy")
    options = list(set(train_signal_labels))
    
    train_signal_paths = train_signal_paths[:10]
 
    feature_names = [
        # 'min', 'max', 'mean', 'variance', 'skewness', 'kurtosis', 
                         'moment_0', 'moment_1', 'moment_2', 'moment_3', 'moment_4', 
                         'moment_5', 'moment_6', 'moment_7', 'moment_8', 'moment_9',
                         'kstat_1', 'kstat_2', 'kstat_3', 'kstat_4',
                         'kstatvar_1', 'kstatvar_2']
    
    train_data = get_processed_data(train_signal_paths, train_signal_labels, train_signal_snr, feature_names, example_paths, scaler=None, discretizers=None, decimal_precision=3, add_context=True)  
    
    # Save the processed data to a file
    save_processed_data(train_data, '../../data/RadioML/train_data.pkl')
    print("Processed train data saved successfully.")

    test_signal_paths = glob('../../data/RadioML/*/test/*/*.npy')
    test_signal_labels = [get_radioml_label(sig) for sig in test_signal_paths]
    test_signal_snr = [get_radioml_snr(sig) for sig in test_signal_paths]

    test_signal_paths = test_signal_paths[:10]

    test_data = get_processed_data(test_signal_paths, test_signal_labels, test_signal_snr, feature_names, example_paths, scaler=train_data['scaler'], discretizers=train_data['discretizers'], decimal_precision=3, add_context=True)

    # Save the processed test data to a file
    save_processed_data(test_data, '../../data/RadioML/test_data.pkl')
    print("Processed test data saved successfully.")

# %%
