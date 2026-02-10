"""
Data Processing Module for Signal Feature Extraction and Prompt Generation.

Provides functions for loading signal data, extracting statistical features,
scaling/discretizing features, and generating LLM prompts with optional
few-shot examples.
"""
# -*- coding: utf-8 -*-
import os
import json
import random
import pickle
from glob import glob

import numpy as np
from tqdm import tqdm
from scipy import stats
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from typing import List, Tuple, Dict, Union, Any, Optional
from sklearn.preprocessing import KBinsDiscretizer

def load_npy_file(file_path: str) -> np.ndarray:
    """Load data from a .npy file."""
    data = np.load(file_path)
    return data

def save_to_json(data: dict, file_path: str) -> None:
    """Save a dictionary to a JSON file with pretty-printing."""
    with open(file_path, 'w') as f:
        # 'indent=2' formats the JSON output with an indentation of 2 spaces for readability
        json.dump(data, f, indent=2)

def load_from_json(file_path: str) -> Any:
    """Load and return data from a JSON file."""
    with open(file_path, 'r') as f:
        # and return the corresponding Python object
        return json.load(f)
    
def save_processed_data(data: Any, file_path: str) -> None:
    """Save data to a pickle file."""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_processed_data(file_path: str) -> Any:
    """Load data from a pickle file."""
    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data

def get_features(signal_data: np.ndarray, feature_names: List[str], snr=None) -> Dict[str, Union[float, np.ndarray, int]]:
    """Calculate statistical features from signal data.
    
    Supported features: nobs, min, max, mean, variance, skewness, kurtosis,
    moment_k, kstat_n, kstatvar_n, snr.
    """
    stats_summary: Dict[str, Union[float, np.ndarray, int]] = {}
    des_stats: Any = None

    for feature in feature_names:
        if feature == 'snr':
            if snr is not None:
                stats_summary['snr'] = snr
        elif feature in ('nobs', 'min', 'max', 'mean', 'variance', 'skewness', 'kurtosis'):
            if des_stats is None:
                des_stats = stats.describe(signal_data, axis=None)
            stats_summary[feature] = getattr(des_stats, feature)
        elif feature.startswith('moment_'):
            moment_order = int(feature.split('_')[1])
            stats_summary[feature] = stats.moment(signal_data, moment=moment_order, axis=None)
        elif feature.startswith('kstat_'):
            kstat_order = int(feature.split('_')[1])
            stats_summary[feature] = stats.kstat(signal_data, n=kstat_order)
        elif feature.startswith('kstatvar_'):
            kstatvar_order = int(feature.split('_')[1])
            stats_summary[feature] = stats.kstatvar(signal_data, n=kstatvar_order)
        else:
            raise ValueError(f"Unknown feature: {feature}")

    return stats_summary

def dict_to_np(signal_summary: Dict[str, Union[float, np.ndarray, int]], feature_names: List[str]) -> np.ndarray:
    """Flatten a feature dict into a 1D numpy array, ordered by feature_names."""
    flat_features = []
    for feature in feature_names:
        value = signal_summary[feature]
        if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
            flat_features.extend(value)
        else:
            flat_features.append(value)

    return np.array(flat_features, dtype=float)
def convert_signal_to_complex(signal: np.ndarray) -> np.ndarray:
    """Convert 2D real/imag array to 1D complex array."""
    real_part = signal[:, 0]
    imaginary_part = signal[:, 1]
    complex_signal = real_part + 1j * imaginary_part
    return complex_signal

def split_real_imaginary(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split complex signal into (real, imag) column-stacked array."""
    real_part = np.real(signal)
    imaginary_part = np.imag(signal)
    signal = np.column_stack((real_part, imaginary_part))
    return signal
def create_options(class_names: List[str]) -> str:
    """Format class names as multiple-choice options: "[A: name1, B: name2, ...]"."""
    assert 1 <= len(class_names) <= 26, "Class names must be between 1 and 26 to map to A-Z."

    options_str = "["
    for i, class_name in enumerate(class_names):
        options_str += f"{chr(64 + i + 1)}: {class_name}, "

    options_str = options_str[:-2]
    options_str += "]"
    return options_str

def discretize_features(
    signal_stats: np.ndarray, 
    n_bins: int = 10, 
    strategy: str = 'uniform', 
    discretizers: Optional[Dict[int, KBinsDiscretizer]] = None
) -> Tuple[np.ndarray, Dict[int, KBinsDiscretizer]]:
    """Discretize each feature column independently using KBinsDiscretizer.
    
    Uses pre-fitted discretizers if provided, otherwise fits new ones.
    Returns (discretized_data, discretizers_dict).
    """
    continuous_data: np.ndarray = np.array(signal_stats)
    
    if discretizers is None:
        discretizers: Dict[int, KBinsDiscretizer] = {}
        
    discretized_data = np.zeros_like(continuous_data, dtype=int)
    
    for feature_idx in range(continuous_data.shape[1]):
        feature_values: np.ndarray = continuous_data[:, feature_idx].reshape(-1, 1)
        
        if feature_idx in discretizers:
            discretizer: KBinsDiscretizer = discretizers[feature_idx]
            discretized_feature: np.ndarray = discretizer.transform(feature_values)
        else:
            discretizer = KBinsDiscretizer(
                n_bins=n_bins, 
                encode='ordinal',  # Output bin indices (0 to n_bins-1)
                strategy=strategy,
            )
            discretized_feature = discretizer.fit_transform(feature_values)
            discretizers[feature_idx] = discretizer
        
        # of the output array. Flatten the result to ensure it fits into the 1D slice.
        discretized_data[:, feature_idx] = discretized_feature.flatten()

    return discretized_data, discretizers

def get_feature_dim(info: Dict[str, Any]) -> Dict[str, int]:
    """Return {feature_name: num_elements} for each feature in info dict."""
    feature_dim: Dict[str, int] = {}
    for key in info.keys():
        # Check if the value associated with the key is iterable (e.g., list, tuple, np.ndarray)
        if hasattr(info[key], '__iter__') and not isinstance(info[key], (str, bytes)):
            feature_dim[key] = len(info[key])
        else:
            feature_dim[key] = 1
    return feature_dim

def get_discrete_info(
    info: Dict[str, Union[float, np.ndarray, int]], 
    discretizers: Dict[int, KBinsDiscretizer]
) -> Dict[str, Union[int, np.ndarray]]:
    """Discretize a feature dict using pre-fitted discretizers, returning bin indices."""
    np_info: np.ndarray = dict_to_np(info, list(info.keys())) 
    
    discretized_summary = np.zeros_like(np_info, dtype=int)

    for feature_idx in range(np_info.shape[0]):
        feature_values = np_info[feature_idx].reshape(-1, 1)
        
        discretizer: KBinsDiscretizer = discretizers[feature_idx]
        
        discretized_feature: np.ndarray = discretizer.transform(feature_values)
        
        discretized_summary[feature_idx] = discretized_feature.ravel()[0]

    signal_info: Dict[str, Union[int, np.ndarray]] = {}
    i = 0
    for key in info.keys():
        if hasattr(info[key], '__iter__') and not isinstance(info[key], (str, bytes)):
            signal_info[key] =  np.array([discretized_summary[i], discretized_summary[i+1]], dtype=int)
            i += 2
        else:
            signal_info[key] = discretized_summary[i]
            i += 1
    
    return signal_info

def get_scaled_info(
    info: Dict[str, Union[float, np.ndarray, int]], 
    scaler: StandardScaler # Or a more general type like BaseEstimator if other scalers are used
) -> Dict[str, Union[float, np.ndarray]]:
    """Scale a feature dict using a pre-fitted scaler, preserving dict structure."""
    np_info: np.ndarray = dict_to_np(info, list(info.keys())).reshape(1, -1)
    
    scaled_summary: np.ndarray = scaler.transform(np_info)
    
    signal_info: Dict[str, Union[float, np.ndarray]] = {}
    i = 0
    for key in info.keys():
        if hasattr(info[key], '__iter__') and not isinstance(info[key], (str, bytes)):
            signal_info[key] =  np.array([scaled_summary[0, i], scaled_summary[0, i+1]])
            i += 2
        else:
            # from the `scaled_summary` array. scaled_summary[0, i] accesses the value.
            signal_info[key] = scaled_summary[0, i]
            i += 1
    
    return signal_info

def get_text_info(info: Dict[str, Union[float, int, np.ndarray]], decimal_precision: int = 3) -> str:
    """Format feature dict as text string with rounded values."""
    text_info = ""
    for key in info.keys():
        value = info[key]
        # but not a string or bytes object
        if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
            rounded_values = [f"{round(x, ndigits=decimal_precision)}" for x in value]
            text_info += f"{key}: (" + ', '.join(rounded_values) + "), "
        else:
            text_info += f"{key}: {round(value, ndigits=decimal_precision)}, "
            
    return text_info

def _to_base26_string(n: int) -> str:
    """Converts a 0-indexed integer to a base-26 string (A, B,..., Z, AA, AB...)."""
    if n < 0:
        return ""
    result = ""
    num = n + 1
    while num > 0:
        rem = (num - 1) % 26
        result = chr(65 + rem) + result
        num = (num - 1) // 26
    return result

def get_discrete_text_info(info: Dict[str, Union[int, np.ndarray]]) -> str:
    """Format discretized feature dict as text with letter codes (0->A, 1->B, ...)."""
    text_info = ""
    for key in info.keys():
        value = info[key]
        if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
            letter_values = [_to_base26_string(int(x)) for x in value]
            text_info += f"{key}: (" + ', '.join(letter_values) + "), "
        else:
            text_info += f"{key}: {_to_base26_string(int(value))}, "

    return text_info

def get_question_answer(
    signal: Union[np.ndarray, Dict[str, Union[float, int, np.ndarray]]], 
    options: List[str], 
    template: str,
    template_format: List[str], 
    answer: Optional[str] = None, 
    feature_names: Optional[List[str]] = None, 
    processed: bool = True,
    snr: Optional[float] = None,
    decimal_precision: Optional[int] = None, 
    discretizers: Optional[Dict[int, KBinsDiscretizer]] = None, 
    scaler: Optional[StandardScaler] = None, 
    discretized: bool = False
) -> str:
    """
    Generates a question or a question-answer pair based on signal features.

    This function takes either raw signal data or pre-processed features, 
    formats the features into a text string (either continuous rounded values 
    or discrete letter codes), and inserts this information, along with 
    multiple-choice options and optionally the correct answer letter, into a 
    provided template string.

    Args:
        signal (Union[np.ndarray, Dict[str, Union[float, int, np.ndarray]]]): 
            Either the raw signal data (if processed=False) or a dictionary 
            containing pre-calculated features (if processed=True).
        options (List[str]): A list of possible class names (answers) for the signal.
        template (str): A format string with placeholders for signal information, 
                        options, and the answer letter (e.g., "Info: {}\nOptions: {}\nAnswer: {}").
        template_format (List[str]): A list of format specifiers for the template string.
                                      Each specifier corresponds to a placeholder in the template.
        answer (Optional[str], optional): The correct class name for the signal. 
                                          If provided, its corresponding letter (A, B, C...) 
                                          will be included in the output. Defaults to None.
        feature_names (Optional[List[str]], optional): A list of feature names to 
                                                      calculate if processed=False. 
                                                      Required if processed=False. Defaults to None.
        processed (bool, optional): Flag indicating if the input `signal` is already 
                                    a dictionary of features (True) or raw data (False). 
                                    Defaults to True.
        decimal_precision (Optional[int], optional): The number of decimal places for 
                                                     formatting continuous features. 
                                                     Required if discretized=False. Defaults to None.
        discretizers (Optional[Dict[int, KBinsDiscretizer]], optional): 
                                                     A dictionary mapping feature indices to 
                                                     pre-fitted discretizers. Required if 
                                                     discretized=True and processed=False. 
                                                     Defaults to None.
        scaler (Optional[StandardScaler], optional): A pre-fitted scaler object. Required if 
                                                     discretized=False and processed=False. 
                                                     Defaults to None.
        discretized (bool, optional): Flag indicating whether to use discretized features 
                                      (True) or scaled continuous features (False). 
                                      Defaults to False.

    Returns:
        str: The formatted text string based on the template, containing the signal 
             information, options, and optionally the answer letter.
             
    Raises:
        AssertionError: If required arguments (feature_names, discretizers, scaler, 
                        decimal_precision) are missing based on the `processed` and 
                        `discretized` flags.
    """
    signal_info: Dict[str, Union[float, int, np.ndarray]] = signal if processed else get_features(signal, feature_names, snr=snr) # type: ignore

    if discretized:
        if not processed:  
            assert (discretizers is not None), "Discretizers must be provided when processing raw data for discretized output."
        discrete_signal_info: Dict[str, Union[int, np.ndarray]] = signal_info if processed else get_discrete_info(signal_info, discretizers) # type: ignore
        formatted_signal_info: str = get_discrete_text_info(discrete_signal_info)
    else:
        assert (isinstance(decimal_precision, int)) and (decimal_precision > 0), "Decimal precision must be a positive integer for continuous output."
        if not processed:
            assert (scaler is not None), "Scaler must be provided when processing raw data for continuous output."
        scaled_signal_info: Dict[str, Union[float, np.ndarray]] = signal_info if processed else get_scaled_info(signal_info, scaler) # type: ignore
        formatted_signal_info: str = get_text_info(scaled_signal_info, decimal_precision)
    
    answer_letter= answer if answer else ""
    
    text: str = template.format(formatted_signal_info, *template_format, answer_letter)
    
    return text

def ktop_example(k_top: List[str], example_dict: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
    """Select entries from example_dict matching the k_top keys."""
    return {k: example_dict[k] for k in k_top if k in example_dict}

def generate_prompt(
    signal_data: Union[np.ndarray, Dict[str, Union[float, int, np.ndarray]]], 
    question_template: str,
    question_template_format: List[str], 
    instruction_template: str, 
    instruction_template_format: List[str],
    feature_names: List[str], 
    options: List[str],
    processed: bool = True, 
    add_context: bool = True, 
    example_dict: Optional[Dict[str, List[np.ndarray]]] = None, 
    decimal_precision: int = 3, 
    discretizers: Optional[Dict[int, KBinsDiscretizer]] = None, 
    scaler: Optional[StandardScaler] = None, 
    discretized: bool = False, 
    example_per_class: int = 1,
    examples_processed: bool = False,
) -> str:
    """
    Generates a complete prompt string including instructions, optional few-shot examples (context), 
    and the final question based on signal data or features.

    This function orchestrates the creation of a prompt suitable for a language model. 
    It first generates the question part using `get_question_answer`. If `add_context` 
    is True, it then constructs few-shot examples using the provided `example_dict`, 
    formatting each example as a question-answer pair using `get_question_answer`. 
    Finally, it combines the instruction template (filled with the context) and the 
    generated question.

    Args:
        signal_data (Union[np.ndarray, Dict[str, Union[float, int, np.ndarray]]]): 
            The input signal data. Can be raw signal (np.ndarray if processed=False) 
            or a dictionary of pre-calculated features (if processed=True).
        question_template (str): 
            The template string for formatting the question part (e.g., "Info: {}\nOptions: {}\nAnswer: {}").
        question_template_format (List[str]):
            A list of format specifiers for the question template, used to insert
        instruction_template (str): 
            The template string for the overall instruction, which should include a 
            placeholder for the context (e.g., "Based on the examples:\n{}\nAnswer the following question:").
        instruction_template_format (List[str]):
            A list of format specifiers for the instruction template, used to insert
        feature_names (List[str]): 
            List of feature names to be used, especially if `processed` is False.
        options (List[str]): 
            A list of possible class names (answers) for the signal, used for formatting options.
        processed (bool, optional): 
            Flag indicating if `signal_data` is already processed features (True) or raw data (False). 
            Defaults to True.
        add_context (bool, optional): 
            Flag indicating whether to add few-shot examples (context) to the prompt. 
            Defaults to True.
        example_dict (Optional[Dict[str, List[np.ndarray]]], optional): 
            A dictionary where keys are class labels (str) and values are lists of raw signal 
            examples (np.ndarray) for that class. Required if `add_context` is True. 
            Defaults to None.
        decimal_precision (int, optional): 
            Number of decimal places for formatting continuous features. Used by `get_question_answer`. 
            Defaults to 3.
        discretizers (Optional[Dict[int, KBinsDiscretizer]], optional): 
            Pre-fitted discretizers, used if `discretized` is True. Passed to `get_question_answer`. 
            Defaults to None.
        scaler (Optional[StandardScaler], optional): 
            Pre-fitted scaler, used if `discretized` is False. Passed to `get_question_answer`. 
            Defaults to None.
        discretized (bool, optional): 
            Flag indicating whether to use discretized features (True) or scaled continuous features (False). 
            Passed to `get_question_answer`. Defaults to False.
        example_per_class (int, optional): 
            The number of examples to include in the context for each class. 
            Defaults to 1.
        examples_processed (bool, optional):
            If True, the few-shot examples in *example_dict* are already
            pre-processed feature dicts (e.g. from embedding pipeline).
            They will be passed to ``get_question_answer`` with
            ``processed=True``.  Defaults to False (raw signals processed
            on the fly).

    Returns:
        str: The fully constructed prompt string, including instructions, context (if added), and the question.
        
    Raises:
        AssertionError: If `add_context` is True but `example_dict` is None.
    """
    question: str = get_question_answer(
        signal=signal_data, 
        options=options, 
        template=question_template, 
        template_format=question_template_format,
        processed=processed, 
        feature_names=feature_names, 
        decimal_precision=decimal_precision, 
        discretizers=discretizers, 
        scaler=scaler, 
        discretized=discretized
    )
    
    context: str = ""
    if add_context:
        assert example_dict is not None, "Example dictionary must be provided for context."
        for key in example_dict.keys():
            for i in range(example_per_class):
                if i < len(example_dict[key]):
                    context += get_question_answer(
                        signal=example_dict[key][i][0] if isinstance(example_dict[key][i], tuple) else example_dict[key][i],
                        options=options, 
                        template=question_template,
                        template_format=question_template_format,
                        answer=key,
                        processed=examples_processed,
                        snr=example_dict[key][i][1] if (isinstance(example_dict[key][i], tuple) and len(example_dict[key][i]) > 1) else None,
                        feature_names=feature_names, 
                        decimal_precision=decimal_precision, 
                        discretizers=discretizers, 
                        scaler=scaler, 
                        discretized=discretized
                    ) + "\n"
            
    instruct: str = instruction_template.format(*instruction_template_format, context)
    
    prompt: str = instruct + question

    return prompt
    
def get_processed_data(
    signal_paths: List[str], 
    signal_labels: List[str], 
    signal_snr: List[Union[int, float]], 
    feature_names: Optional[List[str]], 
    question_template: str, 
    instruction_template: str, 
    example_dict: Optional[Dict[str, List[np.ndarray]]], 
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
        question_template (str): Template string for generating questions.
        instruction_template (str): Template string for generating instructions (including context).
        example_dict (Optional[Dict[str, List[np.ndarray]]]): Dictionary mapping labels to lists 
                                                               of example signals for few-shot context. 
                                                               Required if add_context is True.
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
    if not feature_names:
        feature_names = ['nobs', 'min', 'max', 'mean', 'variance', 'skewness', 'kurtosis', 
                         'moment_0', 'moment_1', 'moment_2', 'moment_3', 'moment_4', 
                         'moment_5', 'moment_6', 'moment_7', 'moment_8', 'moment_9',
                         'kstat_1', 'kstat_2', 'kstat_3', 'kstat_4',
                         'kstatvar_1', 'kstatvar_2']

    signals_data: List[np.ndarray] = [load_npy_file(path) for path in signal_paths]
    signal_summaries: List[Dict[str, Union[float, np.ndarray, int]]] = [get_features(sig, feature_names) for sig in tqdm(signals_data, desc="Calculating features")]
    signal_features: List[np.ndarray] = [dict_to_np(sig, feature_names) for sig in tqdm(signal_summaries, desc="Converting features to array")]
    
    if discretizers is None:
        discretizers: Dict[int, KBinsDiscretizer] = {}
        _, discretizers = discretize_features(np.array(signal_features), n_bins=10, strategy='uniform')
    signal_discretized_feature: List[Dict[str, Union[int, np.ndarray]]] = [get_discrete_info(sig, discretizers) for sig in tqdm(signal_summaries, desc="Discretizing features")]

    if scaler is None:
        scaler = StandardScaler()
    
    signal_stats: List[Dict[str, Union[float, np.ndarray]]] = [get_scaled_info(sig, scaler) for sig in tqdm(signal_summaries, desc="Scaling features")]

    options: List[str] = list(set(signal_labels))

    options_str: str = create_options(options)
    question_template_format: List[str] = [options_str]
    instruction_template_format: List[str] = []

    context_prompts: List[str] = [
        generate_prompt(
            sig_info, question_template, question_template_format, instruction_template, instruction_template_format, feature_names, 
            processed=True, add_context=add_context, example_dict=example_dict, 
            decimal_precision=decimal_precision, options=options, 
            discretizers=None, scaler=scaler, discretized=False
        ) for sig_info in tqdm(signal_stats, desc="Generating continuous prompts")
    ]
    
    discret_context_prompts: List[str] = [
        generate_prompt(
            sig_info, question_template, question_template_format, instruction_template, instruction_template_format, feature_names, 
            processed=True, add_context=add_context, example_dict=example_dict, 
            decimal_precision=decimal_precision, options=options, 
            discretizers=discretizers, scaler=scaler, discretized=True
        ) for sig_info in tqdm(signal_discretized_feature, desc="Generating discrete prompts")
    ]

    data: Dict[str, Any] = {
    }
    
    return data

def get_family_example(family: Dict[str, Any], example_paths: Dict[str, str]) -> Dict[str,str]:
    """
    Generates a dictionary of example signal paths for each family in the family dictionary.
    This function takes a family dictionary where keys are family names and values are lists of example signal indices.
    It constructs a new dictionary where each key corresponds to a family name and the value is a list of file paths
    to the example signals. The file paths are obtained from the example_paths dictionary, which maps signal indices to file paths. 
    Args:
        family (Dict[str, Any]): A dictionary where keys are family names (str) and values are lists of example signal indices.
        example_paths (Dict[str, str]): A dictionary mapping signal indices (str) to file paths (str).
    Returns:
        Dict[str, str]: A dictionary where keys are family names (str) and values are lists of file paths (str) to the example signals.
    """
    family_example = {key: [] for key in family.keys()}
    for key in family.keys():
        ele = family[key]
        if isinstance(ele, dict):
            for k in ele.keys():
                for v in ele[k]:
                    family_example[key].extend(example_paths[v])
        elif isinstance(ele, list):
            for k in ele:
                family_example[key].extend(example_paths[k])
        else:
            # If the element is not a list or dict, treat it as a single example
            family_example[key].append(example_paths[ele])
    
    return family_example

def reduce_example_dict(
    example_dict: Dict[str, List[Any]], # Assuming values are lists of examples
    label: str,
    max_examples: int = 10
) -> Dict[str, List[Any]]:
    """
    Selects up to max_examples examples, ensuring diversity and label inclusion.

    This function selects a total of up to `max_examples` individual examples
    from the input dictionary `example_dict`. It prioritizes including at least
    one example from the category specified by `label`. It then tries to include
    one example from as many other distinct categories (keys) as possible.
    If more examples are needed to reach `max_examples`, they are chosen randomly
    from the remaining pool of available examples.

    Args:
        example_dict (Dict[str, List[Any]]): A dictionary where keys are category
                                            labels (str) and values are lists of
                                            corresponding examples (Any type).
        label (str): The specific category label (key) that must have at least
                     one example included in the output.
        max_examples (int, optional): The target maximum total number of individual
                                      examples in the returned dictionary. Defaults to 10.

    Returns:
        Dict[str, List[Any]]: A dictionary containing the selected examples, grouped
                              by their original keys. The values are lists of the
                              selected examples for that key. The total number of
                              examples across all lists will be at most `max_examples`.
                              Returns an empty dictionary if the input is empty or
                              contains only empty lists. Returns all examples grouped
                              by key if the total number available is less than or
                              equal to `max_examples`.

    Raises:
        ValueError: If `max_examples` is less than 1.
        ValueError: If the specified `label` is not found as a key in `example_dict`.
        ValueError: If the list associated with the `label` key is empty after filtering
                    keys with no examples.
    """
    if max_examples < 1:
        raise ValueError("max_examples must be at least 1.")

    if label not in example_dict:
        raise ValueError(f"Label '{label}' not found as a key in the example dictionary.")

    valid_example_dict = {k: v for k, v in example_dict.items() if v}

    if not valid_example_dict:
        return {}

    if label not in valid_example_dict:
         raise ValueError(f"Example list for label '{label}' is empty or the key was removed.")

    all_examples: List[Tuple[str, Any]] = [
        (key, item) for key, items in valid_example_dict.items() for item in items
    ]

    if len(all_examples) <= max_examples:
        # print(f"Warning: Total available examples ({len(all_examples)}) is less than or equal to max_examples ({max_examples}). Returning all available examples grouped by key.")
        return valid_example_dict

    selected_examples_flat: List[Tuple[str, Any]] = []
    remaining_indices_pool = list(range(len(all_examples)))

    label_indices_in_pool = [
        idx for idx in remaining_indices_pool if all_examples[idx][0] == label
    ]
    chosen_label_pool_idx = random.choice(label_indices_in_pool)
    selected_examples_flat.append(all_examples[chosen_label_pool_idx])
    remaining_indices_pool.remove(chosen_label_pool_idx)
    num_selected = 1

    other_keys = [k for k in valid_example_dict.keys() if k != label]

    key_to_pool_indices: Dict[str, List[int]] = {}
    for idx in remaining_indices_pool:
        key = all_examples[idx][0]
        if key not in key_to_pool_indices:
            key_to_pool_indices[key] = []
        key_to_pool_indices[key].append(idx)

    for key in other_keys:
        if num_selected >= max_examples:
            break
        if key in key_to_pool_indices and key_to_pool_indices[key]:
            chosen_pool_idx = random.choice(key_to_pool_indices[key])

            selected_examples_flat.append(all_examples[chosen_pool_idx])
            indices_to_remove_from_pool.append(chosen_pool_idx)
            keys_represented.add(key)
            num_selected += 1

            key_to_pool_indices[key].remove(chosen_pool_idx)

    indices_removed_in_step2_set = set(indices_to_remove_from_pool)
    remaining_indices_pool = [idx for idx in remaining_indices_pool if idx not in indices_removed_in_step2_set]

    remaining_needed = max_examples - num_selected
    if remaining_needed > 0 and remaining_indices_pool:
        num_to_sample = min(remaining_needed, len(remaining_indices_pool))
        randomly_chosen_pool_indices = random.sample(remaining_indices_pool, num_to_sample)

        for pool_idx in randomly_chosen_pool_indices:
             selected_examples_flat.append(all_examples[pool_idx])

    new_example_dict: Dict[str, List[Any]] = {}
    for key, item in selected_examples_flat:
        if key not in new_example_dict:
            new_example_dict[key] = []
        new_example_dict[key].append(item)

    return new_example_dict

def get_family_label(signal_label: str, family: Dict[str, Any]) -> str:
    for key in family.keys():
        ele = family[key]
        if isinstance(ele, dict):
            for k in ele.keys():
                if signal_label in ele[k]:
                    return key
        elif isinstance(ele, list):
            if signal_label in ele:
                return key
    return ''

