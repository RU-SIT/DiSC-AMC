"""
Shared utilities for LLM evaluation across all providers (Gemini, OpenAI, Unsloth).

Contains metrics, data sampling, result I/O, and text extraction functions
that were previously duplicated in each provider file.
"""

import os
import json
import pickle
import sys
import traceback
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from src.naming import test_pkl_name


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_think_text(response: str) -> Tuple[int, str]:
    """Extract text between <think> tags. Returns (end_index, extracted_text)."""
    return extract_tag(response, "<think>", "</think>")


def extract_tag(response: str, start_tag: str, end_tag: str) -> Tuple[int, str]:
    """Extract text between arbitrary tags. Returns (end_index, extracted_text)."""
    start_idx = response.find(start_tag)
    end_idx = response.find(end_tag, start_idx + len(start_tag))
    if start_idx != -1 and end_idx != -1:
        return end_idx + len(end_tag), response[start_idx + len(start_tag):end_idx].strip()
    return 0, ""


# ---------------------------------------------------------------------------
# Data sampling
# ---------------------------------------------------------------------------

def sample_per_label(data: dict, per_label: int = 20, seed: int = 42, shuffle: bool = True):
    """
    Sample up to `per_label` indices per label from `data['labels']`.
    Returns (sampled_data, selected_indices, rng).
    """
    rng = np.random.default_rng(seed)

    indices_by_label = defaultdict(list)
    for i, label in enumerate(data['labels']):
        indices_by_label[label].append(i)

    selected_indices = []
    for _, indices in indices_by_label.items():
        num_to_sample = min(per_label, len(indices))
        sampled = rng.choice(indices, size=num_to_sample, replace=False)
        selected_indices.extend(int(i) for i in sampled)

    if shuffle:
        rng.shuffle(selected_indices)

    sampled_data = {}
    for key, value in data.items():
        if isinstance(value, (list, np.ndarray)):
            sampled_data[key] = [value[i] for i in selected_indices if i < len(value)]
        else:
            sampled_data[key] = value

    print(f"Total samples selected: {len(sampled_data.get('labels', []))}")
    unique_labels, counts = np.unique(sampled_data.get('labels', []), return_counts=True)
    print("Label distribution:", dict(zip(unique_labels, counts)))

    return sampled_data, selected_indices, rng


# ---------------------------------------------------------------------------
# Result sorting & grouping
# ---------------------------------------------------------------------------

def get_unique_prompts(results: List[Dict]) -> List[str]:
    """Get unique prompt filenames from results."""
    return list(set(r['filename'] for r in results))


def sort_results_by_prompt(results: List[Dict]) -> Dict[int, List[Dict]]:
    """Sort results by prompt filename and assign numeric keys (deterministic order)."""
    sorted_results = {}
    prompts_to_id = {}
    prompt_id = 0

    all_prompts = list(dict.fromkeys(r['filename'] for r in results))
    for prompt in all_prompts:
        if prompt not in prompts_to_id:
            prompts_to_id[prompt] = prompt_id
            prompt_id += 1

    for result in results:
        current_id = prompts_to_id[result['filename']]
        sorted_results.setdefault(current_id, []).append(result)

    return sorted_results


# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------

def find_classes_in_text(text: str, class_names: List[str]) -> List[str]:
    """Find which class names appear in text."""
    return [c for c in class_names if c in text]


def acc(results: Dict[int, List[Dict]]) -> Tuple[int, int, float]:
    """Calculate raw accuracy across all tries."""
    correct = total = 0
    for value in results.values():
        true_label = value[0]['true_label']
        for v in value:
            if true_label in v['response_label']:
                correct += 1
            total += 1
    return correct, total, correct / total


def clean_acc(results: Dict[int, List[Dict]], class_names: List[str]) -> Tuple[int, int, float]:
    """Calculate accuracy only on responses containing a valid class name."""
    correct = total = 0
    for value in results.values():
        true_label = value[0]['true_label']
        for v in value:
            found = find_classes_in_text(v['response_label'], class_names)
            if not found:
                continue
            if true_label in found:
                correct += 1
            total += 1
    return correct, total, correct / total


def pass_acc(results: Dict[int, List[Dict]]) -> Tuple[int, int, float]:
    """Calculate pass@k accuracy (at least one correct in k tries)."""
    passed = 0
    total = len(results)
    for value in results.values():
        true_label = value[0]['true_label']
        if any(true_label in v['response_label'] for v in value):
            passed += 1
    return passed, total, passed / total


def majority_acc(results: Dict[int, List[Dict]]) -> Tuple[int, int, float]:
    """Calculate majority-vote accuracy."""
    passed = 0
    total = len(results)
    threshold = len(next(iter(results.values()))) / 2
    for value in results.values():
        true_label = value[0]['true_label']
        correct_count = sum(1 for v in value if true_label in v['response_label'])
        if correct_count > threshold:
            passed += 1
    return passed, total, passed / total


def per_class_acc(results: Dict[int, List[Dict]], class_names: List[str]) -> Dict[str, float]:
    """Calculate per-class accuracy after cleaning labels."""
    class_correct = {c: 0 for c in class_names}
    class_total = {c: 0 for c in class_names}

    for value in results.values():
        true_label = value[0]['true_label']
        if true_label not in class_names:
            continue
        for v in value:
            found = find_classes_in_text(v['response_label'], class_names)
            if not found:
                continue
            class_total[true_label] += 1
            if true_label in found:
                class_correct[true_label] += 1

    return {
        c: (class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0)
        for c in class_names
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_data(data_path: str, noise_mode: str, n_bins: int, top_k: int,
              prediction_source: str = "dnn", feature_tag: str = ""):
    """Load pickle data and sample per label.

    Parameters
    ----------
    prediction_source : str
        Which shortlisting source was used to build the pkl.
        Filename is resolved by :func:`naming.test_pkl_name`.
    feature_tag : str
        Optional tag for embedding-based pkl files (e.g. ``"emb10"``).
    """
    fname = test_pkl_name(prediction_source, noise_mode, n_bins, top_k,
                          feature_tag=feature_tag)
    with open(os.path.join(data_path, fname), 'rb') as f:
        whole_data = pickle.load(f)
    return sample_per_label(whole_data, per_label=20, seed=42, shuffle=True)


def build_prompts_data(data: dict, prompt_type: str, limit: int = 332) -> List[Dict]:
    """Build a list of prompt dicts from data."""
    return [
        {'prompt': prompt, 'filename': os.path.basename(data['signal_paths'][i])}
        for i, prompt in enumerate(data[prompt_type][:limit])
    ]


def load_existing_results(filepath: str, num_tries: int, key_field: str = 'filename'):
    """
    Load existing results from a JSON file and determine which prompts still need processing.
    Returns (results, prompts_done_counts, completed_prompt_set).
    """
    if not os.path.exists(filepath):
        return [], {}, set()

    with open(filepath, 'r') as f:
        try:
            results = json.load(f)
            if not isinstance(results, list):
                results = []
        except (json.JSONDecodeError, TypeError):
            results = []

    if not results:
        return [], {}, set()

    print(f"Loaded {len(results)} existing results.")
    prompts_done = {}
    for r in results:
        if r.get('raw_response'):
            prompts_done[r[key_field]] = prompts_done.get(r[key_field], 0) + 1

    completed = {p for p, count in prompts_done.items() if count >= num_tries}
    results = [r for r in results if r[key_field] in completed]
    print(f"Found {len(completed)} fully completed prompts, keeping {len(results)} results.")

    return results, prompts_done, completed


def get_prompts_to_process(all_prompts_data: List[Dict], completed: set,
                           prompts_done: Dict, key_field: str = 'filename') -> List:
    """Filter prompts that still need processing."""
    to_process = []
    for p in all_prompts_data:
        if p[key_field] not in completed:
            num_done = prompts_done.get(p[key_field], 0)
            to_process.append((p, num_done))
    print(f"Identified {len(to_process)} prompts to process.")
    return to_process


def save_results_atomic(results: List[Dict], filepath: str):
    """Save results using atomic temp-file write + rename."""
    tmp = filepath + ".tmp"
    try:
        with open(tmp, 'w') as f:
            json.dump(results, f, indent=4)
        os.replace(tmp, filepath)
        print(f"Results saved to {filepath}")
    except Exception as e:
        print(f"CRITICAL: Failed to save results to {filepath}. Error: {e}")


def build_result_entry(filename: str, try_idx: int, prompt: str,
                       raw_response: str, start_tag: str = "<think>",
                       end_tag: str = "</think>") -> Dict:
    """Build a standardized result dictionary."""
    true_label = filename.split('_')[0]
    end_idx, reasoning = extract_tag(raw_response, start_tag, end_tag)
    return {
        'filename': filename,
        'try': try_idx,
        'prompt': prompt,
        'raw_response': raw_response,
        'true_label': true_label,
        'reasoning': reasoning,
        'response_label': raw_response[end_idx:],
    }


def print_metrics(sorted_results: Dict, class_names: List[str]):
    """Print all standard metrics."""
    print(f"acc: {acc(sorted_results)}")
    print(f"clean-acc: {clean_acc(sorted_results, class_names=class_names)}")
    n_tries = len(next(iter(sorted_results.values())))
    print(f"{n_tries}-pass: {pass_acc(sorted_results)}")
    print(f"{n_tries}-majority: {majority_acc(sorted_results)}")
    print(f"per-class-acc: {per_class_acc(sorted_results, class_names=class_names)}")
