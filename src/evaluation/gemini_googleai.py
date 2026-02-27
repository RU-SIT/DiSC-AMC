"""Gemini API evaluation provider for modulation classification."""

import os
import sys
import traceback
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as genai

from src.naming import ExperimentConfig, eval_result_name

from .utils import (
    load_data, build_prompts_data, load_existing_results,
    get_prompts_to_process, save_results_atomic, build_result_entry,
    sort_results_by_prompt, get_unique_prompts, print_metrics, clean_acc,
)

load_dotenv()

CLASS_NAMES = ['4ASK', '4PAM', '8ASK', '16PAM', 'CPFSK', 'DQPSK', 'GFSK', 'GMSK', 'OQPSK', 'OOK']

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


def get_gemini_response(prompt: str, model, temperature: float = 0.7) -> str:
    """Get response from Gemini API."""
    if isinstance(model, str):
        model = genai.GenerativeModel(model)
    try:
        response = model.generate_content(prompt, generation_config={"temperature": temperature})
        return response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        return ""


def clean_up_response_label(results):
    """Fix response labels longer than 10 chars (Gemini flash artifact)."""
    problematic = []
    for sample in results:
        if len(sample['response_label']) > 10:
            problematic.append(sample)
            sample['response_label'] = sample['raw_response'][-6:]
    return results, problematic


def _output_path(cfg: ExperimentConfig, prompt_type: str, model_name: str) -> str:
    return eval_result_name(cfg, prompt_type, model_name, "gemini")


def main(dataset_folder='unlabeled_10k', prompt_type='discret_prompts',
         model_name="gemini-2.5-flash", noise_mode='noisySignal',
         n_bins=10, top_k=5, num_tries=3, prediction_source='dnn',
         feature_type='stats', n_components=0,
         ood_train_folder='', use_rag=False, rag_k=0,
         output_dir='.', prompt_version='v1'):
    cfg = ExperimentConfig(
        dataset_folder=dataset_folder,
        prediction_source=prediction_source,
        noise_mode=noise_mode,
        n_bins=n_bins,
        top_k=top_k,
        feature_type=feature_type,
        n_components=n_components,
        ood_train_folder=ood_train_folder,
        use_rag=use_rag,
        rag_k=rag_k,
        prompt_version=prompt_version,
    )
    results = []
    filepath = os.path.join(output_dir, _output_path(cfg, prompt_type, model_name))

    try:
        data, _, _ = load_data(f'../../data/own/{dataset_folder}', noise_mode, n_bins, top_k,
                               prediction_source=prediction_source,
                               feature_tag=f'emb{n_components}' if feature_type == 'embeddings' else '')

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model_obj = genai.GenerativeModel(model_name)

        all_prompts_data = build_prompts_data(data, prompt_type)
        results, prompts_done, completed = load_existing_results(filepath, num_tries)
        prompts_to_process = get_prompts_to_process(all_prompts_data, completed, prompts_done)

        for prompt_data, num_done in tqdm(prompts_to_process):
            try:
                prompt, filename = prompt_data['prompt'], prompt_data['filename']
                for i in range(num_done, num_tries):
                    raw_response = get_gemini_response(prompt, model=model_obj, temperature=0.2)
                    if not raw_response:
                        raise Exception("reached limits of Gemini API, stopping")
                    results.append(build_result_entry(filename, i, prompt, raw_response))
            except Exception as e:
                if "reached limits" in str(e):
                    print("Reached Gemini API limits, stopping.")
                    break
                print(f"Error processing {prompt_data['filename']}: {e}")
                print(traceback.format_exc())

    except KeyboardInterrupt:
        print("\nInterrupted. Saving partial results...")
    except Exception as e:
        print(traceback.format_exc())
    finally:
        save_results_atomic(results, filepath)


def read_results(dataset_folder='unlabeled_10k', prompt_type='discret_prompts',
                 model_name='gemini-2.5-flash', noise_mode='noisySignal',
                 n_bins=5, top_k=5, prediction_source='dnn',
                 feature_type='stats', n_components=0,
                 ood_train_folder='', use_rag=False, rag_k=0,
                 output_dir='.', prompt_version='v1'):
    """Read and optionally clean results."""
    import json
    cfg = ExperimentConfig(
        dataset_folder=dataset_folder,
        prediction_source=prediction_source,
        noise_mode=noise_mode,
        n_bins=n_bins,
        top_k=top_k,
        feature_type=feature_type,
        n_components=n_components,
        ood_train_folder=ood_train_folder,
        use_rag=use_rag,
        rag_k=rag_k,
        prompt_version=prompt_version,
    )
    filepath = os.path.join(output_dir, _output_path(cfg, prompt_type, model_name))
    with open(filepath, 'r') as f:
        results = json.load(f)
    if model_name == "gemini-2.5-flash":
        results, problematic = clean_up_response_label(results)
        if problematic:
            print(f"Fixed {len(problematic)} problematic response labels.")
    return results


if __name__ == '__main__':
    DATASET_FOLDER = 'unlabeled_10k'
    PROMPT_TYPE = 'discret_prompts'
    MODEL_NAME = "gemini-2.5-flash"
    NOISE_MODE = 'noisySignal'
    N_BINS, TOP_K, NUM_TRIES = 5, 5, 1

    # main(DATASET_FOLDER, PROMPT_TYPE, MODEL_NAME, NOISE_MODE, N_BINS, TOP_K, NUM_TRIES)
    results = read_results(DATASET_FOLDER, PROMPT_TYPE, MODEL_NAME, NOISE_MODE, N_BINS, TOP_K)
    sorted_results = sort_results_by_prompt(results)
    print(f"Unique prompts: {len(get_unique_prompts(results))}")
    print_metrics(sorted_results, CLASS_NAMES)
