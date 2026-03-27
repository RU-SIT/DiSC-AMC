"""OpenAI GPT evaluation provider for modulation classification."""

import os
import sys
import traceback
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

from src.naming import ExperimentConfig, eval_result_name

from .utils import (
    load_data, build_prompts_data, load_existing_results,
    get_prompts_to_process, save_results_atomic, build_result_entry,
    sort_results_by_prompt, get_unique_prompts, print_metrics,
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


def get_openai_response(client: OpenAI, prompt: str, model: str = "o3-mini") -> str:
    """Get response from OpenAI API with retry logic."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == 2:
                raise
    return ""


def _output_path(cfg: ExperimentConfig, prompt_type: str, model_name: str) -> str:
    return eval_result_name(cfg, prompt_type, model_name, "openai")


def main(dataset_folder='unlabeled_10k', prompt_type='discret_prompts',
         model_name="o3-mini", noise_mode='noisySignal',
         n_bins=10, top_k=5, num_tries=3, prediction_source='dnn',
         feature_type='stats', n_components=0,
         ood_train_folder='', use_rag=False, rag_k=0,
         output_dir='.', prompt_version='v1', backbone='dino'):
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
        backbone=backbone,
    )
    results = []
    filepath = os.path.join(output_dir, _output_path(cfg, prompt_type, model_name))

    try:
        data, _, _ = load_data(f'../../data/own/{dataset_folder}', noise_mode, n_bins, top_k,
                               prediction_source=prediction_source,
                               feature_tag=f'emb{n_components}' if feature_type == 'embeddings' else '',
                               backbone=backbone)
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        all_prompts_data = build_prompts_data(data, prompt_type)
        results, prompts_done, completed = load_existing_results(filepath, num_tries, key_field='prompt')
        prompts_to_process = get_prompts_to_process(all_prompts_data, completed, prompts_done, key_field='prompt')

        for prompt_data, num_done in tqdm(prompts_to_process):
            try:
                prompt, filename = prompt_data['prompt'], prompt_data['filename']
                for i in range(num_done, num_tries):
                    raw_response = get_openai_response(client, prompt, model=model_name)
                    results.append(build_result_entry(filename, i, prompt, raw_response))
            except Exception as e:
                print(f"Error processing {prompt_data['filename']}: {e}")
                print("Stopping to save progress.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted. Saving partial results...")
    except Exception as e:
        print(traceback.format_exc())
    finally:
        save_results_atomic(results, filepath)


def read_results(dataset_folder='unlabeled_10k', prompt_type='discret_prompts',
                 model_name='o3-mini', noise_mode='noisySignal',
                 n_bins=5, top_k=5, prediction_source='dnn',
                 feature_type='stats', n_components=0,
                 ood_train_folder='', use_rag=False, rag_k=0,
                 output_dir='.', prompt_version='v1', backbone='dino'):
    """Read results from JSON file."""
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
        backbone=backbone,
    )
    filepath = os.path.join(output_dir, _output_path(cfg, prompt_type, model_name))
    with open(filepath, 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    DATASET_FOLDER = 'unlabeled_10k'
    PROMPT_TYPE = 'discret_prompts'
    MODEL_NAME = "o3-mini"
    NOISE_MODE = 'noisySignal'
    N_BINS, TOP_K, NUM_TRIES = 10, 5, 3

    main(DATASET_FOLDER, PROMPT_TYPE, MODEL_NAME, NOISE_MODE, N_BINS, TOP_K, NUM_TRIES)
    results = read_results(DATASET_FOLDER, PROMPT_TYPE, MODEL_NAME, NOISE_MODE, N_BINS, TOP_K)
    sorted_results = sort_results_by_prompt(results)
    print(f"Unique prompts: {len(get_unique_prompts(results))}")
    print_metrics(sorted_results, CLASS_NAMES)
