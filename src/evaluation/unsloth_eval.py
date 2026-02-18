"""Unsloth local model evaluation provider for modulation classification."""

import os
import sys
import traceback
from tqdm import tqdm
from typing import Dict, List

import torch
from unsloth import FastLanguageModel, FastVisionModel
from unsloth.chat_templates import get_chat_template

from src.naming import ExperimentConfig, eval_result_name

from .utils import (
    extract_tag, load_data, build_prompts_data, load_existing_results,
    get_prompts_to_process, save_results_atomic, build_result_entry,
    sort_results_by_prompt, get_unique_prompts, print_metrics,
)

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

CLASS_NAMES = ['4ASK', '4PAM', '8ASK', '16PAM', 'CPFSK', 'DQPSK', 'GFSK', 'GMSK', 'OQPSK', 'OOK']


def load_model_and_tokenizer(model_name: str = 'unsloth/gemma-3-27b-it-unsloth-bnb-4bit', cache_dir="../../models"):
    """Load an Unsloth model and tokenizer."""
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_name,
        max_seq_length=200000,
        dtype=None,
        load_in_4bit=False,
        cache_dir=cache_dir,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def get_model_response(prompt: str, model, tokenizer,
                       temperature: float = 0.7, chat_template: str = 'gemma-3') -> str:
    """Get response from a local Unsloth model."""
    if chat_template == 'gemma-3':
        tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").to("cuda")
    else:
        messages = [{"role": "user", "content": prompt}]
        text = prompt
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to("cuda")

    outputs = model.generate(
        **inputs, max_new_tokens=3000,
        temperature=temperature, top_p=0.95, top_k=64, use_cache=True,
    )
    raw_decoded = tokenizer.batch_decode(outputs)
    return raw_decoded[0][len(text) - 41:]


def _detect_chat_template(model_name: str) -> str:
    if 'gemma-3' in model_name:
        return 'gemma-3'
    elif 'gpt' in model_name:
        return 'gpt'
    return ''


def _get_response_tags(chat_template: str):
    if chat_template == 'gpt':
        return "<|message|>", "<|message|>"
    return "<think>", "</think>"


def _output_path(cfg: ExperimentConfig, prompt_type: str, model_name: str) -> str:
    clean_name = model_name.replace('unsloth/', '')
    return eval_result_name(cfg, prompt_type, clean_name, "custom")


def main(dataset_folder, prompt_type='discret_prompts',
         model_name="gemini-2.5-flash", noise_mode='noisySignal',
         n_bins=10, top_k=5, num_tries=3, prediction_source='dnn',
         feature_type='stats', n_components=0,
         ood_train_folder='', use_rag=False, rag_k=0,
         cache_dir="../../models", data_root="../../data/own",
         output_dir="."):
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
        )
    results = []
    filepath = os.path.join(output_dir, _output_path(cfg, prompt_type, model_name))

    try:
        data, _, _ = load_data(f'{data_root}/{dataset_folder}', noise_mode, n_bins, top_k,
                               prediction_source=prediction_source,
                               feature_tag=f'emb{n_components}' if feature_type == 'embeddings' else '')
        model, tokenizer = load_model_and_tokenizer(model_name, cache_dir=cache_dir)

        all_prompts_data = build_prompts_data(data, prompt_type)
        results, prompts_done, completed = load_existing_results(filepath, num_tries)
        prompts_to_process = get_prompts_to_process(all_prompts_data, completed, prompts_done)

        chat_template = _detect_chat_template(model_name)
        start_tag, end_tag = _get_response_tags(chat_template)

        for prompt_data, num_done in tqdm(prompts_to_process):
            try:
                prompt, filename = prompt_data['prompt'], prompt_data['filename']
                for i in range(num_done, num_tries):
                    raw_response = get_model_response(
                        prompt, model=model, tokenizer=tokenizer,
                        temperature=0.2, chat_template=chat_template,
                    )
                    if not raw_response:
                        raise Exception("Empty response, stopping")
                    results.append(build_result_entry(filename, i, prompt, raw_response, start_tag, end_tag))
            except Exception as e:
                if "stopping" in str(e).lower():
                    print(str(e))
                    break
                print(f"Error processing {prompt_data['filename']}: {e}")
                print(traceback.format_exc())

    except KeyboardInterrupt:
        print("\nInterrupted. Saving partial results...")
    except Exception as e:
        print(traceback.format_exc())
    finally:
        save_results_atomic(results, filepath)


def read_results(dataset_folder, prompt_type, model_name, noise_mode, n_bins, top_k,
                 prediction_source='dnn', feature_type='stats', n_components=0,
                 ood_train_folder='', use_rag=False, rag_k=0,
                 output_dir='.'):
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
    )
    filepath = os.path.join(output_dir, _output_path(cfg, prompt_type, model_name))
    with open(filepath, 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    DATASET_FOLDER = '-30dB'
    PROMPT_TYPE = 'discret_prompts'
    MODEL_NAME = 'unsloth/DeepSeek-R1-Distill-Qwen-32B-unsloth-bnb-4bit'
    NOISE_MODE = 'noisySignal'
    N_BINS, TOP_K, NUM_TRIES = 5, 5, 1

    main(DATASET_FOLDER, PROMPT_TYPE, MODEL_NAME, NOISE_MODE, N_BINS, TOP_K, NUM_TRIES)
    results = read_results(DATASET_FOLDER, PROMPT_TYPE, MODEL_NAME, NOISE_MODE, N_BINS, TOP_K)
    sorted_results = sort_results_by_prompt(results)
    print(f"Unique prompts: {len(get_unique_prompts(results))}")
    print_metrics(sorted_results, CLASS_NAMES)
