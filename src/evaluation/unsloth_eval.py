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


def load_model_and_tokenizer(
    model_name: str = 'unsloth/gemma-3-27b-it-unsloth-bnb-4bit',
    cache_dir: str = "../../models",
    adapter_path: str | None = None,
):
    """Load an Unsloth model and tokenizer.

    Parameters
    ----------
    model_name
        HuggingFace / Unsloth model identifier.
    cache_dir
        Directory for cached model weights.
    adapter_path
        Optional path to a LoRA adapter directory.  When provided the
        base model is loaded first and then the adapter is applied on
        top, so the finetuned weights are used for inference.
    """
    if adapter_path is not None:
        # ── Finetuned model: load base in 4-bit then apply LoRA ──────
        print(f"Loading base model (4-bit) for adapter: {model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=200000,
            dtype=None,
            load_in_4bit=True,
            cache_dir=cache_dir,
        )
        print(f"Loading LoRA adapter from: {adapter_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        # ── Original path: unchanged behaviour ───────────────────────
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=model_name,
            max_seq_length=200000,
            dtype=None,
            load_in_4bit=False,
            cache_dir=cache_dir,
        )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def _apply_chat_template(prompt: str, tokenizer, chat_template: str) -> str:
    """Convert a raw prompt string to the model's chat format (returns a string)."""
    if chat_template == 'gemma-3':
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    else:
        messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def get_model_response(prompt: str, model, tokenizer,
                       temperature: float = 0.7, chat_template: str = 'gemma-3',
                       max_new_tokens: int = 512) -> str:
    """Get response from a local Unsloth model (single-sample fallback)."""
    text = _apply_chat_template(prompt, tokenizer, chat_template)
    inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=0.95, top_k=64, use_cache=True,
    )
    # Decode only the newly generated tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def get_model_response_batch(
    prompts: List[str],
    model,
    tokenizer,
    temperature: float = 0.7,
    chat_template: str = '',
    max_new_tokens: int = 512,
) -> List[str]:
    """Batched inference — process multiple prompts in one model.generate() call.

    Uses left-padding so all sequences end at the same position, which is
    required for correct auto-regressive generation with padding tokens.
    """
    texts = [_apply_chat_template(p, tokenizer, chat_template) for p in prompts]

    # Ensure pad token exists and use left-padding for generation
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, add_special_tokens=False
    ).to("cuda")
    input_lengths = inputs["attention_mask"].sum(dim=1)  # true (non-pad) length per sample

    tokenizer.padding_side = orig_padding_side  # restore

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=0.95, top_k=64, use_cache=True,
        )

    # Decode only the newly generated tokens for each sample
    responses = []
    for i, out in enumerate(outputs):
        new_tokens = out[input_lengths[i]:]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return responses


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
         output_dir=".", adapter_path=None,
         inference_batch_size: int = 8,
         max_new_tokens: int = 512,
         prompt_version: str = 'v1',
         backbone: str = 'dino'):
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
        data, _, _ = load_data(f'{data_root}/{dataset_folder}', noise_mode, n_bins, top_k,
                               prediction_source=prediction_source,
                               feature_tag=f'emb{n_components}' if feature_type == 'embeddings' else '',
                               backbone=backbone)
        model, tokenizer = load_model_and_tokenizer(model_name, cache_dir=cache_dir, adapter_path=adapter_path)

        all_prompts_data = build_prompts_data(data, prompt_type)
        results, prompts_done, completed = load_existing_results(filepath, num_tries)
        prompts_to_process = get_prompts_to_process(all_prompts_data, completed, prompts_done)

        chat_template = _detect_chat_template(model_name)
        start_tag, end_tag = _get_response_tags(chat_template)

        # Apply chat template to tokenizer once (gemma-3 path only)
        if chat_template == 'gemma-3':
            tokenizer = get_chat_template(tokenizer, chat_template=chat_template)

        # ── Batched inference ─────────────────────────────────────────
        # Flatten (prompt_data, num_done) pairs into individual work items
        work_items = []
        for prompt_data, num_done in prompts_to_process:
            for i in range(num_done, num_tries):
                work_items.append((prompt_data, i))

        print(f"  → {len(work_items)} inference calls, batch_size={inference_batch_size}, "
              f"max_new_tokens={max_new_tokens}")

        for batch_start in tqdm(range(0, len(work_items), inference_batch_size),
                                desc="Batched inference"):
            batch = work_items[batch_start: batch_start + inference_batch_size]
            prompts_batch = [item[0]['prompt'] for item in batch]
            try:
                responses = get_model_response_batch(
                    prompts_batch, model=model, tokenizer=tokenizer,
                    temperature=0.2, chat_template=chat_template,
                    max_new_tokens=max_new_tokens,
                )
                for (prompt_data, i), raw_response in zip(batch, responses):
                    if not raw_response:
                        print(f"  Empty response for {prompt_data['filename']}, skipping.")
                        continue
                    results.append(build_result_entry(
                        prompt_data['filename'], i, prompt_data['prompt'],
                        raw_response, start_tag, end_tag,
                    ))
            except Exception as e:
                print(f"Batch error at index {batch_start}: {e}")
                print(traceback.format_exc())
                # Fall back to single-sample inference for this batch
                for prompt_data, i in batch:
                    try:
                        raw_response = get_model_response(
                            prompt_data['prompt'], model=model, tokenizer=tokenizer,
                            temperature=0.2, chat_template=chat_template,
                            max_new_tokens=max_new_tokens,
                        )
                        if raw_response:
                            results.append(build_result_entry(
                                prompt_data['filename'], i, prompt_data['prompt'],
                                raw_response, start_tag, end_tag,
                            ))
                    except Exception as inner_e:
                        print(f"  Fallback failed for {prompt_data['filename']}: {inner_e}")

    except KeyboardInterrupt:
        print("\nInterrupted. Saving partial results...")
    except Exception as e:
        print(traceback.format_exc())
    finally:
        save_results_atomic(results, filepath)


def read_results(dataset_folder, prompt_type, model_name, noise_mode, n_bins, top_k,
                 prediction_source='dnn', feature_type='stats', n_components=0,
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
