"""
QLoRA finetuning script for DiSC-AMC using Unsloth + TRL SFTTrainer.

Loads a pre-quantised base model, applies LoRA adapters, and finetunes on
zero-shot signal classification samples built from the existing train
``.pkl`` data.

Usage
-----
::

    python -m src.finetuning.train \
        --model_name unsloth/DeepSeek-R1-Distill-Qwen-7B \
        --pkl_path data/own/unlabeled_10k/train_centroid_noisySignal_5_5_data.pkl \
        --output_dir exp/qlora_deepseek7b_discret \
        --prompt_style discret \
        --epochs 3 --batch_size 2 --lr 2e-4
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from trl import SFTTrainer, SFTConfig

from src.finetuning.dataset import create_hf_dataset, load_train_pkl


# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0
DEFAULT_LR = 2e-4
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRAD_ACCUM = 4
DEFAULT_MAX_SEQ_LEN = 2048
DEFAULT_WARMUP_STEPS = 5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_SEED = 3407

# Modules targeted by LoRA adapters (covers attention + MLP for most LLMs)
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ── Chat template detection ─────────────────────────────────────────────────

def _detect_chat_template(model_name: str) -> str:
    """Heuristic to pick the right chat template for a model."""
    name_lower = model_name.lower()
    if "qwen" in name_lower or "deepseek" in name_lower:
        return "qwen-2.5"
    if "gemma" in name_lower:
        return "gemma-3"
    if "llama" in name_lower:
        return "llama-3.1"
    if "mistral" in name_lower:
        return "mistral"
    if "phi" in name_lower:
        return "phi-4"
    # Fallback — let Unsloth auto-detect
    return "chatml"


# ── Main training function ──────────────────────────────────────────────────

def train(
    model_name: str,
    pkl_path: str,
    output_dir: str,
    prompt_style: str = "discret",
    use_thinking: bool = True,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    lr: float = DEFAULT_LR,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    gradient_accumulation_steps: int = DEFAULT_GRAD_ACCUM,
    max_seq_length: int = DEFAULT_MAX_SEQ_LEN,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    seed: int = DEFAULT_SEED,
    cache_dir: str = "../../models",
    save_merged: bool = False,
):
    """Run QLoRA finetuning.

    Parameters
    ----------
    model_name
        Unsloth model identifier (e.g. ``unsloth/DeepSeek-R1-Distill-Qwen-7B``).
    pkl_path
        Path to the train ``.pkl`` from ``src.prompt.generated_dataset``.
    output_dir
        Directory to save adapter weights (and optionally merged model).
    prompt_style
        ``"discret"`` or ``"continuous"`` feature representation.
    use_thinking
        Whether to include ``<think>`` reasoning in training targets.
    save_merged
        If True, also save a merged 16-bit model alongside the adapter.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load base model (4-bit quantised) ─────────────────────────────
    print(f"Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,            # auto-detect
        load_in_4bit=True,     # QLoRA
        cache_dir=cache_dir,
    )

    # ── 2. Apply LoRA adapters ───────────────────────────────────────────
    print(f"Applying LoRA (r={lora_r}, alpha={lora_alpha}) to: {LORA_TARGET_MODULES}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",  # memory-efficient
        random_state=seed,
    )

    # ── 3. Set chat template ─────────────────────────────────────────────
    chat_template = _detect_chat_template(model_name)
    print(f"Chat template: {chat_template}")
    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)

    # ── 4. Build dataset ─────────────────────────────────────────────────
    print(f"Loading training data: {pkl_path}")
    data = load_train_pkl(pkl_path)
    print(f"  → {data['num_samples']} samples, {data['#classes']} classes")

    dataset = create_hf_dataset(
        data,
        prompt_style=prompt_style,
        use_thinking=use_thinking,
    )
    dataset = standardize_sharegpt(dataset)
    print(f"  → Dataset: {len(dataset)} rows")

    # ── 5. Formatting function ───────────────────────────────────────────
    def formatting_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False,
            )
            for convo in convos
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_func, batched=True)

    # ── 6. Training config ───────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        seed=seed,
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        report_to="none",       # set to "wandb" if you want W&B logging
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
    )

    # ── 7. Train ─────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    print("\n" + "=" * 60)
    print("  Starting QLoRA finetuning")
    print("=" * 60)
    trainer_stats = trainer.train()

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Total steps:    {trainer_stats.global_step}")
    print(f"  Training loss:  {trainer_stats.training_loss:.4f}")
    print("=" * 60)

    # ── 8. Save ──────────────────────────────────────────────────────────
    # Save LoRA adapter
    adapter_dir = os.path.join(output_dir, "lora_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"LoRA adapter saved → {adapter_dir}")

    # Optionally merge and save full model
    if save_merged:
        merged_dir = os.path.join(output_dir, "merged_model")
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        print(f"Merged model saved → {merged_dir}")

    return trainer_stats


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="QLoRA finetune an LLM for signal classification (DiSC-AMC).",
    )

    # Required
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Unsloth model name (e.g. unsloth/DeepSeek-R1-Distill-Qwen-7B).",
    )
    parser.add_argument(
        "--pkl_path", type=str, required=True,
        help="Path to a train .pkl from generated_dataset.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save adapter weights and training logs.",
    )

    # Data
    parser.add_argument(
        "--prompt_style", type=str, default="discret",
        choices=["discret", "continuous"],
        help="Feature representation in prompts (default: discret).",
    )
    parser.add_argument(
        "--no_thinking", action="store_true", default=False,
        help="Disable <think> reasoning in training completions.",
    )

    # LoRA
    parser.add_argument("--lora_r", type=int, default=DEFAULT_LORA_R,
                        help=f"LoRA rank (default: {DEFAULT_LORA_R}).")
    parser.add_argument("--lora_alpha", type=int, default=DEFAULT_LORA_ALPHA,
                        help=f"LoRA alpha (default: {DEFAULT_LORA_ALPHA}).")

    # Training
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of training epochs (default: {DEFAULT_EPOCHS}).")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Per-device batch size (default: {DEFAULT_BATCH_SIZE}).")
    parser.add_argument("--gradient_accumulation_steps", type=int,
                        default=DEFAULT_GRAD_ACCUM,
                        help=f"Gradient accumulation steps (default: {DEFAULT_GRAD_ACCUM}).")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help=f"Learning rate (default: {DEFAULT_LR}).")
    parser.add_argument("--max_seq_length", type=int, default=DEFAULT_MAX_SEQ_LEN,
                        help=f"Max sequence length (default: {DEFAULT_MAX_SEQ_LEN}).")
    parser.add_argument("--warmup_steps", type=int, default=DEFAULT_WARMUP_STEPS,
                        help=f"Warmup steps (default: {DEFAULT_WARMUP_STEPS}).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default: {DEFAULT_SEED}).")

    # Paths
    parser.add_argument("--cache_dir", type=str, default="../../models",
                        help="Model cache directory (default: ../../models).")

    # Output
    parser.add_argument("--save_merged", action="store_true", default=False,
                        help="Also save merged 16-bit model (large!).")

    args = parser.parse_args()

    train(
        model_name=args.model_name,
        pkl_path=args.pkl_path,
        output_dir=args.output_dir,
        prompt_style=args.prompt_style,
        use_thinking=not args.no_thinking,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_seq_length,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        cache_dir=args.cache_dir,
        save_merged=args.save_merged,
    )


if __name__ == "__main__":
    main()
