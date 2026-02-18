"""
QLoRA finetuning module for DiSC-AMC.

Provides tools to convert existing train `.pkl` files into Hugging Face
Datasets (without few-shot context) and finetune LLMs via Unsloth + TRL
SFTTrainer.
"""
