"""
Dataset builder for QLoRA finetuning.

Converts existing train ``.pkl`` files (from ``src.prompt.generated_dataset``)
into Hugging Face ``Dataset`` objects formatted for SFT training.

**Key principle**: No few-shot context — each sample is a single
``(instruction + signal features → label)`` pair so the model learns the
mapping in its weights, not from in-context examples.

Usage
-----
Preview training samples from a pkl::

    python -m src.finetuning.dataset \
        --pkl_path data/own/unlabeled_10k/train_centroid_noisySignal_5_5_data.pkl \
        --prompt_style discret --num_preview 3
"""

from __future__ import annotations

import argparse
import os
import pickle
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset


# ── Prompt templates (zero-shot, no context) ─────────────────────────────────

FINETUNE_SYSTEM_PROMPT = (
    "You are an expert AI signal classifier specialising in wireless "
    "communication modulation schemes."
)

FINETUNE_INSTRUCTION_DISCRET = textwrap.dedent("""\
    **OBJECTIVE:**
    Classify the modulation scheme of a wireless signal based on its discretised statistical features.

    **CONTEXT:**
    The classification is based on the principle that moments and cumulants of a signal's I/Q components create a unique feature set. The provided statistics are the features extracted from a signal after being processed by a `KBinsDiscretizer`, which quantifies the signal's distribution. Map these statistical features to the most likely modulation scheme.

    **Signal Statistics:** {features}
    **Classification Options:** {options}

    **RESPONSE RULES:**
    1. You **MUST** first use `<think>` tags to detail your step-by-step reasoning.
    2. After the closing `</think>` tag, provide **only** the final classification.
    3. The output **MUST** be a single entry from the classification options.
    4. No additional words, explanations, or introductory text.

    Classify the signal based on the provided data and options, following all rules above.""")

FINETUNE_INSTRUCTION_CONTINUOUS = textwrap.dedent("""\
    **OBJECTIVE:**
    Classify the modulation scheme of a wireless signal based on its statistical features.

    **CONTEXT:**
    The classification is based on the principle that moments and cumulants of a signal's I/Q components create a unique feature set. The provided statistics are scaled continuous features of a signal. Map these statistical features to the most likely modulation scheme.

    **Signal Statistics:** {features}
    **Classification Options:** {options}

    **RESPONSE RULES:**
    1. You **MUST** first use `<think>` tags to detail your step-by-step reasoning.
    2. After the closing `</think>` tag, provide **only** the final classification.
    3. The output **MUST** be a single entry from the classification options.
    4. No additional words, explanations, or introductory text.

    Classify the signal based on the provided data and options, following all rules above.""")

FINETUNE_INSTRUCTION_DISCRET_NO_THINK = textwrap.dedent("""\
    **OBJECTIVE:**
    Classify the modulation scheme of a wireless signal based on its discretised statistical features.

    **CONTEXT:**
    The classification is based on the principle that moments and cumulants of a signal's I/Q components create a unique feature set. The provided statistics are the features extracted from a signal after being processed by a `KBinsDiscretizer`, which quantifies the signal's distribution.

    **Signal Statistics:** {features}
    **Classification Options:** {options}

    Respond with **only** the modulation class name. No other text.""")

FINETUNE_INSTRUCTION_CONTINUOUS_NO_THINK = textwrap.dedent("""\
    **OBJECTIVE:**
    Classify the modulation scheme of a wireless signal based on its statistical features.

    **CONTEXT:**
    The classification is based on the principle that moments and cumulants of a signal's I/Q components create a unique feature set. The provided statistics are scaled continuous features of a signal.

    **Signal Statistics:** {features}
    **Classification Options:** {options}

    Respond with **only** the modulation class name. No other text.""")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _to_base26(n: int) -> str:
    """Convert integer to letter code: 0→A, 1→B, …, 25→Z, 26→AA, …"""
    result = ""
    while True:
        result = chr(ord("A") + n % 26) + result
        n = n // 26 - 1
        if n < 0:
            break
    return result


def _format_feature_dict(
    feature_dict: dict,
    discretized: bool,
    decimal_precision: int = 3,
) -> str:
    """Format a feature dictionary into a human-readable string."""
    parts: List[str] = []
    for key, value in feature_dict.items():
        if discretized:
            # Discretized features → letter codes
            if hasattr(value, "__iter__"):
                letters = ", ".join(_to_base26(int(v)) for v in value)
                parts.append(f"{key}: [{letters}]")
            else:
                parts.append(f"{key}: {_to_base26(int(value))}")
        else:
            # Continuous features → rounded decimals
            if hasattr(value, "__iter__"):
                vals = ", ".join(f"{float(v):.{decimal_precision}f}" for v in value)
                parts.append(f"{key}: [{vals}]")
            else:
                parts.append(f"{key}: {float(value):.{decimal_precision}f}")
    return ", ".join(parts)


def _format_options(classes: List[str]) -> str:
    """Format class names as ``[A: cls1, B: cls2, ...]``."""
    items = [f"{_to_base26(i)}: {c}" for i, c in enumerate(classes)]
    return "[" + ", ".join(items) + "]"


# ── v2 reasoning: feature-aware <think> completions ──────────────────────────

# Which feature keys are "higher-order" discriminative cumulants
_DISCRIMINATIVE_FEATURES = {"kstat_3", "kstat_4", "kurtosis", "skewness", "variance"}

# Brief domain knowledge per modulation class (used to ground the reasoning)
_MOD_HINTS: Dict[str, str] = {
    "4ASK":  "4-level ASK exhibits moderate variance with low kurtosis and near-zero skewness due to its symmetric amplitude levels",
    "4PAM":  "4-PAM shows distinct amplitude spacing reflected in higher variance and characteristic moment patterns",
    "8ASK":  "8-ASK has finer amplitude granularity leading to elevated variance and broader moment spread",
    "16PAM": "16-PAM produces the widest amplitude range, resulting in high variance and distinctive higher-order cumulants",
    "CPFSK": "CPFSK is a constant-envelope FM scheme with near-zero kurtosis excess and low skewness",
    "DQPSK": "DQPSK has constant envelope and phase transitions that yield near-zero variance in amplitude with moderate kstat values",
    "GFSK":  "GFSK has a Gaussian-shaped frequency pulse, producing smooth spectral features and characteristic low kstat_3",
    "GMSK":  "GMSK is minimum-shift with Gaussian filtering, resulting in near-constant envelope and distinctive cumulant ratios",
    "OQPSK": "OQPSK offsets I/Q by half a symbol, producing reduced amplitude fluctuations and specific kstat_4 behaviour",
    "OOK":   "OOK is binary amplitude keying with high skewness (asymmetric on/off) and distinctively large kurtosis",
}


def _build_v2_reasoning(
    feature_dict: dict,
    label: str,
    discretized: bool,
) -> str:
    """Build a feature-aware reasoning string for the ``<think>`` block.

    Instead of a generic one-liner, this references the actual
    discriminative features and domain knowledge about the modulation.
    """
    # Identify which discriminative features are present in this sample
    present_disc = [k for k in feature_dict if k in _DISCRIMINATIVE_FEATURES]

    # Summarise the most informative features
    if discretized:
        feat_summary_parts = []
        for k in present_disc[:3]:  # top-3 discriminative features
            v = feature_dict[k]
            if hasattr(v, "__iter__"):
                letters = ", ".join(_to_base26(int(x)) for x in v)
                feat_summary_parts.append(f"{k}=[{letters}]")
            else:
                feat_summary_parts.append(f"{k}={_to_base26(int(v))}")
    else:
        feat_summary_parts = []
        for k in present_disc[:3]:
            v = feature_dict[k]
            if hasattr(v, "__iter__"):
                vals = ", ".join(f"{float(x):.3f}" for x in v)
                feat_summary_parts.append(f"{k}=[{vals}]")
            else:
                feat_summary_parts.append(f"{k}={float(v):.3f}")

    feat_summary = ", ".join(feat_summary_parts) if feat_summary_parts else "the overall feature pattern"

    # Domain hint for the label
    hint = _MOD_HINTS.get(label, f"{label} has a characteristic statistical fingerprint")

    lines = [
        f"Examining the key discriminative features: {feat_summary}.",
        f"Recall that {hint}.",
        f"The observed feature values are consistent with {label} modulation.",
    ]
    return "\n".join(lines)


# ── Core dataset builder ────────────────────────────────────────────────────

def load_train_pkl(path: str) -> Dict[str, Any]:
    """Load a train pkl produced by ``src.prompt.generated_dataset``."""
    with open(path, "rb") as f:
        return pickle.load(f)


def build_sft_samples(
    data: Dict[str, Any],
    prompt_style: str = "discret",
    use_thinking: bool = True,
    decimal_precision: int = 3,
    completion_version: str = "v1",
) -> List[Dict[str, str]]:
    """Build zero-shot SFT training samples from pkl data.

    Each sample is a dict with ``"prompt"`` and ``"completion"`` keys.
    No few-shot context is included.

    Parameters
    ----------
    data
        Loaded ``.pkl`` dict (must have ``stats``, ``discret_stats``,
        ``labels``, ``feature_names``).
    prompt_style
        ``"discret"`` → letter-coded discretised features,
        ``"continuous"`` → scaled decimal features.
    use_thinking
        If True, the completion contains a ``<think>...</think>`` stub
        followed by the label.  If False, completion is the label only.
    decimal_precision
        Decimal places for continuous features.
    completion_version
        ``"v1"`` → generic one-liner reasoning,
        ``"v2"`` → feature-aware reasoning that references discriminative
        features and domain knowledge about the modulation class.

    Returns
    -------
    list[dict]
        Each dict: ``{"prompt": ..., "completion": ...}``.
    """
    discretized = prompt_style == "discret"
    feature_key = "discret_stats" if discretized else "stats"
    feature_dicts: list = data[feature_key]
    labels: list = data["labels"]

    # Build consistent class list
    classes = sorted(set(labels))
    options_str = _format_options(classes)

    # Pick template
    if use_thinking:
        template = (
            FINETUNE_INSTRUCTION_DISCRET if discretized
            else FINETUNE_INSTRUCTION_CONTINUOUS
        )
    else:
        template = (
            FINETUNE_INSTRUCTION_DISCRET_NO_THINK if discretized
            else FINETUNE_INSTRUCTION_CONTINUOUS_NO_THINK
        )

    samples: List[Dict[str, str]] = []
    for feat_dict, label in zip(feature_dicts, labels):
        features_str = _format_feature_dict(
            feat_dict, discretized=discretized,
            decimal_precision=decimal_precision,
        )
        prompt = template.format(features=features_str, options=options_str)

        if use_thinking:
            if completion_version == "v2":
                reasoning = _build_v2_reasoning(feat_dict, label, discretized)
                completion = (
                    f"<think>\n{reasoning}\n</think>\n{label}"
                )
            else:
                completion = (
                    f"<think>\nBased on the signal statistics, the features point "
                    f"towards {label} modulation.\n</think>\n{label}"
                )
        else:
            completion = label

        samples.append({"prompt": prompt, "completion": completion})

    return samples


def create_hf_dataset(
    data: Dict[str, Any],
    prompt_style: str = "discret",
    use_thinking: bool = True,
    decimal_precision: int = 3,
    chat_template: str = "auto",
    completion_version: str = "v1",
) -> Dataset:
    """Build a Hugging Face Dataset formatted as chat conversations.

    Each row has a ``"conversations"`` column containing a list of
    ``{"role": ..., "content": ...}`` dicts suitable for SFTTrainer with
    chat template formatting.

    Parameters
    ----------
    data
        Loaded train ``.pkl`` dict.
    prompt_style
        ``"discret"`` or ``"continuous"``.
    use_thinking
        Whether to include ``<think>`` reasoning in assistant response.
    decimal_precision
        Decimal places for continuous features.
    chat_template
        Chat template name hint (unused here, consumed by train.py).
    completion_version
        ``"v1"`` (generic reasoning) or ``"v2"`` (feature-aware reasoning).

    Returns
    -------
    datasets.Dataset
    """
    samples = build_sft_samples(
        data,
        prompt_style=prompt_style,
        use_thinking=use_thinking,
        decimal_precision=decimal_precision,
        completion_version=completion_version,
    )

    conversations = []
    for s in samples:
        conversations.append([
            {"role": "system", "content": FINETUNE_SYSTEM_PROMPT},
            {"role": "user", "content": s["prompt"]},
            {"role": "assistant", "content": s["completion"]},
        ])

    return Dataset.from_dict({"conversations": conversations})


# ── CLI preview ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preview / export QLoRA finetuning dataset from a train pkl.",
    )
    parser.add_argument(
        "--pkl_path", type=str, required=True,
        help="Path to a train .pkl file from generated_dataset.",
    )
    parser.add_argument(
        "--prompt_style", type=str, default="discret",
        choices=["discret", "continuous"],
        help="Feature representation in prompts (default: discret).",
    )
    parser.add_argument(
        "--use_thinking", action="store_true", default=True,
        help="Include <think> reasoning in completions (default: True).",
    )
    parser.add_argument(
        "--no_thinking", action="store_true", default=False,
        help="Disable <think> reasoning in completions.",
    )
    parser.add_argument(
        "--num_preview", type=int, default=3,
        help="Number of samples to preview (default: 3).",
    )
    parser.add_argument(
        "--export_jsonl", type=str, default=None,
        help="If set, export full dataset to this JSONL path.",
    )
    parser.add_argument(
        "--completion_version", type=str, default="v1",
        choices=["v1", "v2"],
        help="Completion reasoning version: v1 (generic) or v2 (feature-aware) (default: v1).",
    )

    args = parser.parse_args()
    use_thinking = not args.no_thinking

    print(f"Loading pkl: {args.pkl_path}")
    data = load_train_pkl(args.pkl_path)
    print(f"  → {data['num_samples']} samples, {data['#classes']} classes")

    samples = build_sft_samples(
        data,
        prompt_style=args.prompt_style,
        use_thinking=use_thinking,
        completion_version=args.completion_version,
    )
    print(f"  → Built {len(samples)} SFT samples")

    # Preview
    print("\n" + "=" * 70)
    for i, s in enumerate(samples[: args.num_preview]):
        print(f"\n{'─' * 60}")
        print(f"SAMPLE {i + 1}")
        print(f"{'─' * 60}")
        print(f"\n[USER]\n{s['prompt']}")
        print(f"\n[ASSISTANT]\n{s['completion']}")
    print("\n" + "=" * 70)

    # Export
    if args.export_jsonl:
        import json

        ds = create_hf_dataset(
            data,
            prompt_style=args.prompt_style,
            use_thinking=use_thinking,
            completion_version=args.completion_version,
        )
        with open(args.export_jsonl, "w") as f:
            for row in ds:
                f.write(json.dumps(row) + "\n")
        print(f"Exported {len(ds)} rows → {args.export_jsonl}")


if __name__ == "__main__":
    main()
