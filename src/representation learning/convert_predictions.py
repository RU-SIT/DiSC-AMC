"""
Convert prediction JSON keys from ``.png`` filenames to ``.npy`` filenames.

The classifier-based and centroid-based prediction pipelines (``inference.py``)
produce JSON files keyed by ``.png`` constellation image filenames.  The prompt
generation pipeline (``generated_dataset.py``) works with ``.npy`` signal files.
This script bridges the two by rewriting the JSON keys so that every
``<name>.png`` key becomes ``<name>.npy``.

Usage
-----
::

    cd src/representation\\ learning/

    # Convert classifier-head predictions
    python convert_predictions.py \\
        --input ../../data/own/unlabeled_10k/top5_dnn_predictions.json \\
        --output ../../data/own/unlabeled_10k/ntop5_dnn_predictions.json

    # Convert centroid-based predictions
    python convert_predictions.py \\
        --input ../../data/own/unlabeled_10k/top5_centroid_predictions.json \\
        --output ../../data/own/unlabeled_10k/ntop5_centroid_predictions.json

Input format  (from ``inference.py``)::

    {
        "4ASK_-5.57dB__076_20250127_145624.png": {
            "noisyImg": ["4ASK", "8ASK", "OOK", "CPFSK", "4PAM"],
            "noiseLessImg": ["4ASK", "8ASK", "OOK", "4PAM", "CPFSK"]
        },
        ...
    }

Output format (for ``generated_dataset.py``)::

    {
        "4ASK_-5.57dB__076_20250127_145624.npy": {
            "noisyImg": ["4ASK", "8ASK", "OOK", "CPFSK", "4PAM"],
            "noiseLessImg": ["4ASK", "8ASK", "OOK", "4PAM", "CPFSK"]
        },
        ...
    }
"""

from __future__ import annotations

import argparse
import json
import os


def convert_png_to_npy_keys(data: dict) -> dict:
    """Replace ``.png`` with ``.npy`` in all top-level keys of *data*.

    Keys that do not end with ``.png`` are kept unchanged.
    """
    converted = {}
    for key, value in data.items():
        new_key = key.replace(".png", ".npy") if key.endswith(".png") else key
        converted[new_key] = value
    return converted


def main(args: argparse.Namespace) -> None:
    """Read input JSON, convert keys, write output JSON."""
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    with open(args.input, "r") as fh:
        data = json.load(fh)

    original_count = len(data)
    converted = convert_png_to_npy_keys(data)
    renamed_count = sum(
        1 for old, new in zip(data.keys(), converted.keys()) if old != new
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(converted, fh, indent=4)

    print(f"Converted {renamed_count}/{original_count} keys (.png → .npy)")
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert .png keys to .npy in a predictions JSON file.",
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to the input predictions JSON (with .png keys).",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to the output predictions JSON (with .npy keys).",
    )

    args = parser.parse_args()
    main(args)
