"""
Compute per-class centroids from a trained classifier's encoder features.

This script loads a trained ImageClassifier, extracts encoder features for
every sample in a training dataset, computes the mean feature vector (centroid)
for each class, and saves the result as a JSON file.

Usage
-----
::

    cd src/representation\\ learning/

    python compute_centroids.py \\
        --backbone dino \\
        --weights ../../exp/dino_classifier.pth \\
        --dataset_path ../../data/own/unlabeled_10k/train \\
        --output ../../data/own/unlabeled_10k/train/class_centers.json

The output JSON maps each class name to its mean feature vector and can be
consumed by ``inference.py predict --centroid_path``.
"""

from __future__ import annotations

import argparse
import os

from torchvision import transforms

from .classifier_training import ImageClassifier
from .data_loader import DatasetWithPath
from .embedding_pipeline import compute_class_centroids, save_centroids, find_closest_to_centroids

# ── Constants ────────────────────────────────────────────────────────────────

CLASSES = [
    "OOK", "4ASK", "8ASK", "OQPSK", "CPFSK",
    "GFSK", "4PAM", "DQPSK", "16PAM", "GMSK",
]


def main(args: argparse.Namespace) -> None:
    """Load model, extract features, compute centroids, save JSON."""
    from inference import load_classifier  # local import to avoid circular

    classes = args.classes.split(",")

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    # 1. Load model
    model, device = load_classifier(
        backbone=args.backbone,
        weights_path=args.weights,
        num_classes=len(classes),
    )

    # 2. Prepare dataset (with paths)
    dataset = DatasetWithPath(
        dataset_path=args.dataset_path,
        classes=classes,
        transform=transform,
    )
    print(f"Dataset: {len(dataset)} samples from {args.dataset_path}")

    # 3. Compute centroids
    centroids = compute_class_centroids(
        encoder=model.encoder,
        device=device,
        dataset=dataset,
        classes=classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Computed centroids for {len(centroids)} classes: {list(centroids.keys())}")

    # 4. Save
    save_centroids(centroids, args.output)

    # 5. Optionally find closest samples
    if args.find_closest:
        closest = find_closest_to_centroids(
            encoder=model.encoder,
            device=device,
            dataset=dataset,
            classes=classes,
            centroids=centroids,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print("\nClosest sample to each class centroid:")
        for cls_name, path in closest.items():
            print(f"  {cls_name}: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-class centroids from a trained classifier.",
    )
    parser.add_argument(
        "--backbone", type=str, default="dino", choices=["dino", "resnet"],
        help="Encoder backbone type (default: dino).",
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to trained classifier weights (.pth).",
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Path to training data directory (must contain noisyImg/ and/or noiseLessImg/).",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for the centroids JSON file.",
    )
    parser.add_argument(
        "--classes", type=str,
        default=",".join(CLASSES),
        help="Comma-separated class names (default: %(default)s).",
    )
    parser.add_argument(
        "--image_size", type=int, default=96,
        help="Image resize dimension (default: 96).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for feature extraction (default: 32).",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of data loader workers (default: 4).",
    )
    parser.add_argument(
        "--find_closest", action="store_true",
        help="Also print the closest sample to each centroid.",
    )

    args = parser.parse_args()
    main(args)
