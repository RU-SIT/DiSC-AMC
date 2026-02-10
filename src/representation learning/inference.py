"""
Inference & evaluation utilities for trained classifiers.

Provides three capabilities that were previously ad-hoc cells in
``classify.py``:

1. **Test-set evaluation** with both classifier-head accuracy and
   centroid-based accuracy (``evaluate_test_set``).
2. **Per-image top-k predictions** using the classifier head
   (``predict_topk_classifier``).
3. **Per-image top-k predictions** using centroid distances
   (``predict_topk_centroids``).

All functions are parameterised so they work with any backbone supported
by :class:`classifier_training.ImageClassifier`.
"""

from __future__ import annotations

import json
import os
from glob import glob
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from classifier_training import ImageClassifier, topk_accuracy, topk_centroid_accuracy
from data_loader import ConstilationDataset
from embedding_pipeline import load_centroids

# ── Constants ────────────────────────────────────────────────────────────────

CLASSES: List[str] = [
    "OOK", "4ASK", "8ASK", "OQPSK", "CPFSK",
    "GFSK", "4PAM", "DQPSK", "16PAM", "GMSK",
]

_DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
])

# ── Model loading helper ────────────────────────────────────────────────────


def load_classifier(
    backbone: str,
    weights_path: str,
    num_classes: int,
    freeze_encoder: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[ImageClassifier, torch.device]:
    """Load a trained :class:`ImageClassifier` and set to eval mode.

    Returns ``(model, device)``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ImageClassifier(
        backbone=backbone,
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
    )
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, device


# ── Test-set evaluation ──────────────────────────────────────────────────────


def evaluate_test_set(
    model: ImageClassifier,
    device: torch.device,
    test_dataset_path: str,
    classes: List[str] = CLASSES,
    centroid_path: Optional[str] = None,
    topk: int = 5,
    batch_size: int = 32,
    num_workers: int = 4,
    transform: transforms.Compose = _DEFAULT_TRANSFORM,
) -> Dict[str, float]:
    """Evaluate a trained classifier on a test dataset.

    Computes top-1 and top-k accuracy using the classifier head.  If
    *centroid_path* is provided, also computes centroid-based top-k accuracy.

    Parameters
    ----------
    model
        Trained :class:`ImageClassifier` (already on *device*, eval mode).
    device
        Torch device.
    test_dataset_path
        Path containing ``noisyImg/`` and ``noiseLessImg/`` sub-directories.
    classes
        Ordered class names matching label indices.
    centroid_path
        Optional path to a centroids JSON (see :func:`embedding_pipeline.save_centroids`).
    topk
        ``k`` for top-k accuracy.
    batch_size, num_workers
        DataLoader parameters.
    transform
        Image transform pipeline.

    Returns
    -------
    metrics : dict
        Keys: ``'top1_accuracy'``, ``'topk_accuracy'``, and optionally
        ``'centroid_topk_accuracy'``.
    """
    dataset = ConstilationDataset(
        dataset_path=test_dataset_path,
        classes=classes,
        transform=transform,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)

    centroid_tensor = None
    if centroid_path is not None:
        centroid_tensor, centroid_class_names = load_centroids(centroid_path, device)
        # Re-order centroid tensor rows to match `classes` ordering
        reorder_idx = [centroid_class_names.index(c) for c in classes]
        centroid_tensor = centroid_tensor[reorder_idx]

    correct = 0
    topk_correct = 0
    centroid_topk_correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            features = model.encoder(images)

            # Classifier-head predictions
            if model.backbone_type == "resnet":
                pooled = model.pool(features)
                flat = torch.flatten(pooled, 1)
                outputs = model.classifier_head(flat)
            else:
                outputs = model.classifier_head(features)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            topk_correct += topk_accuracy(outputs, labels, k=topk)

            if centroid_tensor is not None:
                # For centroid distance we need flat feature vectors
                if model.backbone_type == "resnet":
                    feat_flat = flat
                else:
                    feat_flat = features
                centroid_topk_correct += topk_centroid_accuracy(
                    feat_flat, centroid_tensor, labels, k=topk,
                )

    metrics: Dict[str, float] = {
        "top1_accuracy": 100.0 * correct / total,
        "topk_accuracy": 100.0 * topk_correct / total,
    }
    if centroid_tensor is not None:
        metrics["centroid_topk_accuracy"] = 100.0 * centroid_topk_correct / total

    for key, val in metrics.items():
        print(f"  {key}: {val:.2f}%")
    return metrics


# ── Per-image predictions (classifier head) ──────────────────────────────────


def predict_topk_classifier(
    model: ImageClassifier,
    device: torch.device,
    dataset_path: str,
    classes: List[str] = CLASSES,
    topk: int = 5,
    transform: transforms.Compose = _DEFAULT_TRANSFORM,
    output_path: Optional[str] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """Run per-image top-k predictions using the classifier head.

    Iterates over ``noiseLessImg/`` and ``noisyImg/`` sub-directories,
    classifies each image, and returns (and optionally saves) a dict
    mapping ``{filename: {image_type: [class1, class2, …]}}``.
    """
    noiseless_dir = os.path.join(dataset_path, "noiseLessImg")
    noisy_dir = os.path.join(dataset_path, "noisyImg")

    image_paths: List[str] = []
    for d in (noiseless_dir, noisy_dir):
        if os.path.isdir(d):
            image_paths.extend(sorted(glob(os.path.join(d, "*.png"))))

    predictions: Dict[str, Dict[str, List[str]]] = {}

    model.eval()
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Predicting (classifier)"):
            image = Image.open(img_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)

            outputs = model(tensor)
            _, indices = torch.topk(outputs, min(topk, len(classes)), dim=1)
            pred_classes = [classes[i] for i in indices[0].cpu().numpy()]

            img_type = os.path.basename(os.path.dirname(img_path))
            fname = os.path.basename(img_path)
            predictions.setdefault(fname, {})[img_type] = pred_classes

    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(predictions, f, indent=4)
        print(f"Top-{topk} classifier predictions saved → {output_path}")

    return predictions


# ── Per-image predictions (centroid distance) ────────────────────────────────


def predict_topk_centroids(
    model: ImageClassifier,
    device: torch.device,
    dataset_path: str,
    centroid_path: str,
    classes: List[str] = CLASSES,
    topk: int = 5,
    transform: transforms.Compose = _DEFAULT_TRANSFORM,
    output_path: Optional[str] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """Run per-image top-k predictions using centroid distances.

    For each image, extracts the encoder feature vector, computes the
    Euclidean distance to every class centroid, and returns the k nearest
    classes.
    """
    centroid_tensor, centroid_class_names = load_centroids(centroid_path, device)

    noiseless_dir = os.path.join(dataset_path, "noiseLessImg")
    noisy_dir = os.path.join(dataset_path, "noisyImg")

    image_paths: List[str] = []
    for d in (noiseless_dir, noisy_dir):
        if os.path.isdir(d):
            image_paths.extend(sorted(glob(os.path.join(d, "*.png"))))

    predictions: Dict[str, Dict[str, List[str]]] = {}

    model.eval()
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Predicting (centroids)"):
            image = Image.open(img_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)

            features = model.encoder(tensor)  # (1, latent_dim)
            if model.backbone_type == "resnet":
                features = torch.flatten(model.pool(features), 1)

            dists = torch.cdist(features, centroid_tensor, p=2)  # (1, num_classes)
            _, topk_idx = torch.topk(
                dists, k=min(topk, len(centroid_class_names)),
                dim=1, largest=False,
            )
            pred_classes = [centroid_class_names[i] for i in topk_idx[0].cpu().numpy()]

            img_type = os.path.basename(os.path.dirname(img_path))
            fname = os.path.basename(img_path)
            predictions.setdefault(fname, {})[img_type] = pred_classes

    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(predictions, f, indent=4)
        print(f"Top-{topk} centroid predictions saved → {output_path}")

    return predictions


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Inference & evaluation for trained classifiers.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- evaluate ---
    eval_p = sub.add_parser("evaluate", help="Evaluate on a test set.")
    eval_p.add_argument("--backbone", type=str, default="dino", choices=["dino", "resnet"])
    eval_p.add_argument("--weights", type=str, required=True, help="Path to classifier weights.")
    eval_p.add_argument("--test_path", type=str, required=True, help="Test dataset path.")
    eval_p.add_argument("--centroid_path", type=str, default=None, help="Optional centroids JSON.")
    eval_p.add_argument("--topk", type=int, default=5)
    eval_p.add_argument("--batch_size", type=int, default=32)
    eval_p.add_argument("--image_size", type=int, default=96)

    # --- predict ---
    pred_p = sub.add_parser("predict", help="Per-image top-k predictions.")
    pred_p.add_argument("--backbone", type=str, default="dino", choices=["dino", "resnet"])
    pred_p.add_argument("--weights", type=str, required=True)
    pred_p.add_argument("--dataset_path", type=str, required=True, help="Path with noisyImg/ & noiseLessImg/.")
    pred_p.add_argument("--centroid_path", type=str, default=None, help="If given, use centroids instead of classifier head.")
    pred_p.add_argument("--topk", type=int, default=5)
    pred_p.add_argument("--output", type=str, required=True, help="Output JSON path.")
    pred_p.add_argument("--image_size", type=int, default=96)

    args = parser.parse_args()

    t = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    model, device = load_classifier(
        backbone=args.backbone,
        weights_path=args.weights,
        num_classes=len(CLASSES),
    )

    if args.command == "evaluate":
        evaluate_test_set(
            model, device,
            test_dataset_path=args.test_path,
            centroid_path=args.centroid_path,
            topk=args.topk,
            batch_size=args.batch_size,
            transform=t,
        )

    elif args.command == "predict":
        if args.centroid_path:
            predict_topk_centroids(
                model, device,
                dataset_path=args.dataset_path,
                centroid_path=args.centroid_path,
                topk=args.topk,
                transform=t,
                output_path=args.output,
            )
        else:
            predict_topk_classifier(
                model, device,
                dataset_path=args.dataset_path,
                topk=args.topk,
                transform=t,
                output_path=args.output,
            )
