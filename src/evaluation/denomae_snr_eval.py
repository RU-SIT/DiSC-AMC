"""
Evaluate a finetuned classifier (DINO or DenoMAE2) on each RadioML SNR level.

Usage (DenoMAE2):
    python -m src.evaluation.denomae_snr_eval \
        --backbone denomae \
        --weights exp/denomae_ft_radioml/denoMAE2_rml_finetunedClassifier_best.pth \
        --data_root /mnt/d/Rowan/discrete-llm-amc/data/RadioML \
        --output_dir exp/denomae_ft_radioml/snr_eval

Usage (DINO):
    python -m src.evaluation.denomae_snr_eval \
        --backbone dino \
        --weights exp/radioml_dino_classifier.pth \
        --data_root /mnt/d/Rowan/discrete-llm-amc/data/RadioML \
        --output_dir exp/radioml_dino_snr_eval \
        --image_size 96
"""

import argparse
import csv
import json
import os
import shutil
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from functools import partial
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix


SNR_LEVELS_DEFAULT = ["snr_-20db", "snr_-10db", "snr_0db", "snr_10db", "snr_20db"]


def build_imagefolder(img_dir: str, output_dir: str) -> str:
    """Convert flat image directory to ImageFolder layout using symlinks.

    Images named like ``128APSK_sample_0.png`` → ``output_dir/128APSK/128APSK_sample_0.png``.
    """
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for fname in os.listdir(img_dir):
        if not fname.endswith(".png"):
            continue
        parts = fname.rsplit("_sample_", 1)
        if len(parts) != 2:
            continue
        class_name = parts[0]
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        src = os.path.join(img_dir, fname)
        dst = os.path.join(class_dir, fname)
        if not os.path.exists(dst):
            os.symlink(src, dst)
        count += 1
    return count


def load_model(weights_path: str, device: torch.device, args) -> nn.Module:
    backbone = args.backbone

    if backbone == "dino":
        from src.representation_learning.classifier_training import ImageClassifier
        model = ImageClassifier(
            backbone="dino",
            num_classes=args.num_classes,
            freeze_encoder=True,
        )
        print(f"Loading DINO classifier weights: {weights_path}")
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    elif backbone == "denomae":
        from src.denoMAE2.main import DenoMAE2
        from src.denoMAE2.finetune import DownstreamClassifier

        base_encoder = DenoMAE2(
            img_size=args.image_size,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            depth=args.encoder_depth,
            num_heads=args.encoder_num_heads,
            decoder_embed_dim=args.decoder_embed_dim,
            decoder_depth=args.decoder_depth,
            decoder_num_heads=args.decoder_num_heads,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        ).to(device)

        model = DownstreamClassifier(base_encoder, args.num_classes)
        print(f"Loading DenoMAE2 classifier weights: {weights_path}")
        state_dict = torch.load(weights_path, map_location=device, weights_only=False)
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    model.to(device).eval()
    print("Model loaded successfully.")
    return model


def evaluate_snr(
    model: nn.Module,
    test_dir: str,
    snr: str,
    device: torch.device,
    batch_size: int,
    image_size: int,
    output_dir: str,
):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    class_names = dataset.classes
    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=snr):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(loader)

    print(f"  {snr}: Accuracy = {accuracy:.2f}%  Loss = {avg_loss:.4f}  ({correct}/{total})")

    # Per-class report
    report_dict = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0
    )
    report_str = classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0
    )
    print(report_str)

    # Save classification report
    report_path = os.path.join(output_dir, f"classification_report_{snr}.txt")
    with open(report_path, "w") as f:
        f.write(f"SNR: {snr}\nAccuracy: {accuracy:.2f}%\nLoss: {avg_loss:.4f}\n\n")
        f.write(report_str)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix — {snr} (Acc: {accuracy:.1f}%)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{snr}.png"), dpi=200)
    plt.close()

    return {
        "snr": snr,
        "accuracy": accuracy,
        "loss": avg_loss,
        "correct": correct,
        "total": total,
        "per_class": {cn: report_dict[cn] for cn in class_names},
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate finetuned classifier per SNR")
    parser.add_argument("--backbone", type=str, default="denomae",
                        choices=["dino", "denomae"],
                        help="Backbone type: dino or denomae")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to finetuned classifier weights")
    parser.add_argument("--data_root", type=str,
                        default="/mnt/d/Rowan/discrete-llm-amc/data/RadioML",
                        help="RadioML data root")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results (auto-named if omitted)")
    parser.add_argument("--snr_levels", nargs="+", default=SNR_LEVELS_DEFAULT,
                        help="SNR levels to evaluate")
    parser.add_argument("--num_classes", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--decoder_embed_dim", type=int, default=512)
    parser.add_argument("--encoder_depth", type=int, default=12)
    parser.add_argument("--decoder_depth", type=int, default=8)
    parser.add_argument("--encoder_num_heads", type=int, default=12)
    parser.add_argument("--decoder_num_heads", type=int, default=8)
    parser.add_argument("--gpu", type=str, default="0,1")
    args = parser.parse_args()

    # Auto-name output dir based on backbone
    if args.output_dir is None:
        if args.backbone == "dino":
            args.output_dir = "exp/radioml_dino_snr_eval"
        else:
            args.output_dir = "exp/denomae_ft_radioml/snr_eval"
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        gpus = [int(g) for g in args.gpu.split(",")]
        device = torch.device(f"cuda:{gpus[0]}")
        print(f"Using GPU(s): {gpus}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args.weights, device, args)

    results = []
    for snr in args.snr_levels:
        img_dir = os.path.join(args.data_root, snr, "test", "img")
        if not os.path.isdir(img_dir):
            print(f"[SKIP] {snr}: no images at {img_dir}")
            continue

        # Build ImageFolder layout with symlinks
        imagefolder_dir = os.path.join(args.output_dir, f"test_{snr}")
        if not os.path.isdir(imagefolder_dir):
            n = build_imagefolder(img_dir, imagefolder_dir)
            n_classes = len(os.listdir(imagefolder_dir))
            print(f"  Built ImageFolder for {snr}: {n} images, {n_classes} classes")
        else:
            print(f"  ImageFolder for {snr} already exists")

        r = evaluate_snr(model, imagefolder_dir, snr, device,
                         args.batch_size, args.image_size, args.output_dir)
        results.append(r)

    # Summary
    tag = args.backbone
    csv_path = os.path.join(args.output_dir, f"{tag}_snr_results.csv")
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'SNR':<12} {'Accuracy':>10} {'Loss':>10} {'Correct':>10} {'Total':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["SNR", "Accuracy", "Loss", "Correct", "Total"])
        for r in results:
            writer.writerow([r["snr"], f"{r['accuracy']:.2f}", f"{r['loss']:.4f}",
                             r["correct"], r["total"]])
            print(f"  {r['snr']:<12} {r['accuracy']:>9.2f}% {r['loss']:>10.4f} "
                  f"{r['correct']:>10} {r['total']:>10}")

    print(f"\n  Results saved to {csv_path}")

    # Save detailed JSON
    json_path = os.path.join(args.output_dir, f"{tag}_snr_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Detailed results saved to {json_path}")


if __name__ == "__main__":
    main()
