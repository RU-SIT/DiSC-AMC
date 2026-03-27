"""
Cross-domain evaluation: use encoders trained on 'own' data (10 classes) to
classify RadioML data (24 classes) via transfer learning.

Approach:
  1. Load encoder weights from a model trained on own data (10 classes).
  2. Attach a fresh classifier head for 24 RadioML classes.
  3. Train only the new head on RadioML constellation images (encoder frozen).
  4. Evaluate per-SNR level on RadioML test images.

Supports both DINO and DenoMAE2 backbones.

Usage (DenoMAE2):
    python -m src.evaluation.cross_domain_eval \
        --backbone denomae \
        --weights models/denoMAE2_finetunedClassifier.pth \
        --data_root /mnt/d/Rowan/discrete-llm-amc/data/RadioML \
        --output_dir exp/cross_domain_denomae \
        --train_epochs 20

Usage (DINO):
    python -m src.evaluation.cross_domain_eval \
        --backbone dino \
        --weights exp/dino_classifier.pth \
        --data_root /mnt/d/Rowan/discrete-llm-amc/data/RadioML \
        --output_dir exp/cross_domain_dino \
        --train_epochs 20 \
        --image_size 96
"""

import argparse
import csv
import json
import os
from functools import partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

SNR_LEVELS_DEFAULT = ["snr_-20db", "snr_-10db", "snr_0db", "snr_10db", "snr_20db"]


# ── ImageFolder construction ────────────────────────────────────────────────


def build_imagefolder(img_dir: str, output_dir: str) -> int:
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
        src = os.path.abspath(os.path.join(img_dir, fname))
        dst = os.path.join(class_dir, fname)
        if not os.path.exists(dst):
            os.symlink(src, dst)
        count += 1
    return count


# ── Encoder loading (own-data weights, discard old classifier head) ──────────


def load_encoder_denomae(weights_path: str, device: torch.device, args):
    """Load DenoMAE2 encoder from own-data classifier weights (10-class)."""
    from src.denoMAE2.main import DenoMAE2

    encoder = DenoMAE2(
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

    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Extract only encoder keys (discard classification_head)
    encoder_keys = {
        k.replace("encoder.", ""): v
        for k, v in state_dict.items()
        if k.startswith("encoder.")
    }

    if encoder_keys:
        encoder.load_state_dict(encoder_keys, strict=False)
        print(f"Loaded {len(encoder_keys)} encoder keys from {weights_path}")
    else:
        # Weights may be a raw DenoMAE2 checkpoint (pretrain)
        encoder.load_state_dict(state_dict, strict=False)
        print(f"Loaded full checkpoint into encoder from {weights_path}")

    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder


def load_encoder_dino(weights_path: str, device: torch.device, args):
    """Load DINO encoder from own-data classifier weights (10-class)."""
    from src.representation_learning.autoencoder_vit import DinoV2Autoencoder

    autoencoder = DinoV2Autoencoder(freeze_encoder=True)

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)

    # Extract encoder keys (discard classifier_head)
    encoder_keys = {
        k.replace("encoder.", ""): v
        for k, v in state_dict.items()
        if k.startswith("encoder.")
    }

    if encoder_keys:
        autoencoder.encoder.load_state_dict(encoder_keys, strict=False)
        print(f"Loaded {len(encoder_keys)} encoder keys from {weights_path}")
    else:
        autoencoder.load_state_dict(state_dict, strict=False)
        print(f"Loaded full checkpoint into autoencoder from {weights_path}")

    encoder = autoencoder.encoder
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder


# ── Wrapper model: frozen encoder + trainable head ──────────────────────────


class CrossDomainClassifier(nn.Module):
    """Frozen encoder from own-data + new trainable head for RadioML."""

    def __init__(self, encoder: nn.Module, backbone: str, num_classes: int = 24):
        super().__init__()
        self.encoder = encoder
        self.backbone = backbone
        latent_dim = 768  # both DINO and DenoMAE2 use 768

        if backbone == "denomae":
            self.head = nn.Sequential(
                nn.Linear(latent_dim, 512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, num_classes),
            )
        else:  # dino
            self.head = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, num_classes),
            )

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.backbone == "denomae":
                features, _, _ = self.encoder.forward_encoder(x, mask_ratio=0)
                return features[:, 1:, :].mean(dim=1)  # mean-pool patch tokens
            else:
                return self.encoder(x)  # DINO returns CLS token directly

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._extract_features(x)
        return self.head(features)


# ── Training the new head ────────────────────────────────────────────────────


def train_head(
    model: CrossDomainClassifier,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
):
    """Train only the classifier head (encoder stays frozen)."""
    model.to(device)
    model.encoder.eval()
    model.head.train()

    optimizer = optim.AdamW(model.head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        total_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.1f}%")

        scheduler.step()
        acc = 100.0 * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch}: Loss={avg_loss:.4f}  Acc={acc:.2f}%")

    return model


# ── Evaluation ───────────────────────────────────────────────────────────────


def evaluate_snr(
    model: CrossDomainClassifier,
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

    model.eval()
    all_preds, all_labels = [], []
    total_loss, correct, total = 0.0, 0, 0

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

    report_dict = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0
    )
    report_str = classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0
    )
    print(report_str)

    report_path = os.path.join(output_dir, f"classification_report_{snr}.txt")
    with open(report_path, "w") as f:
        f.write(f"SNR: {snr}\nAccuracy: {accuracy:.2f}%\nLoss: {avg_loss:.4f}\n\n")
        f.write(report_str)

    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title(f"Cross-Domain Confusion Matrix — {snr} (Acc: {accuracy:.1f}%)")
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


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Cross-domain eval: own-data encoder → RadioML classification"
    )
    parser.add_argument("--backbone", type=str, default="denomae",
                        choices=["dino", "denomae"])
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to own-data classifier weights (10-class)")
    parser.add_argument("--data_root", type=str,
                        default="/mnt/d/Rowan/discrete-llm-amc/data/RadioML",
                        help="RadioML data root")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--train_snr", nargs="+", default=SNR_LEVELS_DEFAULT,
                        help="SNR levels to use for training the new head")
    parser.add_argument("--eval_snr", nargs="+", default=SNR_LEVELS_DEFAULT,
                        help="SNR levels to evaluate on")
    parser.add_argument("--num_classes", type=int, default=24)
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--decoder_embed_dim", type=int, default=512)
    parser.add_argument("--encoder_depth", type=int, default=12)
    parser.add_argument("--decoder_depth", type=int, default=8)
    parser.add_argument("--encoder_num_heads", type=int, default=12)
    parser.add_argument("--decoder_num_heads", type=int, default=8)
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"exp/cross_domain_{args.backbone}"

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu.split(',')[0]}")
        print(f"Using GPU: {args.gpu}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load encoder (discard old 10-class head) ──
    print(f"\n{'='*60}")
    print(f"  Loading {args.backbone} encoder from own-data weights")
    print(f"{'='*60}")
    if args.backbone == "denomae":
        encoder = load_encoder_denomae(args.weights, device, args)
    else:
        encoder = load_encoder_dino(args.weights, device, args)

    # ── 2. Build CrossDomainClassifier with new 24-class head ──
    model = CrossDomainClassifier(encoder, args.backbone, args.num_classes)
    model.to(device)
    print(f"Built CrossDomainClassifier: {args.backbone} encoder → {args.num_classes}-class head")

    # ── 3. Prepare RadioML training data (ImageFolder from constellation images) ──
    print(f"\n{'='*60}")
    print(f"  Preparing RadioML training data")
    print(f"{'='*60}")

    transform_train = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_datasets = []
    for snr in args.train_snr:
        img_dir = os.path.join(args.data_root, snr, "train", "img")
        if not os.path.isdir(img_dir):
            print(f"  [SKIP] {snr}: no train images at {img_dir}")
            continue

        folder_dir = os.path.join(args.output_dir, f"train_{snr}")
        if not os.path.isdir(folder_dir):
            n = build_imagefolder(img_dir, folder_dir)
            n_cls = len([d for d in os.listdir(folder_dir)
                        if os.path.isdir(os.path.join(folder_dir, d))])
            print(f"  Built ImageFolder for {snr}: {n} images, {n_cls} classes")
        else:
            print(f"  ImageFolder for {snr} already exists")

        ds = datasets.ImageFolder(root=folder_dir, transform=transform_train)
        train_datasets.append(ds)
        print(f"  {snr}: {len(ds)} training images")

    if not train_datasets:
        raise RuntimeError("No training data found! Check --data_root and --train_snr.")

    combined_train = ConcatDataset(train_datasets)
    # Use class_to_idx from first dataset (all should be identical)
    print(f"  Total training images: {len(combined_train)}")

    train_loader = DataLoader(
        combined_train, batch_size=args.batch_size, shuffle=True, num_workers=4,
    )

    # ── 4. Train the new classifier head ──
    print(f"\n{'='*60}")
    print(f"  Training {args.num_classes}-class head ({args.train_epochs} epochs)")
    print(f"{'='*60}")

    model = train_head(model, train_loader, device, epochs=args.train_epochs, lr=args.lr)

    # Save the cross-domain model
    save_path = os.path.join(args.output_dir, f"{args.backbone}_cross_domain_classifier.pth")
    torch.save(model.state_dict(), save_path)
    print(f"  Saved cross-domain model to {save_path}")

    # ── 5. Evaluate per SNR ──
    print(f"\n{'='*60}")
    print(f"  Evaluating per SNR level")
    print(f"{'='*60}")

    results = []
    for snr in args.eval_snr:
        img_dir = os.path.join(args.data_root, snr, "test", "img")
        if not os.path.isdir(img_dir):
            print(f"  [SKIP] {snr}: no test images at {img_dir}")
            continue

        folder_dir = os.path.join(args.output_dir, f"test_{snr}")
        if not os.path.isdir(folder_dir):
            n = build_imagefolder(img_dir, folder_dir)
            n_cls = len([d for d in os.listdir(folder_dir)
                        if os.path.isdir(os.path.join(folder_dir, d))])
            print(f"  Built ImageFolder for {snr}: {n} images, {n_cls} classes")
        else:
            print(f"  ImageFolder for {snr} already exists")

        r = evaluate_snr(model, folder_dir, snr, device,
                         args.batch_size, args.image_size, args.output_dir)
        results.append(r)

    # ── 6. Summary ──
    tag = args.backbone
    csv_path = os.path.join(args.output_dir, f"{tag}_cross_domain_results.csv")

    print(f"\n{'='*60}")
    print(f"  CROSS-DOMAIN RESULTS SUMMARY ({tag})")
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

    print(f"\n  Results CSV: {csv_path}")

    json_path = os.path.join(args.output_dir, f"{tag}_cross_domain_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Detailed JSON: {json_path}")

    # Mean accuracy across SNR levels
    if results:
        mean_acc = np.mean([r["accuracy"] for r in results])
        print(f"\n  Mean accuracy across SNR levels: {mean_acc:.2f}%")


if __name__ == "__main__":
    main()
