"""
Train DINOv2 autoencoder + classifier on RadioML constellation images.

Combines train/test images across multiple SNR levels (e.g. snr_0db,
snr_10db, snr_20db) into a single training and test pool.  Follows the
same two-stage approach used for the "own" dataset:

  Stage 1 — Autoencoder:  Self-supervised reconstruction pre-training.
  Stage 2 — Classifier:   Supervised fine-tuning with a classification head
                           initialised from the autoencoder encoder.

Usage
-----
::

    # Stage 1: Autoencoder
    python -m src.representation_learning.radioml_dino_training \\
        --stage autoencoder \\
        --data_root /mnt/d/Rowan/discrete-llm-amc/data/RadioML \\
        --snr_levels snr_0db snr_10db snr_20db \\
        --save_path exp/radioml_dino_autoencoder.pth \\
        --num_epochs 50 --learning_rate 1e-4 --batch_size 128

    # Stage 2: Classifier
    python -m src.representation_learning.radioml_dino_training \\
        --stage classifier \\
        --data_root /mnt/d/Rowan/discrete-llm-amc/data/RadioML \\
        --snr_levels snr_0db snr_10db snr_20db \\
        --pretrained_path exp/radioml_dino_autoencoder.pth \\
        --save_path exp/radioml_dino_classifier.pth \\
        --num_epochs 50 --learning_rate 5e-4 --batch_size 128
"""

from __future__ import annotations

import argparse
import os
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from src.representation_learning.autoencoder_vit import DinoV2Autoencoder
from src.representation_learning.classifier_training import (
    ImageClassifier,
    topk_accuracy,
)
from src.representation_learning.data_loader import RadioMLConstellationDataset


# ── RadioML 2018.01A class names (24 classes) ───────────────────────────────

RADIOML_CLASSES: List[str] = [
    "128APSK", "128QAM", "16APSK", "16PSK", "16QAM", "256QAM",
    "32APSK", "32PSK", "32QAM", "4ASK", "64APSK", "64QAM",
    "8ASK", "8PSK", "AM-DSB-SC", "AM-DSB-WC", "AM-SSB-SC", "AM-SSB-WC",
    "BPSK", "FM", "GMSK", "OOK", "OQPSK", "QPSK",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_img_dirs(data_root: str, snr_levels: List[str], split: str) -> List[str]:
    """Return paths to ``img/`` dirs for the given SNR levels and split."""
    dirs = []
    for snr in snr_levels:
        d = os.path.join(data_root, snr, split, "img")
        if os.path.isdir(d):
            dirs.append(d)
        else:
            print(f"  Warning: {d} does not exist — skipping")
    return dirs


def _build_datasets(
    data_root: str,
    snr_levels: List[str],
    classes: List[str],
    image_size: int,
    val_fraction: float = 0.1,
):
    """Create train/val/test datasets from RadioML constellation images.

    Train images are split into train+val via ``random_split``.
    Test images come from the actual ``test/`` split.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # ── Train + Val (random split from train images) ─────────────────────
    train_img_dirs = _build_img_dirs(data_root, snr_levels, "train")
    full_train = RadioMLConstellationDataset(
        img_dirs=train_img_dirs,
        classes=classes,
        transform=transform,
    )
    n_total = len(full_train)
    n_val = int(val_fraction * n_total)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=generator)

    # ── Test (from actual test split) ────────────────────────────────────
    test_img_dirs = _build_img_dirs(data_root, snr_levels, "test")
    test_ds = RadioMLConstellationDataset(
        img_dirs=test_img_dirs,
        classes=classes,
        transform=transform,
    )

    print(f"  Dataset sizes  →  train: {n_train}  val: {n_val}  test: {len(test_ds)}")
    return train_ds, val_ds, test_ds


# ── Stage 1: Autoencoder ────────────────────────────────────────────────────

def train_autoencoder(args):
    """Train the DINOv2 autoencoder on RadioML constellation images.

    Includes early stopping on validation reconstruction loss to prevent
    overfitting.  The best model weights (lowest val loss) are restored
    before saving.
    """
    print("\n=== Stage 1: DINOv2 Autoencoder — RadioML ===\n")

    train_ds, val_ds, test_ds = _build_datasets(
        data_root=args.data_root,
        snr_levels=args.snr_levels,
        classes=RADIOML_CLASSES,
        image_size=args.image_size,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DinoV2Autoencoder(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        eval_step=args.eval_step,
        freeze_encoder=False,
    )
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    print(f"  Training for {args.num_epochs} epochs on {device}")
    print(f"  Early stopping: patience={args.patience}, "
          f"min_delta={args.min_delta:.1e}\n")

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.num_epochs + 1):
        # ── Train ────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader,
                    desc=f"Epoch {epoch}/{args.num_epochs} [Train]",
                    leave=False)
        for images, _ in loop:
            images = images.to(device)
            recon = model(images)
            loss = criterion(recon, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch}/{args.num_epochs}  Train Loss: {avg_train_loss:.6f}")

        # ── Validation ───────────────────────────────────────────────────
        if epoch % args.eval_step == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, _ in tqdm(val_loader,
                                      desc=f"Epoch {epoch} [Val]",
                                      leave=False):
                    images = images.to(device)
                    recon = model(images)
                    val_loss += criterion(recon, images).item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"  Val Loss: {avg_val_loss:.6f}")

            if avg_val_loss < best_val_loss - args.min_delta:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                print(f"  ✓ New best val loss ({best_val_loss:.6f}) — snapshot saved")
            else:
                epochs_without_improvement += args.eval_step
                print(
                    f"  ✗ No improvement for {epochs_without_improvement} epoch(s) "
                    f"(best: {best_val_loss:.6f} @ epoch {best_epoch})"
                )

            if epochs_without_improvement >= args.patience:
                print(
                    f"\n  Early stopping triggered after {epoch} epochs "
                    f"(best epoch: {best_epoch}, val loss: {best_val_loss:.6f})"
                )
                break

    # ── Restore best weights ─────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n  Restored best weights from epoch {best_epoch}")

    # ── Final test evaluation ────────────────────────────────────────────
    print("\n--- Final Test Evaluation ---")
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            recon = model(images)
            test_loss += criterion(recon, images).item()
    print(f"  Test Reconstruction Loss: {test_loss / len(test_loader):.6f}")

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"\nAutoencoder saved → {args.save_path}")


# ── Stage 2: Classifier ─────────────────────────────────────────────────────

def train_classifier(args):
    """Train a classification head on top of the pre-trained DINOv2 encoder."""
    print("\n=== Stage 2: DINOv2 Classifier — RadioML ===\n")

    train_ds, val_ds, test_ds = _build_datasets(
        data_root=args.data_root,
        snr_levels=args.snr_levels,
        classes=RADIOML_CLASSES,
        image_size=args.image_size,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ImageClassifier(
        backbone="dino",
        num_classes=len(RADIOML_CLASSES),
        pretrained_path=args.pretrained_path,
        freeze_encoder=args.freeze_encoder,
    )

    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
    )

    print(f"  Training for {args.num_epochs} epochs on {device}")
    print(f"  Early stopping: patience={args.patience}, "
          f"min_delta={args.min_delta:.1e}\n")

    # ── Early-stopping / best-model state ────────────────────────────────
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    # ── Training loop ────────────────────────────────────────────────────
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader,
                    desc=f"Epoch {epoch}/{args.num_epochs} [Train]",
                    leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch}/{args.num_epochs}  Train Loss: {avg_loss:.4f}")

        # ── Validation ───────────────────────────────────────────────────
        if epoch % args.eval_step == 0:
            model.eval()
            val_loss = correct = topk_correct = total = 0
            with torch.no_grad():
                for images, labels in tqdm(val_loader,
                                           desc=f"Epoch {epoch} [Val]",
                                           leave=False):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    val_loss += criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    topk_correct += topk_accuracy(outputs, labels, k=5)

            avg_val_loss = val_loss / len(val_loader)
            acc = 100.0 * correct / total
            topk_acc = 100.0 * topk_correct / total
            print(
                f"  Val Loss: {avg_val_loss:.4f}  "
                f"Val Top-1: {acc:.2f}%  Val Top-5: {topk_acc:.2f}%"
            )

            # ── Check improvement ────────────────────────────────────────
            if avg_val_loss < best_val_loss - args.min_delta:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                # Snapshot best weights
                _m = model.module if isinstance(model, nn.DataParallel) else model
                best_state = {k: v.clone() for k, v in _m.state_dict().items()}
                print(f"  ✓ New best val loss ({best_val_loss:.4f}) — snapshot saved")
            else:
                epochs_without_improvement += args.eval_step
                print(
                    f"  ✗ No improvement for {epochs_without_improvement} epoch(s) "
                    f"(best: {best_val_loss:.4f} @ epoch {best_epoch})"
                )

            if epochs_without_improvement >= args.patience:
                print(
                    f"\n  Early stopping triggered after {epoch} epochs "
                    f"(best epoch: {best_epoch}, val loss: {best_val_loss:.4f})"
                )
                break

    # ── Restore best weights ─────────────────────────────────────────────
    if best_state is not None:
        _m = model.module if isinstance(model, nn.DataParallel) else model
        _m.load_state_dict(best_state)
        print(f"\n  Restored best weights from epoch {best_epoch}")

    # ── Final test evaluation ────────────────────────────────────────────
    print("\n--- Final Test Evaluation ---")
    model.eval()
    correct = topk_correct = total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            topk_correct += topk_accuracy(outputs, labels, k=5)

    print(f"  Test Top-1: {100.0 * correct / total:.2f}%")
    print(f"  Test Top-5: {100.0 * topk_correct / total:.2f}%")

    # ── Save ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), args.save_path)
    else:
        torch.save(model.state_dict(), args.save_path)
    print(f"\nClassifier saved → {args.save_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train DINOv2 autoencoder / classifier on RadioML data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stage", required=True,
        choices=["autoencoder", "classifier"],
        help="Which stage to run.",
    )
    parser.add_argument(
        "--data_root", type=str,
        default="/mnt/d/Rowan/discrete-llm-amc/data/RadioML",
        help="Root of RadioML data (contains snr_*db/ folders).",
    )
    parser.add_argument(
        "--snr_levels", nargs="+",
        default=["snr_0db", "snr_10db", "snr_20db"],
        help="SNR level folders to include (default: snr_0db snr_10db snr_20db).",
    )
    parser.add_argument(
        "--save_path", type=str, required=True,
        help="Where to save the trained model weights.",
    )
    parser.add_argument(
        "--pretrained_path", type=str, default=None,
        help="Path to autoencoder checkpoint (classifier stage only).",
    )
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--eval_step", type=int, default=5)
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument(
        "--patience", type=int, default=15,
        help="Early stopping patience in epochs — stop if val loss does not "
             "improve for this many epochs (default: 15).",
    )
    parser.add_argument(
        "--min_delta", type=float, default=1e-4,
        help="Minimum val loss decrease to count as improvement (default: 1e-4).",
    )
    parser.add_argument(
        "--freeze_encoder", action="store_true",
        help="Freeze the encoder during classifier training.",
    )

    args = parser.parse_args()

    if args.stage == "autoencoder":
        train_autoencoder(args)
    else:
        if args.pretrained_path is None:
            parser.error("--pretrained_path is required for classifier stage")
        train_classifier(args)


if __name__ == "__main__":
    main()
