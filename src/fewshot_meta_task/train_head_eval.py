"""Paper-style evaluation: train a real classification head on each
episodically-trained encoder and evaluate it exactly the way Table III of
the DiSC-AMC paper does, for a number directly comparable to the existing
`denomae_ft_snr_results.csv` baseline.

This is a different readout than `fewshot_eval.py`'s K-shot prototype
classification: here the encoder is frozen, a real MLP head
(`DownstreamClassifier`, same class `src/denoMAE2/finetune.py` uses) is
trained with ordinary cross-entropy on the full in-distribution training
set, and evaluated with plain correct/total + classification_report +
confusion matrix via the same `evaluate_snr()` the original baseline used.
It answers "is this backbone's representation better," not "how well does
K-shot adaptation work" -- both are useful, which is why `fewshot_eval.py`
is kept separate rather than replaced.

Usage:
    python -m src.fewshot_meta_task.train_head_eval \
        --encoder_checkpoint exp/meta_task_fewshot_v2/encoder_control.pth \
        --condition control \
        --output_dir exp/meta_task_fewshot_v2/paper_style_eval/control

    python -m src.fewshot_meta_task.train_head_eval \
        --encoder_checkpoint exp/meta_task_fewshot_v2/encoder_meta_task.pth \
        --condition meta_task \
        --output_dir exp/meta_task_fewshot_v2/paper_style_eval/meta_task
"""
import argparse
import csv
import os

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import datasets
from torch.utils.data import DataLoader

from src.denoMAE2.finetune import DownstreamClassifier, train_classifier, evaluate_classifier
from src.evaluation.denomae_snr_eval import evaluate_snr, build_imagefolder
from .denomae_common import add_denomae_args, build_denomae2, build_transform, load_pretrained_encoder_weights


def main():
    parser = argparse.ArgumentParser(description="Train a classification head + paper-style SNR eval")
    parser.add_argument("--encoder_checkpoint", type=str, required=True,
                         help="Episodic checkpoint from episodic_train.py")
    parser.add_argument("--condition", type=str, required=True, help="Label for this run, e.g. control/meta_task")
    parser.add_argument("--train_data_path", type=str, default="exp/denomae_ft_radioml/merged_train")
    parser.add_argument("--val_data_path", type=str, default="exp/denomae_ft_radioml/merged_test")
    parser.add_argument("--snr_test_dirs", nargs="+",
                         default=["exp/denomae_ft_radioml/snr_eval/test_snr_-10db",
                                  "exp/denomae_ft_radioml/snr_eval/test_snr_-20db"],
                         help="Already-built ImageFolder test dirs -- same ones Table III/IV used")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=24)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--freeze_encoder", dest="freeze_encoder", action="store_true",
                         help="Frozen encoder + trained head only (matches finetune.py's "
                              "DownstreamClassifier default -- the convention the original "
                              "DenoMAE2.0 Table III baseline actually used).")
    parser.add_argument("--no_freeze_encoder", dest="freeze_encoder", action="store_false",
                         help="Real end-to-end finetuning: encoder weights get updated too, "
                              "matching how DINOv2's ImageClassifier baseline was built "
                              "(freeze_encoder=False default there).")
    parser.set_defaults(freeze_encoder=True)
    parser.add_argument("--gpu", type=str, default="0")
    add_denomae_args(parser)
    args = parser.parse_args()

    torch.set_num_threads(4)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    transform = build_transform(args.image_size)
    train_dataset = datasets.ImageFolder(root=args.train_data_path, transform=transform)
    val_dataset = datasets.ImageFolder(root=args.val_data_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Train: {len(train_dataset)} images, {len(train_dataset.classes)} classes. "
          f"Val: {len(val_dataset)} images.")

    base_encoder = build_denomae2(args, device)
    load_pretrained_encoder_weights(base_encoder, args.encoder_checkpoint, device)

    model = DownstreamClassifier(base_encoder, args.num_classes, freeze_encoder=args.freeze_encoder).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_path = os.path.join(args.output_dir, "classifier_best.pth")
    for epoch in range(args.num_epochs):
        train_loss, train_acc = train_classifier(model, train_loader, optimizer, criterion, device)
        print(f"[{args.condition}] epoch {epoch + 1}/{args.num_epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}%")

        if (epoch + 1) % args.eval_every == 0 or epoch == args.num_epochs - 1:
            val_loss, val_acc = evaluate_classifier(model, val_loader, criterion, device)
            print(f"[{args.condition}]   val_loss={val_loss:.4f} val_acc={val_acc:.2f}%")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_path)
                print(f"[{args.condition}]   new best val_acc={best_val_acc:.2f}%, saved")
        scheduler.step()

    if os.path.isfile(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"[{args.condition}] loaded best checkpoint (val_acc={best_val_acc:.2f}%) for SNR eval")
    model.eval()

    rows = []
    for snr_dir in args.snr_test_dirs:
        snr_name = os.path.basename(snr_dir).replace("test_", "")
        r = evaluate_snr(model, snr_dir, snr_name, device, args.batch_size, args.image_size, args.output_dir)
        rows.append(r)

    csv_path = os.path.join(args.output_dir, f"{args.condition}_paper_style_snr_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "SNR", "Accuracy", "Loss", "Correct", "Total"])
        for r in rows:
            writer.writerow([args.condition, r["snr"], f"{r['accuracy']:.2f}", f"{r['loss']:.4f}",
                              r["correct"], r["total"]])
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
