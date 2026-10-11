"""K-shot adaptation + eval on held-out SNRs, for the A/B/C comparison.

Runs the *same* seeded episodes (same classes, same images, same
support/query split) against every checkpoint passed in, so B vs. C is a
paired comparison rather than independent random draws from the same small
held-out pool.

`--snr_dirs` entries point at the flat `<snr>/train/img/` directories (files
named `CLASS_sample_N.png`), matching the layout `src/evaluation/
denomae_snr_eval.py` already expects -- it symlinks them into a per-class
ImageFolder layout, which we reuse here so both scripts cache the same
staging directories instead of building near-duplicate ones.

Usage:
    python -m src.fewshot_meta_task.fewshot_eval \
        --checkpoints control=exp/meta_task_fewshot/encoder_control.pth \
                      meta_task=exp/meta_task_fewshot/encoder_meta_task.pth \
        --snr_dirs snr_-10db=/mnt/d/Rowan/discrete-llm-amc/data/RadioML/snr_-10db/train/img \
                   snr_-20db=/mnt/d/Rowan/discrete-llm-amc/data/RadioML/snr_-20db/train/img \
        --shots 1 5 10 \
        --output_csv exp/meta_task_fewshot/results.csv

Note: SNR labels must not start with "-" (argparse would read `-10db` as a
flag) -- use the `snr_-10db` style labels shown above.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets

from src.evaluation.denomae_snr_eval import build_imagefolder
from .denomae_common import add_denomae_args, build_denomae2, build_transform
from .episodic_sampler import EpisodicSampler
from .proto_utils import encode_mean_pool, prototype_logits, accuracy


def parse_kv_list(items):
    out = {}
    for item in items:
        key, _, value = item.partition("=")
        out[key] = value
    return out


def load_encoder_checkpoint(args, device, checkpoint_path):
    encoder = build_denomae2(args, device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["encoder_state_dict"] if isinstance(ckpt, dict) and "encoder_state_dict" in ckpt else ckpt
    encoder.load_state_dict(state_dict)
    encoder.eval()
    train_seconds = ckpt.get("train_seconds", 0.0) if isinstance(ckpt, dict) else 0.0
    return encoder, train_seconds


def run_episodes(encoder, sampler, n_way, shot, query, n_episodes, seed, device):
    sampler.reset_rng(seed)  # same seed -> same episode sequence across checkpoints; image cache is kept
    accs = []
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for _ in range(n_episodes):
            support_set, support_labels, query_set, query_labels = sampler.sample_episode(
                n_way, shot, query, device=device
            )
            support_feats = encode_mean_pool(encoder, support_set, mask_ratio=0.0)
            query_feats = encode_mean_pool(encoder, query_set, mask_ratio=0.0)
            logits = prototype_logits(support_feats, support_labels, query_feats, n_way)
            accs.append(accuracy(logits, query_labels))
    mean = float(np.mean(accs))
    ci95 = float(1.96 * np.std(accs) / np.sqrt(len(accs))) if len(accs) > 1 else 0.0
    return mean, ci95


def pooled_eval(encoder, sampler, n_way, shot, seed, device, batch_size, output_dir, condition, snr_label):
    """Paper-style readout for the same K-shot prototype method: one fixed
    (seeded) support draw per class, classify *every* remaining image in the
    pool as the query set (not a small per-episode subsample), report a
    single accuracy + classification_report + confusion matrix -- same
    artifacts `src/evaluation/denomae_snr_eval.py` saves for Table III/IV,
    so this is directly comparable in format even though the classifier
    itself (nearest prototype, not a trained head) is different.
    """
    sampler.reset_rng(seed)
    class_names = sampler.dataset.classes
    classes = sampler.classes[:n_way]  # fixed order -> reproducible, and with n_way == all classes
                                        # (the usual case here) this is just every class, no sampling needed

    support_imgs, support_labels = [], []
    query_idx_by_way = {}
    for way_idx, cls in enumerate(classes):
        pool = sampler.by_class[cls]
        support_idx = set(sampler.rng.sample(pool, shot))
        query_idx_by_way[way_idx] = [i for i in pool if i not in support_idx]
        for idx in support_idx:
            support_imgs.append(sampler._get_image(idx))
            support_labels.append(way_idx)

    autocast_kwargs = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    with torch.no_grad(), torch.autocast(**autocast_kwargs):
        support_set = torch.stack(support_imgs).to(device)
        support_labels_t = torch.tensor(support_labels, device=device)
        support_feats = encode_mean_pool(encoder, support_set, mask_ratio=0.0)
        prototypes = torch.stack([support_feats[support_labels_t == c].mean(dim=0) for c in range(n_way)])

    all_preds, all_labels = [], []
    with torch.no_grad(), torch.autocast(**autocast_kwargs):
        for way_idx, idxs in query_idx_by_way.items():
            for i in range(0, len(idxs), batch_size):
                batch_idx = idxs[i:i + batch_size]
                imgs = torch.stack([sampler._get_image(j) for j in batch_idx]).to(device)
                feats = encode_mean_pool(encoder, imgs, mask_ratio=0.0)
                preds = (-torch.cdist(feats, prototypes)).argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend([way_idx] * len(batch_idx))

    total = len(all_labels)
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    acc = 100.0 * correct / total

    name_lookup = {way_idx: class_names[cls] for way_idx, cls in enumerate(classes)}
    label_names = [name_lookup[i] for i in range(n_way)]

    report_str = classification_report(all_labels, all_preds, labels=list(range(n_way)),
                                        target_names=label_names, zero_division=0)
    os.makedirs(output_dir, exist_ok=True)
    tag = f"{condition}_{snr_label}_{shot}shot"
    with open(os.path.join(output_dir, f"classification_report_{tag}.txt"), "w") as f:
        f.write(f"condition={condition} snr={snr_label} shot={shot}\nAccuracy: {acc:.2f}%\n\n{report_str}")

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(n_way)))
    cm_norm = cm.astype("float") / np.clip(cm.sum(axis=1)[:, np.newaxis], 1, None)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.title(f"{condition} {snr_label} {shot}-shot (Acc: {acc:.1f}%)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{tag}.png"), dpi=200)
    plt.close()

    return acc, correct, total


def main():
    parser = argparse.ArgumentParser(description="Paired A/B/C few-shot held-out-SNR evaluation")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                         help='condition=path pairs, e.g. control=... meta_task=...')
    parser.add_argument("--snr_dirs", nargs="+", required=True,
                         help='label=path pairs for held-out-SNR flat img/ dirs, e.g. snr_-10db=...')
    parser.add_argument("--imagefolder_cache", type=str, default="exp/meta_task_fewshot/imagefolder_cache",
                         help="where to stage the per-class symlink layout built from each flat img/ dir")
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--query", type=int, default=5)
    parser.add_argument("--n_episodes", type=int, default=200)
    parser.add_argument("--eval_seed", type=int, default=1234,
                         help="shared seed across conditions -> paired episodes/support draws")
    parser.add_argument("--eval_style", choices=["episodic", "pooled"], default="episodic",
                         help="episodic: average many small random K-shot episodes with a CI "
                              "(this script's original behavior). pooled: one fixed K-shot support "
                              "draw, classify every remaining held-out image, report plain accuracy "
                              "+ classification_report + confusion matrix -- same convention as "
                              "src/evaluation/denomae_snr_eval.py's Table III/IV numbers.")
    parser.add_argument("--pooled_batch_size", type=int, default=64)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--report_dir", type=str, default=None,
                         help="pooled mode only: where to save classification_report/confusion_matrix "
                              "files (defaults to output_csv's directory)")
    parser.add_argument("--gpu", type=str, default="0")
    add_denomae_args(parser)
    args = parser.parse_args()
    if args.report_dir is None:
        args.report_dir = os.path.dirname(args.output_csv) or "."

    torch.set_num_threads(4)  # see episodic_train.py for why

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    checkpoints = parse_kv_list(args.checkpoints)
    snr_dirs = parse_kv_list(args.snr_dirs)
    transform = build_transform(args.image_size)

    datasets_by_snr = {}
    for snr_label, flat_img_dir in snr_dirs.items():
        imagefolder_dir = os.path.join(args.imagefolder_cache, snr_label)
        if not os.path.isdir(imagefolder_dir):
            n = build_imagefolder(flat_img_dir, imagefolder_dir)
            print(f"  Built ImageFolder for {snr_label}: {n} images")
        ds = datasets.ImageFolder(root=imagefolder_dir, transform=transform)
        n_way = min(args.n_way, len(ds.classes))
        # One sampler per SNR, reused (with reseeding) across conditions and
        # shots: it caches decoded images in memory, so the same ~2-3k
        # images aren't re-read from disk for every (condition, shot) combo.
        sampler = EpisodicSampler(ds)
        datasets_by_snr[snr_label] = (sampler, n_way)
        print(f"{snr_label}: {len(ds)} images, {len(ds.classes)} classes (using n_way={n_way})")

    rows = []
    for condition, ckpt_path in checkpoints.items():
        encoder, train_seconds = load_encoder_checkpoint(args, device, ckpt_path)
        for snr_label, (sampler, n_way) in datasets_by_snr.items():
            for shot in args.shots:
                need = shot + (args.query if args.eval_style == "episodic" else 1)
                max_per_class = min(len(v) for v in sampler.by_class.values())
                if need > max_per_class:
                    print(f"[SKIP] {condition} {snr_label} shot={shot}: needs {need}/class, "
                          f"only {max_per_class} available")
                    continue

                if args.eval_style == "episodic":
                    mean_acc, ci95 = run_episodes(
                        encoder, sampler, n_way, shot, args.query, args.n_episodes, args.eval_seed, device
                    )
                    print(f"[{condition}] {snr_label} shot={shot}: acc={mean_acc:.4f} +/- {ci95:.4f}")
                    rows.append({
                        "condition": condition, "snr": snr_label, "shot": shot,
                        "query": args.query, "n": args.n_episodes,
                        "accuracy": mean_acc, "ci95_or_na": ci95, "train_seconds": train_seconds,
                    })
                else:
                    acc, correct, total = pooled_eval(
                        encoder, sampler, n_way, shot, args.eval_seed, device,
                        args.pooled_batch_size, args.report_dir, condition, snr_label
                    )
                    print(f"[{condition}] {snr_label} shot={shot}: acc={acc:.2f}% ({correct}/{total})")
                    rows.append({
                        "condition": condition, "snr": snr_label, "shot": shot,
                        "query": total, "n": 1,
                        "accuracy": acc / 100.0, "ci95_or_na": "", "train_seconds": train_seconds,
                    })

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "condition", "snr", "shot", "query", "n", "accuracy", "ci95_or_na", "train_seconds"
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved results to {args.output_csv}")


if __name__ == "__main__":
    main()
