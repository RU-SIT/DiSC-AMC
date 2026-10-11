"""Stage M/B: episodic training of a DenoMAE2 encoder on in-distribution SNRs.

--mode control    -> condition B: prototype classification loss only.
--mode meta_task  -> condition C: prototype loss + DenoMAE2's own native
                     masked-reconstruction/patch-location loss on the same
                     batch, as the Meta-Task-inspired auxiliary regularizer.

Both modes see identical episodes (same seed, same sampler) and the same
main-task optimizer/LR -- the only difference is condition C's extra
backward/step through a *separate* optimizer for the auxiliary loss, at
its own (much lower) learning rate. This matches the original Meta-Task
reference implementation's own pattern (a dedicated low-LR optimizer for
the auxiliary autoencoder, not a loss-weight scalar on a combined loss
through one shared optimizer) -- combining the two losses into one before
backprop under AdamW is a weaker, less controllable knob than it looks,
since Adam's per-parameter gradient normalization largely absorbs constant
scalar multipliers on a loss term.

Usage:
    python -m src.fewshot_meta_task.episodic_train \
        --mode control \
        --train_data_path exp/denomae_ft_radioml/merged_train \
        --init_weights exp/denomae_ft_radioml/denoMAE2_rml_finetunedClassifier_best.pth \
        --output_path exp/meta_task_fewshot/encoder_control.pth \
        --num_episodes 2000

    python -m src.fewshot_meta_task.episodic_train \
        --mode meta_task \
        --train_data_path exp/denomae_ft_radioml/merged_train \
        --init_weights exp/denomae_ft_radioml/denoMAE2_rml_finetunedClassifier_best.pth \
        --output_path exp/meta_task_fewshot/encoder_meta_task.pth \
        --num_episodes 2000
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F
from torchvision import datasets

from .denomae_common import (
    add_denomae_args, build_denomae2, build_transform,
    load_pretrained_encoder_weights, reconstruction_loss,
)
from .episodic_sampler import EpisodicSampler
from .proto_utils import encode_mean_pool, prototype_logits, accuracy


def main():
    parser = argparse.ArgumentParser(description="Episodic training: control (B) vs Meta-Task (C)")
    parser.add_argument("--mode", choices=["control", "meta_task"], required=True)
    parser.add_argument("--train_data_path", type=str, required=True,
                         help="ImageFolder root with in-distribution-SNR classes (e.g. merged_train)")
    parser.add_argument("--init_weights", type=str, default=None,
                         help="Pretrained DenoMAE2 / DownstreamClassifier checkpoint to start from")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=2000)
    parser.add_argument("--shot", type=int, default=5)
    parser.add_argument("--query", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5,
                         help="main (prototype classification) task learning rate")
    parser.add_argument("--aux_lr", type=float, default=1e-6,
                         help="auxiliary (reconstruction) task learning rate -- separate "
                              "optimizer, separate backward/step call. Reference "
                              "implementation default is 1e-6, ~2 orders of magnitude "
                              "below a typical main-task lr of 1e-4.")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--mask_ratio", type=float, default=0.75,
                         help="mask ratio for the auxiliary masked-reconstruction loss")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--gpu", type=str, default="0")
    add_denomae_args(parser)
    args = parser.parse_args()

    # PyTorch's CPU intra-op parallelism defaults to one thread per core;
    # that thrashes badly when processing many tiny per-image tensor ops
    # (transforms) one at a time in a Python loop instead of a batched
    # DataLoader. Cap it rather than let it burn dozens of cores for no gain.
    torch.set_num_threads(4)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = build_transform(args.image_size)
    train_dataset = datasets.ImageFolder(root=args.train_data_path, transform=transform)
    n_way = min(args.n_way, len(train_dataset.classes))
    print(f"Loaded {len(train_dataset)} images, {len(train_dataset.classes)} classes "
          f"(using n_way={n_way})")

    sampler = EpisodicSampler(train_dataset, seed=args.seed)

    encoder = build_denomae2(args, device)
    if args.init_weights:
        load_pretrained_encoder_weights(encoder, args.init_weights, device)
    encoder.train()

    main_optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    aux_optimizer = None
    if args.mode == "meta_task":
        aux_optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.aux_lr, weight_decay=args.weight_decay)

    autocast_kwargs = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

    t0 = time.time()
    for step in range(1, args.num_episodes + 1):
        support_set, support_labels, query_set, query_labels = sampler.sample_episode(
            n_way, args.shot, args.query, device=device
        )

        # Auxiliary step first (separate optimizer, separate backward/step,
        # own low LR) -- matches the reference implementation's order:
        # the encoder gets a self-supervised update before the episode's
        # classification loss is computed and applied.
        rec_loss_val = 0.0
        if args.mode == "meta_task":
            with torch.autocast(**autocast_kwargs):
                # Reconstruction loss on the support set only (not
                # support+query) -- halves the extra memory this mode needs
                # over the control condition, which matters at 24-way on a
                # single GPU. Still a legitimate self-supervised regularizer
                # on in-distribution episode data.
                rec_loss = reconstruction_loss(encoder, support_set, support_set, mask_ratio=args.mask_ratio)
            aux_optimizer.zero_grad()
            rec_loss.backward()
            aux_optimizer.step()
            rec_loss_val = rec_loss.item()

        with torch.autocast(**autocast_kwargs):
            support_feats = encode_mean_pool(encoder, support_set, mask_ratio=0.0)
            query_feats = encode_mean_pool(encoder, query_set, mask_ratio=0.0)
            logits = prototype_logits(support_feats, support_labels, query_feats, n_way)
            proto_loss = F.cross_entropy(logits, query_labels)

        main_optimizer.zero_grad()
        proto_loss.backward()
        main_optimizer.step()

        if step % args.log_every == 0 or step == args.num_episodes:
            acc = accuracy(logits, query_labels)
            print(f"[{args.mode}] step {step}/{args.num_episodes} "
                  f"proto_loss={proto_loss.item():.4f} acc={acc:.3f} "
                  f"rec_loss={rec_loss_val:.4f}")

    elapsed = time.time() - t0
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "mode": args.mode,
        "train_seconds": elapsed,
        "num_episodes": args.num_episodes,
        "shot": args.shot,
        "query": args.query,
        "n_way": n_way,
        "init_weights": args.init_weights,
    }, args.output_path)
    print(f"Saved {args.mode} encoder to {args.output_path} "
          f"({elapsed:.1f}s for {args.num_episodes} episodes)")


if __name__ == "__main__":
    main()
