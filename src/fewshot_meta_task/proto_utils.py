"""Shared prototype-classification helpers for episodic training/eval.

Feature pooling mirrors `src/denoMAE2/finetune.py`'s `DownstreamClassifier`
(mean-pool over patch tokens from `forward_encoder`), so encoder-level
accuracy numbers stay comparable to the existing finetuned-classifier
baseline.
"""
import torch


def encode_mean_pool(encoder, x, mask_ratio=0.0):
    features, _, _ = encoder.forward_encoder(x, mask_ratio=mask_ratio)
    return features[:, 1:, :].mean(dim=1)


def prototype_logits(support_feats, support_labels, query_feats, n_way):
    prototypes = torch.stack([
        support_feats[support_labels == c].mean(dim=0) for c in range(n_way)
    ])
    dists = torch.cdist(query_feats, prototypes)
    return -dists


def accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()
