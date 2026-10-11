"""Shared DenoMAE2 construction/loading + CLI args for the Meta-Task scripts.

Mirrors the construction kwargs used in `src/evaluation/denomae_snr_eval.py`
and `src/denoMAE2/finetune.py` so checkpoints stay cross-compatible with the
existing finetuning pipeline.
"""
import argparse
from functools import partial

import torch
import torch.nn as nn
from torchvision import transforms

from src.denoMAE2.main import DenoMAE2


def add_denomae_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--decoder_embed_dim", type=int, default=512)
    parser.add_argument("--encoder_depth", type=int, default=12)
    parser.add_argument("--decoder_depth", type=int, default=8)
    parser.add_argument("--encoder_num_heads", type=int, default=12)
    parser.add_argument("--decoder_num_heads", type=int, default=16)
    parser.add_argument("--n_way", type=int, default=24, help="RadioML.2018.01a has 24 modulation classes")
    return parser


def build_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_denomae2(args, device) -> DenoMAE2:
    return DenoMAE2(
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


def reconstruction_loss(encoder: DenoMAE2, imgs, targets, mask_ratio):
    """Masked-reconstruction loss only, bypassing `DenoMAE2.forward()`.

    `DenoMAE2.forward()` also computes a patch-location loss via
    `forward_classification`, which has a pre-existing shape bug when
    `global_pool=False` (the default): it builds a one-hot target of shape
    [N, num_positions, nb_classes] and feeds it to `nn.CrossEntropyLoss`
    against logits of shape [N, nb_classes], which PyTorch rejects
    ("multi-target not supported"). That's a bug in the shared model file,
    not something to patch for this experiment, so the auxiliary loss here
    uses DenoMAE2's masked-reconstruction objective only.
    """
    latent, mask, ids_restore = encoder.forward_encoder(imgs, mask_ratio)
    pred = encoder.forward_decoder(latent, ids_restore)
    return encoder.forward_loss(targets, pred, mask)


def load_pretrained_encoder_weights(encoder: DenoMAE2, weights_path: str, device) -> DenoMAE2:
    """Loads weights from either a raw DenoMAE2 state dict or a
    `DownstreamClassifier` checkpoint (which prefixes encoder params with
    "encoder.").
    """
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    if any(k.startswith("encoder.") for k in state_dict):
        state_dict = {k[len("encoder."):]: v for k, v in state_dict.items() if k.startswith("encoder.")}

    missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
    print(f"Loaded encoder weights from {weights_path} "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    return encoder
