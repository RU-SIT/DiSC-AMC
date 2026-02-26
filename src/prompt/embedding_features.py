"""
Embedding-based feature extraction for LLM prompts.

Bridges the representation-learning encoder pipeline with the prompt
generation pipeline.  Instead of computing statistical features (moments,
cumulants, kurtosis, …) from raw ``.npy`` signals, this module:

1. Maps signal ``.npy`` paths to corresponding ``.png`` constellation images.
2. Extracts encoder embeddings (DINO ViT-B/8 or ResNet-34/50).
3. PCA-reduces the high-dimensional embeddings to *n_components*.
4. Converts PCA components into feature dicts compatible with the existing
   discretization → letter-encoding → prompt-generation flow.

The module is intentionally **model-agnostic**: callers pass a pre-loaded
encoder (any ``nn.Module`` mapping ``(B, C, H, W) → (B, latent_dim)``).

Usage
-----
::

    from embedding_features import (
        signal_path_to_image_path,
        extract_embeddings_from_paths,
        compute_embedding_features,
        prepare_example_embedding_dicts,
    )

    # 1. Map signal paths to image paths
    img_paths = [signal_path_to_image_path(p, "noisySignal") for p in signal_paths]

    # 2. Extract and process
    embeddings = extract_embeddings_from_paths(encoder, device, img_paths)
    (feat_dicts, disc_dicts, scaled_dicts,
     feat_names, pca, discretizers, scaler) = compute_embedding_features(
        embeddings, n_components=10, n_bins=5,
    )

    # 3. Pre-process few-shot examples
    scaled_ex, disc_ex = prepare_example_embedding_dicts(
        encoder, device, example_paths, "noisySignal",
        n_components=10, pca=pca, discretizers=discretizers, scaler=scaler,
    )
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from torchvision import transforms
from tqdm import tqdm

from .data_processing import (
    dict_to_np,
    discretize_features,
    get_discrete_info,
    get_scaled_info,
)


# ── Constants ────────────────────────────────────────────────────────────────

NOISE_MODE_TO_IMG: Dict[str, str] = {
    "noisySignal": "noisyImg",
    "noiselessSignal": "noiseLessImg",
}

_DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ── Path mapping ─────────────────────────────────────────────────────────────

def signal_path_to_image_path(signal_path: str, noise_mode: str) -> str:
    """Convert a ``.npy`` signal path to its ``.png`` constellation image path.

    The signal and image directories are siblings under the same split root::

        data/own/unlabeled_10k/test/
        ├── noisySignal/   ← .npy signals
        └── noisyImg/      ← .png constellation images

    Example
    -------
    >>> signal_path_to_image_path(
    ...     "data/own/unlabeled_10k/test/noisySignal/OOK_2.17dB__0379.npy",
    ...     "noisySignal",
    ... )
    'data/own/unlabeled_10k/test/noisyImg/OOK_2.17dB__0379.png'
    """
    if noise_mode not in NOISE_MODE_TO_IMG:
        raise ValueError(
            f"Unknown noise_mode={noise_mode!r}. "
            f"Expected one of {list(NOISE_MODE_TO_IMG)}."
        )
    img_dir = NOISE_MODE_TO_IMG[noise_mode]
    parent = os.path.dirname(signal_path)   # .../test/noisySignal
    split_root = os.path.dirname(parent)    # .../test
    basename = os.path.splitext(os.path.basename(signal_path))[0] + ".png"
    return os.path.join(split_root, img_dir, basename)


# ── Feature naming ───────────────────────────────────────────────────────────

def pca_feature_names(n_components: int) -> List[str]:
    """Return canonical feature names for PCA components.

    >>> pca_feature_names(3)
    ['pc_0', 'pc_1', 'pc_2']
    """
    return [f"pc_{i}" for i in range(n_components)]


def embedding_to_feature_dict(
    vector: np.ndarray,
    feature_names: List[str],
) -> Dict[str, Union[float, int, np.ndarray]]:
    """Create a feature dict from a flat vector (e.g. PCA-reduced embedding).

    >>> embedding_to_feature_dict(np.array([0.1, 0.2]), ["pc_0", "pc_1"])
    {'pc_0': 0.1, 'pc_1': 0.2}
    """
    return {name: float(val) for name, val in zip(feature_names, vector)}


# ── Image loading & embedding extraction ─────────────────────────────────────

def _load_images_as_batch(
    paths: List[str],
    transform: transforms.Compose = _DEFAULT_TRANSFORM,
) -> torch.Tensor:
    """Load image *paths* into a single ``(B, C, H, W)`` tensor."""
    return torch.stack([
        transform(Image.open(p).convert("RGB")) for p in paths
    ])


def extract_embeddings_from_paths(
    encoder: nn.Module,
    device: torch.device,
    image_paths: List[str],
    batch_size: int = 32,
    verbose: bool = True,
) -> np.ndarray:
    """Run every image in *image_paths* through *encoder*.

    Handles both DINO (``(B, latent_dim)`` output) and ResNet
    (``(B, C, H, W)`` output → adaptive avg-pool + flatten).

    Parameters
    ----------
    encoder
        Any ``nn.Module`` mapping ``(B, C, H, W) → (B, latent_dim)``
        *or* ``(B, C, H, W) → (B, C', H', W')``.
    device
        Torch device the encoder lives on.
    image_paths
        Absolute paths to ``.png`` files.
    batch_size
        Images per forward pass.

    Returns
    -------
    np.ndarray, shape ``(N, latent_dim)``
    """
    parts: List[np.ndarray] = []
    for start in tqdm(
        range(0, len(image_paths), batch_size),
        desc="Extracting embeddings",
        disable=not verbose,
    ):
        batch = _load_images_as_batch(image_paths[start : start + batch_size])
        with torch.no_grad():
            emb = encoder(batch.to(device))            # (B, latent_dim) or (B, C, H, W)
        # Handle 4D output (ResNet feature maps) → pool + flatten
        if emb.dim() == 4:
            emb = torch.nn.functional.adaptive_avg_pool2d(emb, (1, 1)).flatten(1)
        parts.append(emb.cpu().numpy())
    return np.concatenate(parts, axis=0)


# ── Full pipeline: embeddings → feature dicts ────────────────────────────────

def compute_embedding_features(
    embeddings: np.ndarray,
    n_components: int,
    n_bins: int,
    pca: Optional[PCA] = None,
    discretizers: Optional[Dict[int, KBinsDiscretizer]] = None,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[
    List[Dict[str, float]],                         # feature_dicts  (raw PCA values)
    List[Dict[str, Union[int, np.ndarray]]],        # discretized_dicts
    List[Dict[str, Union[float, np.ndarray]]],      # scaled_dicts
    List[str],                                       # feature_names
    PCA,
    Dict[int, KBinsDiscretizer],
    StandardScaler,
]:
    """PCA-reduce, discretize, and scale embedding vectors.

    If *pca* / *discretizers* / *scaler* are ``None`` they are **fit** on
    *embeddings* (train mode).  Otherwise they are applied as-is (test mode).

    Returns
    -------
    feature_dicts
        Raw PCA component values as ``{pc_0: float, …}`` dicts.
    discretized_dicts
        Discretized (bin-index) versions, compatible with
        :func:`data_processing.get_discrete_text_info`.
    scaled_dicts
        StandardScaler-transformed versions, compatible with
        :func:`data_processing.get_text_info`.
    feature_names
        ``['pc_0', …, 'pc_{n-1}']``.
    pca, discretizers, scaler
        Fitted transformers (persist these for the test split).
    """
    # ── PCA ───────────────────────────────────────────────────────────────
    if pca is None:
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(embeddings)
    else:
        reduced = pca.transform(embeddings)

    feature_names = pca_feature_names(n_components)

    # Convert each row to a feature dict
    feature_dicts = [
        embedding_to_feature_dict(row, feature_names) for row in reduced
    ]

    # 2-D array for batch discretization & scaling
    feature_array = np.array(
        [dict_to_np(d, feature_names) for d in feature_dicts]
    )

    # ── Discretize (per column, same as statistical pipeline) ─────────────
    if discretizers is None:
        discretizers = {}
        _, discretizers = discretize_features(
            feature_array, n_bins=n_bins, strategy="uniform",
        )
    discretized_dicts = [
        get_discrete_info(d, discretizers) for d in feature_dicts
    ]

    # ── Scale (StandardScaler) ────────────────────────────────────────────
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(feature_array)
    scaled_dicts = [
        get_scaled_info(d, scaler) for d in feature_dicts
    ]

    return (
        feature_dicts,
        discretized_dicts,
        scaled_dicts,
        feature_names,
        pca,
        discretizers,
        scaler,
    )


# ── Example pre-processing ──────────────────────────────────────────────────

def prepare_example_embedding_dicts(
    encoder: nn.Module,
    device: torch.device,
    example_paths: Dict[str, List[str]],
    noise_mode: str,
    n_components: int,
    pca: PCA,
    discretizers: Dict[int, KBinsDiscretizer],
    scaler: StandardScaler,
    batch_size: int = 32,
    verbose: bool = True,
) -> Tuple[Dict[str, list], Dict[str, list]]:
    """Pre-process example signals into embedding feature dicts.

    Creates two example dictionaries (one scaled, one discretized) that can
    be passed to ``generate_prompt()`` with ``examples_processed=True``.
    Each dict has the structure ``{class_name: [(feature_dict, snr), …]}``.

    Parameters
    ----------
    example_paths
        ``{class_name: [signal_path_1, …], …}`` — raw ``.npy`` signal paths.
    noise_mode
        ``"noisySignal"`` or ``"noiselessSignal"`` — used to resolve image paths.
    pca, discretizers, scaler
        **Already-fitted** transformers from the training set.

    Returns
    -------
    scaled_example_dict
        ``{class_name: [(scaled_feature_dict, snr_str), …], …}``
    discretized_example_dict
        ``{class_name: [(discretized_feature_dict, snr_str), …], …}``
    """
    # Avoid circular import — only need a tiny helper
    from .generated_dataset import get_dataset_snr

    feature_names = pca_feature_names(n_components)

    # ── Collect ALL example image paths + metadata ────────────────────────
    all_image_paths: List[str] = []
    all_meta: List[Tuple[str, str]] = []        # (class_name, snr_str)
    for cls, paths in example_paths.items():
        for sig_path in paths:
            img_path = signal_path_to_image_path(sig_path, noise_mode)
            snr = get_dataset_snr(sig_path)
            all_image_paths.append(img_path)
            all_meta.append((cls, snr))

    if not all_image_paths:
        return {}, {}

    # ── Batch-extract embeddings → PCA → scale / discretize ──────────────
    embeddings = extract_embeddings_from_paths(
        encoder, device, all_image_paths, batch_size, verbose=verbose,
    )
    reduced = pca.transform(embeddings)

    feature_dicts = [
        embedding_to_feature_dict(row, feature_names) for row in reduced
    ]
    scaled_dicts = [get_scaled_info(d, scaler) for d in feature_dicts]
    discretized_dicts = [get_discrete_info(d, discretizers) for d in feature_dicts]

    # ── Group back by class ──────────────────────────────────────────────
    scaled_example_dict: Dict[str, list] = {}
    discretized_example_dict: Dict[str, list] = {}
    for i, (cls, snr) in enumerate(all_meta):
        scaled_example_dict.setdefault(cls, []).append(
            (scaled_dicts[i], snr)
        )
        discretized_example_dict.setdefault(cls, []).append(
            (discretized_dicts[i], snr)
        )

    return scaled_example_dict, discretized_example_dict


# ── Encoder loading helper ───────────────────────────────────────────────────

def load_encoder_for_embeddings(
    backbone: str,
    weights_path: str,
    num_classes: int = 10,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, torch.device]:
    """Load a trained classifier and return its encoder sub-module.

    Dynamically imports :class:`ImageClassifier` from the
    ``representation learning`` directory.

    Parameters
    ----------
    backbone
        ``"dino"`` or ``"resnet"`` — passed to ``ImageClassifier``.
    weights_path
        Path to the saved ``.pth`` checkpoint.
    num_classes
        Number of output classes the classifier was trained with.
    device
        Target device.  ``None`` → auto-detect CUDA / CPU.

    Returns
    -------
    encoder : nn.Module
        The ``.encoder`` sub-module, on *device*, in eval mode.
    device : torch.device
    """
    from src.representation_learning.classifier_training import ImageClassifier

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ImageClassifier(
        backbone=backbone, num_classes=num_classes,
    )
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=False)

    encoder = model.encoder
    encoder.to(device).eval()
    return encoder, device
