"""
Embedding pipeline: DINO encoder → PCA → Discretization → JSON.

Extracts DINO encoder embeddings from spectrogram / constellation images,
reduces dimensionality with PCA, discretizes into bins, and persists results
as JSON.  PCA and discretizers are **fit on the train split** and **reused
on test / validation** for each ``feature_type`` independently.

Usage (standalone)::

    python embedding_pipeline.py \\
        --dataset_folder unlabeled_10k \\
        --weights ../exp/dino_classifier.pth \\
        --n_components 10 --n_bins 10

Design principles
-----------------
* **SRP** – each public function does exactly one thing.
* **OCP** – the pipeline is parameterised (model class, feature types,
  PCA / discretizer settings) so new variants need no code changes.
* **DRY** – shared logic (image loading, PCA, discretisation) is factored
  into small reusable helpers; ``_reduce_and_discretize`` handles both
  fit and transform modes via a single code-path.
* **KISS** – flat module, no unnecessary class hierarchies.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from glob import glob
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.spatial.distance import cdist as scipy_cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────────────

CLASSES: List[str] = [
    "OOK", "4ASK", "8ASK", "OQPSK", "CPFSK",
    "GFSK", "4PAM", "DQPSK", "16PAM", "GMSK",
]

FEATURE_TYPES: List[str] = ["noisyImg", "noiseLessImg"]

_DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ── Data‑classes for fitted transformers ─────────────────────────────────────

@dataclass
class FittedTransformers:
    """Holds PCA and KBinsDiscretizer objects keyed by feature_type."""
    pca: Dict[str, PCA] = field(default_factory=dict)
    discretizer: Dict[str, KBinsDiscretizer] = field(default_factory=dict)

# ── Label extraction ─────────────────────────────────────────────────────────

def extract_label(filename: str) -> str:
    """Return the modulation class from an image filename.

    Expects names like ``16PAM_0.01dB__0701_20250627_153146.png``.
    """
    return os.path.basename(filename).split("_")[0]

# ── Encoder loading ──────────────────────────────────────────────────────────

def load_encoder(
    model_class: Type[nn.Module],
    weights_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, torch.device]:
    """Instantiate *model_class*, load *weights_path*, return ``(encoder, device)``.

    The caller is expected to pass a class whose ``.encoder`` attribute is
    the feature extractor (e.g. ``DinoClassifier``).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_class()
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=False)

    encoder = model.encoder
    encoder.to(device).eval()
    return encoder, device

# ── Image → embedding ────────────────────────────────────────────────────────

def _load_images_as_batch(
    paths: List[str],
    transform: transforms.Compose = _DEFAULT_TRANSFORM,
) -> torch.Tensor:
    """Load a list of image paths into a single ``(B, C, H, W)`` tensor."""
    return torch.stack([
        transform(Image.open(p).convert("RGB")) for p in paths
    ])


def extract_embeddings(
    encoder: nn.Module,
    device: torch.device,
    image_dir: str,
    batch_size: int = 32,
) -> Tuple[List[str], np.ndarray]:
    """Run every ``.png`` in *image_dir* through *encoder*.

    Returns
    -------
    filenames : list[str]
        Base filenames in sorted order.
    embeddings : np.ndarray, shape ``(N, latent_dim)``
    """
    paths = sorted(glob(os.path.join(image_dir, "*.png")))
    if not paths:
        raise FileNotFoundError(f"No .png images in {image_dir}")

    filenames = [os.path.basename(p) for p in paths]
    parts: List[np.ndarray] = []

    for start in tqdm(
        range(0, len(paths), batch_size),
        desc=f"Encoding {os.path.basename(image_dir)}",
    ):
        batch = _load_images_as_batch(paths[start : start + batch_size])
        with torch.no_grad():
            emb = encoder(batch.to(device))  # (B, latent_dim)
        parts.append(emb.cpu().numpy())

    return filenames, np.concatenate(parts, axis=0)

# ── PCA + discretisation (single code path for fit / transform) ──────────────

def _reduce_and_discretize(
    embeddings: np.ndarray,
    n_components: int,
    n_bins: int,
    strategy: str,
    pca: Optional[PCA] = None,
    discretizer: Optional[KBinsDiscretizer] = None,
) -> Tuple[np.ndarray, PCA, KBinsDiscretizer]:
    """PCA‑reduce and discretize *embeddings*.

    If *pca* / *discretizer* are ``None`` they are fit on *embeddings*
    (train mode).  Otherwise they are applied as‑is (test mode).

    Returns ``(discretized, pca, discretizer)``.
    """
    # --- PCA ---
    if pca is None:
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(embeddings)
    else:
        reduced = pca.transform(embeddings)

    # --- Discretisation ---
    if discretizer is None:
        discretizer = KBinsDiscretizer(
            n_bins=n_bins, encode="ordinal", strategy=strategy,
        )
        discretized = discretizer.fit_transform(reduced).astype(int)
    else:
        discretized = discretizer.transform(reduced).astype(int)

    return discretized, pca, discretizer

# ── Per‑split processing ─────────────────────────────────────────────────────

def process_split(
    encoder: nn.Module,
    device: torch.device,
    split_path: str,
    feature_types: List[str] = FEATURE_TYPES,
    n_components: int = 10,
    n_bins: int = 10,
    strategy: str = "uniform",
    batch_size: int = 32,
    pca_dict: Optional[Dict[str, PCA]] = None,
    disc_dict: Optional[Dict[str, KBinsDiscretizer]] = None,
) -> Tuple[
    Dict[str, Dict[str, List[int]]],
    Dict[str, PCA],
    Dict[str, KBinsDiscretizer],
]:
    """Encode → PCA → discretize every image under *split_path*.

    Parameters
    ----------
    encoder, device
        DINO encoder returned by :func:`load_encoder`.
    split_path
        Root of one data split, e.g. ``data/own/unlabeled_10k/train``.
        Must contain sub‑directories named after each *feature_type*.
    feature_types
        Sub‑directories to iterate over (default ``['noisyImg', 'noiseLessImg']``).
    n_components, n_bins, strategy
        PCA and ``KBinsDiscretizer`` hyper‑parameters.
    batch_size
        Images per forward pass.
    pca_dict, disc_dict
        Pre‑fitted transformers keyed by feature_type.  Pass ``None``
        for the **train** split (they will be fit); pass the dicts
        returned by the train call for **test / val** splits.

    Returns
    -------
    results : dict
        ``{filename: {feature_type: [int, …], …}, …}``
    pca_dict : dict[str, PCA]
    disc_dict : dict[str, KBinsDiscretizer]
    """
    fit_mode = pca_dict is None
    if fit_mode:
        pca_dict, disc_dict = {}, {}

    results: Dict[str, Dict[str, List[int]]] = {}

    for ft in feature_types:
        image_dir = os.path.join(split_path, ft)
        filenames, embeddings = extract_embeddings(
            encoder, device, image_dir, batch_size,
        )

        discretized, pca, disc = _reduce_and_discretize(
            embeddings,
            n_components=n_components,
            n_bins=n_bins,
            strategy=strategy,
            pca=pca_dict.get(ft),
            discretizer=disc_dict.get(ft),
        )

        if fit_mode:
            pca_dict[ft] = pca
            disc_dict[ft] = disc

        for fname, vec in zip(filenames, discretized):
            results.setdefault(fname, {})[ft] = vec.tolist()

    return results, pca_dict, disc_dict

# ── I/O ──────────────────────────────────────────────────────────────────────

def save_results(
    results: Dict[str, Dict[str, List[int]]],
    output_path: str,
) -> None:
    """Persist *results* as a JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(results, fh)
    print(f"Saved {len(results)} entries → {output_path}")


def load_results(path: str) -> Dict[str, Dict[str, List[int]]]:
    """Load a previously saved feature JSON."""
    with open(path) as fh:
        return json.load(fh)


# ── Centroid computation ─────────────────────────────────────────────────────

def compute_class_centroids(
    encoder: nn.Module,
    device: torch.device,
    dataset: "DataLoader | torch.utils.data.Dataset",
    classes: List[str],
    batch_size: int = 32,
    num_workers: int = 4,
) -> Dict[str, np.ndarray]:
    """Extract encoder features for every sample and compute per-class centroids.

    Parameters
    ----------
    encoder
        Feature extractor (e.g. ``model.encoder``), already on *device*.
    device
        Torch device.
    dataset
        A dataset that yields ``(image_tensor, label_int, path_str)`` triples
        (e.g. :class:`DatasetWithPath`).  Alternatively pass a ready-made
        ``DataLoader``.
    classes
        Ordered list of class names so that ``classes[label_int]`` gives the
        human-readable name.
    batch_size, num_workers
        DataLoader parameters (ignored when *dataset* is already a DataLoader).

    Returns
    -------
    centroids : dict[str, np.ndarray]
        ``{class_name: mean_feature_vector}`` for every class present in the
        dataset.
    """
    if isinstance(dataset, DataLoader):
        loader = dataset
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)

    features_by_class: Dict[int, List[np.ndarray]] = {i: [] for i in range(len(classes))}

    encoder.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing centroids"):
            images = batch[0].to(device)
            labels = batch[1]
            feats = encoder(images)
            for i in range(len(labels)):
                features_by_class[labels[i].item()].append(feats[i].cpu().numpy())

    centroids: Dict[str, np.ndarray] = {}
    for idx, name in enumerate(classes):
        if features_by_class[idx]:
            centroids[name] = np.mean(features_by_class[idx], axis=0)
    return centroids


def find_closest_to_centroids(
    encoder: nn.Module,
    device: torch.device,
    dataset,
    classes: List[str],
    centroids: Dict[str, np.ndarray],
    batch_size: int = 32,
    num_workers: int = 4,
) -> Dict[str, str]:
    """Find the dataset sample closest (Euclidean) to each class centroid.

    Parameters
    ----------
    encoder, device
        As in :func:`compute_class_centroids`.
    dataset
        Must yield ``(image, label, path)`` triples.
    classes
        Ordered class names.
    centroids
        Output of :func:`compute_class_centroids`.

    Returns
    -------
    closest : dict[str, str]
        ``{class_name: path_of_closest_sample}``.
    """
    if isinstance(dataset, DataLoader):
        loader = dataset
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)

    features_by_class: Dict[int, List[np.ndarray]] = {i: [] for i in range(len(classes))}
    paths_by_class: Dict[int, List[str]] = {i: [] for i in range(len(classes))}

    encoder.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features"):
            images = batch[0].to(device)
            labels = batch[1]
            paths = batch[2]
            feats = encoder(images)
            for i in range(len(labels)):
                idx = labels[i].item()
                features_by_class[idx].append(feats[i].cpu().numpy())
                paths_by_class[idx].append(paths[i])

    closest: Dict[str, str] = {}
    for idx, name in enumerate(classes):
        if name not in centroids or not features_by_class[idx]:
            continue
        center = centroids[name].reshape(1, -1)
        feats = np.array(features_by_class[idx])
        distances = scipy_cdist(feats, center, "euclidean")
        closest_idx = int(np.argmin(distances))
        closest[name] = paths_by_class[idx][closest_idx]

    return closest


def save_centroids(
    centroids: Dict[str, np.ndarray],
    output_path: str,
) -> None:
    """Save class centroids as JSON (converts numpy arrays to lists)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    serialisable = {k: v.tolist() for k, v in centroids.items()}
    with open(output_path, "w") as fh:
        json.dump(serialisable, fh, indent=4)
    print(f"Saved {len(centroids)} centroids → {output_path}")


def load_centroids(
    path: str,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, List[str]]:
    """Load centroids JSON and return a ``(tensor, class_names)`` pair.

    Parameters
    ----------
    path
        JSON file saved by :func:`save_centroids`.
    device
        Target device for the tensor (defaults to CPU).

    Returns
    -------
    centroid_tensor : torch.Tensor
        Shape ``(num_classes, latent_dim)``.
    class_names : list[str]
        Class names matching row order of the tensor.
    """
    with open(path) as fh:
        data = json.load(fh)
    class_names = list(data.keys())
    tensor = torch.tensor(
        np.array([data[c] for c in class_names]), dtype=torch.float32,
    )
    if device is not None:
        tensor = tensor.to(device)
    return tensor, class_names


# ── CLI entry‑point ──────────────────────────────────────────────────────────

def run_pipeline(
    dataset_folder: str = "unlabeled_10k",
    weights_path: str = "../exp/dino_classifier.pth",
    data_root: str = "../data/own",
    n_components: int = 10,
    n_bins: int = 10,
    strategy: str = "uniform",
    batch_size: int = 32,
    model_class: Optional[Type[nn.Module]] = None,
) -> Dict[str, Dict[str, List[int]]]:
    """End‑to‑end: load model → process train & test → save JSON.

    Returns the merged results dict.
    """
    base = os.path.join(data_root, dataset_folder)
    train_path = os.path.join(base, "train")
    test_path = os.path.join(base, "test")
    output_json = os.path.join(base, "dino_features.json")

    # --- lazy import to avoid circular deps when used as a library ---
    if model_class is None:
        raise ValueError(
            "model_class must be provided (e.g. DinoClassifier). "
            "Pass it explicitly or use the notebook entry‑point."
        )

    encoder, device = load_encoder(model_class, weights_path)
    print(f"Encoder loaded on {device}")

    print("Processing TRAIN split …")
    train_results, pca_dict, disc_dict = process_split(
        encoder, device, train_path,
        n_components=n_components,
        n_bins=n_bins,
        strategy=strategy,
        batch_size=batch_size,
    )

    print("\nProcessing TEST split …")
    test_results, _, _ = process_split(
        encoder, device, test_path,
        n_components=n_components,
        n_bins=n_bins,
        strategy=strategy,
        batch_size=batch_size,
        pca_dict=pca_dict,
        disc_dict=disc_dict,
    )

    all_results = {**train_results, **test_results}
    print(f"\nTotal images processed: {len(all_results)}")

    save_results(all_results, output_json)
    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DINO Embedding → PCA → Discretization pipeline",
    )
    parser.add_argument("--dataset_folder", default="unlabeled_10k")
    parser.add_argument("--weights", default="../exp/dino_classifier.pth")
    parser.add_argument("--data_root", default="../data/own")
    parser.add_argument("--n_components", type=int, default=10)
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--strategy", default="uniform")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # Import here so the module stays free of model‑specific deps at top level.
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "spectrogram"))
    from autoencoder_vit import DinoV2Autoencoder  # noqa: E402

    class DinoClassifier(nn.Module):
        """Thin wrapper: DINO autoencoder encoder + classification head."""

        def __init__(
            self,
            num_classes: int = len(CLASSES),
            freeze_encoder: bool = False,
        ):
            super().__init__()
            autoencoder = DinoV2Autoencoder(freeze_encoder=freeze_encoder)
            self.encoder = autoencoder.encoder
            self.classifier_head = nn.Sequential(
                nn.LayerNorm(768),
                nn.Linear(768, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.classifier_head(self.encoder(x))

    run_pipeline(
        dataset_folder=args.dataset_folder,
        weights_path=args.weights,
        data_root=args.data_root,
        n_components=args.n_components,
        n_bins=args.n_bins,
        strategy=args.strategy,
        batch_size=args.batch_size,
        model_class=DinoClassifier,
    )
