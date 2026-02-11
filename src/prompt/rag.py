"""
RAG (Retrieval-Augmented Generation) module for DiSC-AMC.

Builds a FAISS vector index over training signal features and retrieves
the most similar training examples for each test signal at prompt-generation
time.  This replaces the hardcoded / random few-shot example selection with
**similarity-based** retrieval, providing the LLM with the most relevant
context for each query.

Design
------
The module is intentionally **optional**.  When ``--use_rag`` is *not* passed,
the pipeline falls back to the original ``create_dataset_example_paths()`` /
``reduce_example_dict()`` logic unchanged.

Index lifecycle
^^^^^^^^^^^^^^^
1. **build_index** (train time) — compute feature vectors for every training
   signal, build a FAISS ``IndexFlatL2`` (exact search) or ``IndexIVFFlat``
   (approximate, for very large datasets), and persist the index + metadata
   to disk next to the train ``.pkl``.
2. **load_index** (test time) — memory-map the index and metadata.
3. **retrieve** — given a test signal's feature vector, return the *k*
   nearest training signals grouped as an ``example_dict`` compatible with
   ``generate_prompt()``.

Dependencies
^^^^^^^^^^^^
Only ``faiss-cpu`` (or ``faiss-gpu``) is required on top of the existing
environment.  Install with::

    pip install faiss-cpu   # or faiss-gpu for CUDA support
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# -- lazy FAISS import ---------------------------------------------------------
_faiss = None


def _import_faiss():
    """Import FAISS lazily so the rest of the codebase works without it."""
    global _faiss
    if _faiss is None:
        try:
            import faiss
            _faiss = faiss
        except ImportError:
            raise ImportError(
                "RAG mode requires the `faiss` package.\n"
                "Install it with:  pip install faiss-cpu   (or faiss-gpu)"
            )
    return _faiss


# -- Index file naming ---------------------------------------------------------

def _index_path(train_pkl_path: str) -> str:
    """Derive the FAISS index path from the train pkl path."""
    return train_pkl_path.replace("_data.pkl", "_rag.index")


def _meta_path(train_pkl_path: str) -> str:
    """Derive the RAG metadata (signal paths + labels) path."""
    return train_pkl_path.replace("_data.pkl", "_rag_meta.pkl")


# -- Data class ----------------------------------------------------------------

@dataclass
class RAGRetriever:
    """Wraps a FAISS index and the associated training metadata.

    Attributes
    ----------
    index : faiss.Index
        The FAISS nearest-neighbour index.
    signal_paths : list[str]
        Absolute paths to the indexed training ``.npy`` files, aligned with
        the FAISS index rows.
    labels : list[str]
        Class labels for each indexed signal.
    snrs : list[str]
        SNR strings for each indexed signal.
    feature_vectors : np.ndarray
        The (N, D) matrix of feature vectors that were indexed (kept for
        optional re-use, e.g. debugging or re-indexing).
    """
    index: Any                    # faiss.Index
    signal_paths: List[str]
    labels: List[str]
    snrs: List[str]
    feature_vectors: np.ndarray   # (N, D) float32


# -- Build ---------------------------------------------------------------------

def build_rag_index(
    signal_paths: List[str],
    labels: List[str],
    snrs: List[str],
    feature_vectors: np.ndarray,
    train_pkl_path: str,
    use_ivf: bool = False,
    nlist: int = 100,
) -> RAGRetriever:
    """Build a FAISS index from training feature vectors and persist to disk.

    Parameters
    ----------
    signal_paths : list[str]
        Training signal file paths.
    labels, snrs : list[str]
        Per-signal class label and SNR.
    feature_vectors : np.ndarray
        ``(N, D)`` array of *scaled* feature vectors (the same vectors that
        go through ``StandardScaler``).
    train_pkl_path : str
        Path to the train ``.pkl`` -- used to derive the index/meta filenames.
    use_ivf : bool
        If True, use ``IndexIVFFlat`` for approximate search (faster on
        very large datasets, but requires training the index quantiser).
    nlist : int
        Number of Voronoi cells for IVF (ignored when ``use_ivf=False``).

    Returns
    -------
    RAGRetriever
        The built retriever (also saved to disk).
    """
    faiss = _import_faiss()

    vecs = np.ascontiguousarray(feature_vectors.astype(np.float32))
    d = vecs.shape[1]

    if use_ivf and vecs.shape[0] > nlist:
        quantiser = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantiser, d, min(nlist, vecs.shape[0]))
        index.train(vecs)
        index.add(vecs)
        index.nprobe = min(10, nlist)  # search 10 cells by default
    else:
        index = faiss.IndexFlatL2(d)
        index.add(vecs)

    # Persist
    idx_path = _index_path(train_pkl_path)
    meta_path_ = _meta_path(train_pkl_path)

    faiss.write_index(index, idx_path)
    with open(meta_path_, "wb") as f:
        pickle.dump({
            "signal_paths": signal_paths,
            "labels": labels,
            "snrs": snrs,
            "feature_vectors": vecs,
        }, f)

    print(f"RAG index saved -> {idx_path}  ({index.ntotal} vectors, dim={d})")
    print(f"RAG meta  saved -> {meta_path_}")

    return RAGRetriever(
        index=index,
        signal_paths=signal_paths,
        labels=labels,
        snrs=snrs,
        feature_vectors=vecs,
    )


# -- Load ----------------------------------------------------------------------

def load_rag_index(train_pkl_path: str) -> RAGRetriever:
    """Load a previously built RAG index + metadata from disk.

    Parameters
    ----------
    train_pkl_path : str
        Same path used in :func:`build_rag_index` (the train ``.pkl``).

    Returns
    -------
    RAGRetriever
    """
    faiss = _import_faiss()

    idx_path = _index_path(train_pkl_path)
    meta_path_ = _meta_path(train_pkl_path)

    if not os.path.isfile(idx_path):
        raise FileNotFoundError(
            f"RAG index not found: {idx_path}\n"
            "Run with --mode train --use_rag first to build the index."
        )
    if not os.path.isfile(meta_path_):
        raise FileNotFoundError(f"RAG metadata not found: {meta_path_}")

    index = faiss.read_index(idx_path)
    with open(meta_path_, "rb") as f:
        meta = pickle.load(f)

    print(f"RAG index loaded <- {idx_path}  ({index.ntotal} vectors)")

    return RAGRetriever(
        index=index,
        signal_paths=meta["signal_paths"],
        labels=meta["labels"],
        snrs=meta["snrs"],
        feature_vectors=meta["feature_vectors"],
    )


# -- Retrieve ------------------------------------------------------------------

def retrieve_examples(
    retriever: RAGRetriever,
    query_vector: np.ndarray,
    rag_k: int = 10,
    min_classes: int = 0,
    exclude_same_path: Optional[str] = None,
) -> Dict[str, List[Tuple[str, str]]]:
    """Retrieve the *rag_k* nearest training signals for one test query.

    Parameters
    ----------
    retriever : RAGRetriever
        The loaded retriever.
    query_vector : np.ndarray
        ``(D,)`` scaled feature vector of the test signal.
    rag_k : int
        How many nearest neighbours to retrieve.
    min_classes : int
        If > 0, keep expanding the search until at least this many distinct
        classes appear among the retrieved examples (up to 3x ``rag_k``).
    exclude_same_path : str or None
        If provided, drop any retrieved signal whose path matches (useful
        when the test signal might also be in the training set).

    Returns
    -------
    dict[str, list[tuple[str, str]]]
        ``{class_label: [(signal_path, snr), ...]}``, ready to be loaded
        into the ``example_dict`` format used by ``generate_prompt()``.
    """
    q = np.ascontiguousarray(query_vector.reshape(1, -1).astype(np.float32))

    actual_k = min(rag_k, retriever.index.ntotal)
    # Optionally widen search for class diversity
    search_k = actual_k
    if min_classes > 0:
        search_k = min(actual_k * 3, retriever.index.ntotal)

    _, indices = retriever.index.search(q, search_k)
    indices = indices[0]  # shape (search_k,)

    result: Dict[str, List[Tuple[str, str]]] = {}
    count = 0
    for idx in indices:
        if idx < 0:
            continue  # FAISS pads with -1 when fewer results exist
        path = retriever.signal_paths[idx]
        if exclude_same_path and os.path.abspath(path) == os.path.abspath(exclude_same_path):
            continue
        label = retriever.labels[idx]
        snr = retriever.snrs[idx]
        if label not in result:
            result[label] = []
        result[label].append((path, snr))
        count += 1
        if count >= actual_k:
            # Check diversity constraint
            if min_classes <= 0 or len(result) >= min_classes:
                break

    return result


def rag_example_dict_from_paths(
    retrieved: Dict[str, List[Tuple[str, str]]],
) -> Dict[str, List[Tuple[np.ndarray, str]]]:
    """Convert retrieved ``{label: [(path, snr), ...]}`` into the format
    expected by ``generate_prompt()``: ``{label: [(signal_array, snr), ...]}``.

    This loads each ``.npy`` file and splits into real/imaginary columns,
    matching the existing ``all_example_dict`` structure.
    """
    from data_processing import load_npy_file, split_real_imaginary

    example_dict: Dict[str, List[Tuple[np.ndarray, str]]] = {}
    for label, items in retrieved.items():
        example_dict[label] = [
            (split_real_imaginary(load_npy_file(path)), snr)
            for path, snr in items
        ]
    return example_dict


# -- Batch helper (used in get_processed_data) ---------------------------------

def retrieve_example_dict_for_signal(
    retriever: RAGRetriever,
    query_feature_vector: np.ndarray,
    rag_k: int = 10,
    signal_path: Optional[str] = None,
) -> Dict[str, List[Tuple[np.ndarray, str]]]:
    """End-to-end: retrieve + load signals -> ``example_dict``.

    Parameters
    ----------
    retriever : RAGRetriever
        Loaded RAG retriever.
    query_feature_vector : np.ndarray
        The (D,) scaled feature array of the test signal.
    rag_k : int
        Number of neighbours to retrieve.
    signal_path : str or None
        Path of the current test signal (excluded from results to avoid
        self-retrieval).

    Returns
    -------
    dict
        ``{label: [(signal_array, snr), ...]}`` -- same format as the
        ``all_example_dict`` used by the existing pipeline.
    """
    retrieved_paths = retrieve_examples(
        retriever,
        query_feature_vector,
        rag_k=rag_k,
        exclude_same_path=signal_path,
    )
    return rag_example_dict_from_paths(retrieved_paths)
