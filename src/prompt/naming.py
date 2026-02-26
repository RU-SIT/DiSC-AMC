"""Centralized naming conventions for prediction sources.

Every pipeline step — inference → conversion → dataset build → evaluation —
imports from this module so that filename patterns stay consistent and
are defined in exactly **one** place.

Adding a new prediction source
------------------------------
1.  Create a new :class:`PredictionSource` instance.
2.  Register it in :data:`SOURCES`.
3.  The rest of the pipeline picks it up automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


# ── Source definition ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PredictionSource:
    """Naming configuration for a single prediction source."""

    key: str
    """Short identifier used on the CLI (e.g. ``"dnn"``, ``"centroid"``)."""

    label: str
    """Human-readable description (e.g. ``"DNN classifier head"``)."""

    raw_json: str
    """Filename pattern for the *.png*-keyed prediction JSON
    (output of ``inference.py predict``).
    Must contain ``{topk}`` placeholder.
    Example: ``"top{topk}_dnn_predictions.json"``."""

    converted_json: str
    """Filename pattern for the *.npy*-keyed prediction JSON
    (output of ``convert_predictions.py``).
    Must contain ``{topk}`` placeholder.
    Example: ``"ntop{topk}_dnn_predictions.json"``."""

    pkl_tag: str
    """Tag inserted into ``.pkl`` output names.
    Empty string ``""`` for the default source (DNN) so that its
    filenames remain backward-compatible
    (``test_noisySignal_5_5_data.pkl``).
    Non-empty values produce e.g.
    ``test_centroid_noisySignal_5_5_data.pkl``."""


# ── Registry ─────────────────────────────────────────────────────────────────

SOURCES: Dict[str, PredictionSource] = {
    "dnn": PredictionSource(
        key="dnn",
        label="DNN classifier head",
        raw_json="top{topk}_dnn_predictions.json",
        converted_json="ntop{topk}_dnn_predictions.json",
        pkl_tag="",
    ),
    "centroid": PredictionSource(
        key="centroid",
        label="Nearest centroid",
        raw_json="top{topk}_centroid_predictions.json",
        converted_json="ntop{topk}_centroid_predictions.json",
        pkl_tag="centroid",
    ),
    "rf": PredictionSource(
        key="rf",
        label="Random Forest",
        raw_json="top{topk}_rf_predictions.json",
        converted_json="ntop{topk}_rf_predictions.json",
        pkl_tag="rf",
    ),
    "faiss": PredictionSource(
        key="faiss",
        label="FAISS kNN voting",
        raw_json="top{topk}_faiss_predictions.json",
        converted_json="ntop{topk}_faiss_predictions.json",
        pkl_tag="faiss",
    ),
}

VALID_SOURCES: List[str] = list(SOURCES.keys())


def get_source(name: str) -> PredictionSource:
    """Look up a prediction source by key.

    Raises
    ------
    ValueError
        If *name* is not registered in :data:`SOURCES`.
    """
    if name not in SOURCES:
        raise ValueError(
            f"Unknown prediction source {name!r}. "
            f"Choose from: {', '.join(VALID_SOURCES)}"
        )
    return SOURCES[name]


# ── Filename builders ────────────────────────────────────────────────────────

def raw_json_name(source: str, top_k: int) -> str:
    """Filename for the ``.png``-keyed prediction JSON (inference output).

    >>> raw_json_name("centroid", 5)
    'top5_centroid_predictions.json'
    """
    return get_source(source).raw_json.format(topk=top_k)


def converted_json_name(source: str, top_k: int) -> str:
    """Filename for the ``.npy``-keyed prediction JSON (after conversion).

    >>> converted_json_name("centroid", 5)
    'ntop5_centroid_predictions.json'
    """
    return get_source(source).converted_json.format(topk=top_k)


def train_pkl_name(
    source: str, noise_mode: str, n_bins: int, top_k: int,
) -> str:
    """Filename for the train ``.pkl`` dataset.

    >>> train_pkl_name("dnn", "noisySignal", 5, 5)
    'train_noisySignal_5_5_data.pkl'
    >>> train_pkl_name("centroid", "noisySignal", 5, 5)
    'train_centroid_noisySignal_5_5_data.pkl'
    """
    tag = get_source(source).pkl_tag
    if tag:
        return f"train_{tag}_{noise_mode}_{n_bins}_{top_k}_data.pkl"
    return f"train_{noise_mode}_{n_bins}_{top_k}_data.pkl"


def test_pkl_name(
    source: str, noise_mode: str, n_bins: int, top_k: int,
) -> str:
    """Filename for the test ``.pkl`` dataset.

    >>> test_pkl_name("dnn", "noisySignal", 5, 5)
    'test_noisySignal_5_5_data.pkl'
    >>> test_pkl_name("centroid", "noisySignal", 5, 5)
    'test_centroid_noisySignal_5_5_data.pkl'
    """
    tag = get_source(source).pkl_tag
    if tag:
        return f"test_{tag}_{noise_mode}_{n_bins}_{top_k}_data.pkl"
    return f"test_{noise_mode}_{n_bins}_{top_k}_data.pkl"
