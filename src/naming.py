"""Centralized naming conventions for the DiSC-AMC pipeline.

Every pipeline step — inference → conversion → dataset build → evaluation —
imports from this module so that filename patterns stay consistent and
are defined in exactly **one** place.

Core abstraction
----------------
:class:`ExperimentConfig` captures *all* experiment dimensions (prediction
source, OOD flag, feature type, RAG, …).  Every filename builder receives
a config so that output files never collide across configurations.

Adding a new prediction source
------------------------------
1.  Create a new :class:`PredictionSource` instance.
2.  Register it in :data:`SOURCES`.
3.  The rest of the pipeline picks it up automatically.

Adding a new experiment dimension
---------------------------------
1.  Add a field to :class:`ExperimentConfig` (with a sensible default).
2.  Update :meth:`ExperimentConfig.build_tag` to include it when non-default.
3.  Callers don't need to change unless they use the new dimension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


# ── Experiment configuration ────────────────────────────────────────────────

@dataclass(frozen=True)
class ExperimentConfig:
    """All experiment dimensions that affect output filenames.

    Only non-default values appear in the filename tag (via
    :meth:`build_tag`) so that simple runs produce short filenames.

    Examples
    --------
    >>> cfg = ExperimentConfig(
    ...     dataset_folder="-11_-15dB",
    ...     prediction_source="centroid",
    ...     noise_mode="noisySignal", n_bins=5, top_k=5,
    ... )
    >>> cfg.build_tag()
    'centroid'

    >>> cfg2 = ExperimentConfig(
    ...     dataset_folder="-11_-15dB",
    ...     prediction_source="centroid",
    ...     noise_mode="noisySignal", n_bins=5, top_k=5,
    ...     feature_type="embeddings", n_components=10,
    ...     ood_train_folder="unlabeled_10k",
    ...     use_rag=True, rag_k=10,
    ... )
    >>> cfg2.build_tag()
    'centroid_ood_emb10_rag10'
    """

    dataset_folder: str
    """Dataset directory name, e.g. ``"-11_-15dB"``."""

    prediction_source: str = "dnn"
    """Which shortlisting source: ``"dnn"`` | ``"centroid"`` | ``"rf"``."""

    noise_mode: str = "noisySignal"
    """``"noisySignal"`` or ``"noiselessSignal"``."""

    n_bins: int = 5
    """Number of discretisation bins."""

    top_k: int = 5
    """Number of top-k predictions."""

    feature_type: str = "stats"
    """``"stats"`` (statistical features) or ``"embeddings"``."""

    n_components: int = 0
    """PCA components (only relevant when ``feature_type="embeddings"``)."""

    ood_train_folder: str = ""
    """Non-empty when the train data comes from a different folder (OOD).
    The value is the train dataset folder name."""

    use_rag: bool = False
    """Whether RAG retrieval was used for example selection."""

    rag_k: int = 0
    """Number of RAG neighbours (only relevant when ``use_rag=True``)."""

    def build_tag(self) -> str:
        """Build a compact tag string from non-default dimensions.

        The tag is inserted into filenames between the mode prefix and the
        noise/bins/topk segment.  Parts are joined with ``_``.

        Returns
        -------
        str
            Empty string for a default (dnn, stats, in-dist, no-RAG) run.
        """
        parts: List[str] = []

        # prediction source (omit default "dnn" → backward compat)
        src = get_source(self.prediction_source)
        if src.pkl_tag:
            parts.append(src.pkl_tag)

        # OOD
        if self.ood_train_folder:
            parts.append("ood")

        # Embeddings
        if self.feature_type == "embeddings" and self.n_components > 0:
            parts.append(f"emb{self.n_components}")

        # RAG
        if self.use_rag and self.rag_k > 0:
            parts.append(f"rag{self.rag_k}")

        return "_".join(parts)


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


def _pkl_name(mode: str, cfg: ExperimentConfig) -> str:
    """Internal helper — build a ``.pkl`` dataset filename.

    Pattern: ``{mode}[_{tag}]_{noise}_{bins}_{topk}_data.pkl``
    """
    tag = cfg.build_tag()
    parts = [mode]
    if tag:
        parts.append(tag)
    parts += [cfg.noise_mode, str(cfg.n_bins), str(cfg.top_k), "data.pkl"]
    return "_".join(parts)


def train_pkl_name(
    prediction_source: str,
    noise_mode: str,
    n_bins: int,
    top_k: int,
    feature_tag: str = "",
    *,
    cfg: ExperimentConfig | None = None,
) -> str:
    """Filename for the train ``.pkl`` dataset.

    Accepts either the legacy positional args **or** a full
    :class:`ExperimentConfig` via the ``cfg`` keyword.

    >>> train_pkl_name("dnn", "noisySignal", 5, 5)
    'train_noisySignal_5_5_data.pkl'
    >>> train_pkl_name("centroid", "noisySignal", 5, 5)
    'train_centroid_noisySignal_5_5_data.pkl'
    >>> train_pkl_name("dnn", "noisySignal", 5, 5, feature_tag="emb10")
    'train_emb10_noisySignal_5_5_data.pkl'
    >>> from naming import ExperimentConfig
    >>> c = ExperimentConfig(dataset_folder="x", prediction_source="centroid",
    ...     noise_mode="noisySignal", n_bins=5, top_k=5,
    ...     feature_type="embeddings", n_components=10,
    ...     ood_train_folder="y", use_rag=True, rag_k=10)
    >>> train_pkl_name("centroid", "noisySignal", 5, 5, cfg=c)
    'train_centroid_ood_emb10_rag10_noisySignal_5_5_data.pkl'
    """
    if cfg is not None:
        return _pkl_name("train", cfg)
    # Legacy path — build a minimal config from positional args
    _cfg = _legacy_config(
        prediction_source=prediction_source,
        noise_mode=noise_mode,
        n_bins=n_bins,
        top_k=top_k,
        feature_tag=feature_tag,
    )
    return _pkl_name("train", _cfg)


def test_pkl_name(
    prediction_source: str,
    noise_mode: str,
    n_bins: int,
    top_k: int,
    feature_tag: str = "",
    *,
    cfg: ExperimentConfig | None = None,
) -> str:
    """Filename for the test ``.pkl`` dataset.

    Accepts either the legacy positional args **or** a full
    :class:`ExperimentConfig` via the ``cfg`` keyword.

    >>> test_pkl_name("dnn", "noisySignal", 5, 5)
    'test_noisySignal_5_5_data.pkl'
    >>> test_pkl_name("centroid", "noisySignal", 5, 5)
    'test_centroid_noisySignal_5_5_data.pkl'
    >>> test_pkl_name("dnn", "noisySignal", 5, 5, feature_tag="emb10")
    'test_emb10_noisySignal_5_5_data.pkl'
    """
    if cfg is not None:
        return _pkl_name("test", cfg)
    _cfg = _legacy_config(
        prediction_source=prediction_source,
        noise_mode=noise_mode,
        n_bins=n_bins,
        top_k=top_k,
        feature_tag=feature_tag,
    )
    return _pkl_name("test", _cfg)


def eval_result_name(
    cfg: ExperimentConfig,
    prompt_type: str,
    model_name: str,
    provider: str,
) -> str:
    """Filename for evaluation result JSON.

    Pattern:
    ``{dataset}[_{tag}]_{prompt}_{model}_{noise}_{bins}_{topk}_{provider}_responses.json``

    Parameters
    ----------
    cfg : ExperimentConfig
    prompt_type : str
        e.g. ``"discret_prompts"``
    model_name : str
        e.g. ``"DeepSeek-R1-Distill-Qwen-7B"`` (already cleaned of ``unsloth/``).
    provider : str
        ``"gemini"`` | ``"openai"`` | ``"custom"``

    >>> c = ExperimentConfig(dataset_folder="-11_-15dB",
    ...     prediction_source="centroid", noise_mode="noisySignal",
    ...     n_bins=5, top_k=5)
    >>> eval_result_name(c, "discret_prompts", "gemini-2.5-flash", "gemini")
    '-11_-15dB_centroid_discret_prompts_gemini-2.5-flash_noisySignal_5_5_gemini_responses.json'

    >>> c2 = ExperimentConfig(dataset_folder="-11_-15dB",
    ...     prediction_source="centroid", noise_mode="noisySignal",
    ...     n_bins=5, top_k=5, ood_train_folder="unlabeled_10k",
    ...     feature_type="embeddings", n_components=10)
    >>> eval_result_name(c2, "discret_prompts", "o3-mini", "openai")
    '-11_-15dB_centroid_ood_emb10_discret_prompts_o3-mini_noisySignal_5_5_openai_responses.json'
    """
    tag = cfg.build_tag()
    parts = [cfg.dataset_folder]
    if tag:
        parts.append(tag)
    parts += [
        prompt_type,
        model_name,
        cfg.noise_mode,
        str(cfg.n_bins),
        str(cfg.top_k),
        f"{provider}_responses.json",
    ]
    return "_".join(parts)


# ── Legacy helper ────────────────────────────────────────────────────────────

def _legacy_config(
    prediction_source: str,
    noise_mode: str,
    n_bins: int,
    top_k: int,
    feature_tag: str = "",
) -> ExperimentConfig:
    """Build a minimal :class:`ExperimentConfig` from the old-style arguments.

    Used internally so that ``train_pkl_name`` / ``test_pkl_name`` keep
    working with existing callers while producing the same filenames.
    """
    feature_type = "stats"
    n_components = 0
    if feature_tag and feature_tag.startswith("emb"):
        feature_type = "embeddings"
        try:
            n_components = int(feature_tag[3:])
        except ValueError:
            pass

    return ExperimentConfig(
        dataset_folder="",          # not used in pkl names
        prediction_source=prediction_source,
        noise_mode=noise_mode,
        n_bins=n_bins,
        top_k=top_k,
        feature_type=feature_type,
        n_components=n_components,
    )
