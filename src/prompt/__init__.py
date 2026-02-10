"""
Data processing module for signal feature extraction, prompt generation,
and DINO-based embedding pipeline.
"""

from .data_processing import (
    load_npy_file,
    save_to_json,
    load_from_json,
    get_features,
    dict_to_np,
    discretize_features,
    convert_signal_to_complex,
    reduce_example_dict,
    get_processed_data,
    save_processed_data,
    load_processed_data,
)

from ..spectrogram.embedding_pipeline import (
    extract_label,
    load_encoder,
    extract_embeddings,
    process_split,
    save_results,
    load_results,
    run_pipeline,
    CLASSES,
    FEATURE_TYPES,
    FittedTransformers,
)

from .visualization import (
    visualize_tsne,
    save_figure_as_html,
    generate_distinct_colors,
    get_marker_symbols,
    get_3d_marker_symbols,
    plot_confusion_matrix,
)

__all__ = [
    # data_processing
    'load_npy_file', 'save_to_json', 'load_from_json', 'get_features',
    'dict_to_np', 'discretize_features', 'convert_signal_to_complex',
    'reduce_example_dict', 'get_processed_data', 'save_processed_data',
    'load_processed_data',
    # embedding_pipeline
    'extract_label', 'load_encoder', 'extract_embeddings', 'process_split',
    'save_results', 'load_results', 'run_pipeline', 'CLASSES', 'FEATURE_TYPES',
    'FittedTransformers',
    # visualization
    'visualize_tsne', 'save_figure_as_html', 'generate_distinct_colors',
    'get_marker_symbols', 'get_3d_marker_symbols', 'plot_confusion_matrix',
]
