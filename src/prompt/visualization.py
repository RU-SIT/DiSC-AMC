import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from glob import glob  # Added missing import
import colorsys  # For generating distinct colors
from sklearn.metrics import confusion_matrix
import pandas as pd

def generate_distinct_colors(n):
    """
    Generate n visually distinct colors using HSV color space.
    
    Parameters:
    -----------
    n : int
        Number of distinct colors to generate
    
    Returns:
    --------
    list
        List of distinct colors in hex format
    """
    colors = []
    for i in range(n):
        # Use golden ratio to create well-distributed hues
        h = i / n
        s = 0.7 + 0.3 * (i % 2)  # Alternate between 0.7 and 1.0 saturation
        v = 0.8 + 0.2 * ((i//2) % 2)  # Alternate between 0.8 and 1.0 value
        
        # Convert HSV to RGB
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        
        # Convert RGB to hex
        hex_color = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
        colors.append(hex_color)
    
    return colors


def get_marker_symbols(n):
    """
    Generate a list of distinct marker symbols for visualization.
    
    Parameters:
    -----------
    n : int
        Number of distinct marker symbols needed
    
    Returns:
    --------
    list
        List of marker symbol names
    """
    # List of available marker symbols in Plotly
    marker_symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 
                      'triangle-down', 'triangle-left', 'triangle-right', 'pentagon', 
                      'hexagon', 'star', 'hexagram', 'hourglass', 'bowtie', 'diamond-cross', 
                      'diamond-x', 'square-cross', 'square-x', 'circle-cross', 'circle-x']
    
    # If we need more symbols than available, we'll cycle through them
    result = []
    for i in range(n):
        result.append(marker_symbols[i % len(marker_symbols)])
    
    return result


def get_3d_marker_symbols(n):
    """
    Generate a list of distinct marker symbols for 3D visualization.
    
    Parameters:
    -----------
    n : int
        Number of distinct marker symbols needed
    
    Returns:
    --------
    list
        List of valid 3D marker symbol names
    """
    # List of available marker symbols in Plotly for 3D plots
    # These are the only symbols supported by Scatter3d
    marker_symbols = ['circle', 'circle-open', 'cross', 'diamond', 
                      'diamond-open', 'square', 'square-open']
    
    # If we need more symbols than available, we'll cycle through them
    result = []
    for i in range(n):
        result.append(marker_symbols[i % len(marker_symbols)])
    
    return result


def visualize_tsne(data, labels=None, label_name="Category", 
                   second_labels=None, second_label_name="Value",
                   pca_components=50, perplexity=30, n_iter=1000, random_state=42,
                   create_3d=True, fig_height=800, fig_width=1600):
    """
    Create t-SNE visualizations of high-dimensional data.
    
    Parameters:
    -----------
    data : array-like, shape (n_samples, n_features)
        The data to visualize with t-SNE
    labels : array-like, shape (n_samples,), optional
        Category labels for data points (determines color in first subplot)
    label_name : str, optional
        Name for the first category of labels
    second_labels : array-like, shape (n_samples,), optional
        Secondary labels for data points (determines color in second subplot)
    second_label_name : str, optional
        Name for the second category of labels
    pca_components : int, optional
        Number of PCA components to use before t-SNE
    perplexity : float, optional
        Perplexity parameter for t-SNE
    n_iter : int, optional
        Number of iterations for t-SNE
    random_state : int, optional
        Random seed for reproducibility
    create_3d : bool, optional
        Whether to create a 3D t-SNE visualization
    fig_height : int, optional
        Height of the figure in pixels
    fig_width : int, optional
        Width of the figure in pixels
        
    Returns:
    --------
    tuple
        (2D visualization figure, 3D visualization figure if create_3d=True)
    """
    # Convert data to numpy array if not already
    data_array = np.array(data)
    print(f"Data array shape: {data_array.shape}")
    
    # Apply PCA to reduce dimensionality if the data is high-dimensional
    if data_array.shape[1] > pca_components:
        print("Applying PCA...")
        pca = PCA(n_components=pca_components)
        data_reduced = pca.fit_transform(data_array)
        print(f"Explained variance with {pca_components} components: {sum(pca.explained_variance_ratio_):.3f}")
    else:
        data_reduced = data_array
    
    # Apply t-SNE
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity, max_iter=n_iter)
    tsne_results = tsne.fit_transform(data_reduced)
    
    # Create subplot for 2D visualization
    if second_labels is not None:
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(f"t-SNE by {label_name}", f"t-SNE by {second_label_name}"),
                            specs=[[{"type": "scatter"}, {"type": "scatter"}]])
    else:
        fig = make_subplots(rows=1, cols=1,
                            subplot_titles=(f"t-SNE by {label_name}",),
                            specs=[[{"type": "scatter"}]])
    
    # Plot 1: Color by primary labels
    if labels is not None:
        unique_labels = sorted(list(set(labels)))
        # Generate distinct colors for the unique labels
        distinct_colors = generate_distinct_colors(len(unique_labels))
        # Generate distinct marker symbols
        marker_symbols = get_marker_symbols(len(unique_labels))
        
        for idx, label in enumerate(unique_labels):
            indices = [i for i, l in enumerate(labels) if l == label]
            fig.add_trace(
                go.Scatter(
                    x=tsne_results[indices, 0],
                    y=tsne_results[indices, 1],
                    mode='markers',
                    marker=dict(
                        size=6, 
                        opacity=0.6, 
                        color=distinct_colors[idx],
                        symbol=marker_symbols[idx]
                    ),
                    name=str(label),
                    showlegend=True
                ),
                row=1, col=1
            )
    else:
        # If no labels provided, plot all points with the same color
        fig.add_trace(
            go.Scatter(
                x=tsne_results[:, 0],
                y=tsne_results[:, 1],
                mode='markers',
                marker=dict(size=6, opacity=0.7),
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Plot 2: Color by secondary labels (if provided)
    if second_labels is not None:
        try:
            # Try to convert to float for a color gradient
            numeric_labels = [float(label) for label in second_labels]
            fig.add_trace(
                go.Scatter(
                    x=tsne_results[:, 0],
                    y=tsne_results[:, 1],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=numeric_labels,
                        colorscale='Viridis',
                        colorbar=dict(title=second_label_name),
                        opacity=0.7,
                        showscale=True
                    ),
                    showlegend=False
                ),
                row=1, col=2
            )
        except (ValueError, TypeError):
            # If conversion fails, treat as categorical
            unique_secondary_labels = sorted(list(set(second_labels)))
            # Generate distinct colors for secondary labels
            secondary_distinct_colors = generate_distinct_colors(len(unique_secondary_labels))
            # Generate distinct marker symbols for secondary labels
            secondary_marker_symbols = get_marker_symbols(len(unique_secondary_labels))
            
            for idx, label in enumerate(unique_secondary_labels):
                indices = [i for i, l in enumerate(second_labels) if l == label]
                fig.add_trace(
                    go.Scatter(
                        x=tsne_results[indices, 0],
                        y=tsne_results[indices, 1],
                        mode='markers',
                        marker=dict(
                            size=6, 
                            opacity=0.6, 
                            color=secondary_distinct_colors[idx],
                            symbol=secondary_marker_symbols[idx]
                        ),
                        name=str(label),
                        showlegend=True
                    ),
                    row=1, col=2
                )
    
    # Update layout
    fig.update_layout(
        title_text="t-SNE Visualization",
        height=fig_height, 
        width=fig_width,
        legend=dict(
            yanchor="top",
            y=1,
            xanchor="right",
            x=0.99
        )
    )
    
    fig.update_xaxes(title_text="t-SNE component 1", row=1, col=1)
    fig.update_yaxes(title_text="t-SNE component 2", row=1, col=1)
    
    if second_labels is not None:
        fig.update_xaxes(title_text="t-SNE component 1", row=1, col=2)
        fig.update_yaxes(title_text="t-SNE component 2", row=1, col=2)
    
    # 3D t-SNE visualization (if requested)
    fig_3d = None
    if create_3d:
        # Apply 3D t-SNE
        tsne_3d = TSNE(n_components=3, random_state=random_state, perplexity=perplexity, max_iter=n_iter)
        tsne_results_3d = tsne_3d.fit_transform(data_reduced)
        
        # Create 3D plot with Plotly
        fig_3d = go.Figure()
        
        if labels is not None:
            # Get valid 3D marker symbols
            marker_symbols_3d = get_3d_marker_symbols(len(unique_labels))
            
            for idx, label in enumerate(unique_labels):
                indices = [i for i, l in enumerate(labels) if l == label]
                fig_3d.add_trace(
                    go.Scatter3d(
                        x=tsne_results_3d[indices, 0],
                        y=tsne_results_3d[indices, 1],
                        z=tsne_results_3d[indices, 2],
                        mode='markers',
                        marker=dict(
                            size=6, 
                            opacity=0.6, 
                            color=distinct_colors[idx],
                            symbol=marker_symbols_3d[idx]
                        ),
                        name=str(label)
                    )
                )
        else:
            fig_3d.add_trace(
                go.Scatter3d(
                    x=tsne_results_3d[:, 0],
                    y=tsne_results_3d[:, 1],
                    z=tsne_results_3d[:, 2],
                    mode='markers',
                    marker=dict(size=6, opacity=0.7),
                    showlegend=False
                )
            )
        
        # Update 3D layout
        fig_3d.update_layout(
            title=f'3D t-SNE Visualization by {label_name}',
            scene=dict(
                xaxis_title='t-SNE component 1',
                yaxis_title='t-SNE component 2',
                zaxis_title='t-SNE component 3'
            ),
            width=fig_width * 0.625,  # Maintain aspect ratio
            height=fig_height
        )
    
    return (fig, fig_3d) if create_3d else fig


def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False, title=None, 
                          cmap='Blues', fig_height=800, fig_width=800):
    """
    Create an interactive confusion matrix visualization using Plotly.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    class_names : list, optional
        List of class names (if None, class names will be derived from the labels)
    normalize : bool, optional
        Whether to normalize the confusion matrix values (default: False)
    title : str, optional
        Title for the confusion matrix plot
    cmap : str, optional
        Colorscale to use for the heatmap (default: 'Blues')
    fig_height : int, optional
        Height of the figure in pixels
    fig_width : int, optional
        Width of the figure in pixels
    
    Returns:
    --------
    plotly.graph_objects.Figure
        The confusion matrix figure
    """
    if class_names is None:
        classes = sorted(list(set(list(y_true) + list(y_pred))))
    else:
        classes = class_names
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Normalize if required
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title_suffix = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title_suffix = 'Confusion Matrix'
    
    if title is None:
        title = title_suffix
    
    # Create dataframe for better visualization
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    
    # Create annotation text
    annotations = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if normalize:
                text = f"{cm[i, j]:.2f}"
            else:
                text = f"{cm[i, j]}"
                
            # Determine text color based on background darkness
            text_color = 'black' if cm[i, j] < cm.max() / 2 else 'white'
            
            annotations.append(
                dict(
                    x=j, 
                    y=i,
                    text=text,
                    showarrow=False,
                    font=dict(color=text_color)
                )
            )
    
    # Create heatmap figure
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=classes,
        y=classes,
        colorscale=cmap,
        showscale=True,
        colorbar=dict(title="Count" if not normalize else "Ratio")
    ))
    
    fig.update_layout(
        title=title,
        xaxis=dict(title="Predicted label", tickangle=-45),
        yaxis=dict(title="True label", autorange="reversed"),
        width=fig_width,
        height=fig_height,
        annotations=annotations
    )
    
    return fig


def save_figure_as_html(fig, filename):
    """
    Save a Plotly figure as an HTML file.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        The Plotly figure to save
    filename : str
        The filename to save the figure to (should end with .html)
    """
    if not filename.endswith('.html'):
        filename += '.html'
    
    fig.write_html(filename)
    print(f"Figure saved to {filename}")



# Example usage with embedding data
if __name__ == "__main__":
    
    # # Visualizing deepseek 32b qwen embeddings
    # from data_processing import load_npy_file
    # embedding_paths = glob('../../data/RadioML/*/train/*/*_embedding.npy')
    # embedding_vectors = [load_npy_file(path) for path in embedding_paths]

    # # Extract labels and SNR values from file paths
    # labels = []
    # snr_values = []

    # print(f"Processing {len(embedding_paths)} embedding files...")

    # for path in embedding_paths:
    #     parts = path.split(os.sep)
    #     modulation_type = parts[-2]  # Get the modulation type from the path
    #     snr = parts[-4].replace('snr_', '').replace('db', '')  # Extract SNR value
    #     labels.append(modulation_type)
    #     snr_values.append(snr)

    # ## Visualize the embeddings
    # fig_2d, fig_3d = visualize_tsne(
    #     data=embedding_vectors, 
    #     labels=labels, 
    #     label_name="Modulation Type", 
    #     second_labels=snr_values, 
    #     second_label_name="SNR (dB)"
    # )

    # fig_2d.show()
    # fig_3d.show()

    # Visualizing Summary Statistics as Features
    from data_processing import load_processed_data

    # Load the processed data
    data = load_processed_data('../../data/RadioML/train_data.pkl')

    # Plot t-SNE
    fig_2d, fig_3d = visualize_tsne(data['stats'], labels=data['labels'], label_name="Modulation Typ", 
                   second_labels=data['snrs'], second_label_name="SNR (dB)",
                   pca_components=39, perplexity=30, n_iter=1000, random_state=42,
                   create_3d=True, fig_height=800, fig_width=1600)

    save_figure_as_html(fig_2d, 'exp/stats_embeddings/2d_tsne_summary_statistics.html')
    save_figure_as_html(fig_3d, 'exp/stats_embeddings/3d_tsne_summary_statistics.html')

    # Plot t-SNE
    fig_2d, fig_3d = visualize_tsne(data['stats_discretized'], labels=data['labels'], label_name="Modulation Typ", 
                   second_labels=data['snrs'], second_label_name="SNR (dB)",
                   pca_components=39, perplexity=30, n_iter=1000, random_state=42,
                   create_3d=True, fig_height=800, fig_width=1600)

    save_figure_as_html(fig_2d, 'exp/stats_embeddings/2d_tsne_summary_statistics_discretized.html')
    save_figure_as_html(fig_3d, 'exp/stats_embeddings/3d_tsne_summary_statistics_discretized.html')