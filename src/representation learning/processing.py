# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure

from constants import FFT_SIZE


def visualize_signal(signal: np.ndarray, Fs: float = 1) -> matplotlib.figure.Figure:
    """Plots the magnitude of the complex signal over time.

    Args:
        signal: The input complex signal as a NumPy array.
        Fs: The sampling rate in Hz. Defaults to SAMPLING_RATE.

    Returns:
        A Matplotlib Figure object containing the plot. The figure is closed
        to prevent immediate display.
    """
    magnitude = np.abs(signal)
    time = np.arange(signal.size) / Fs

    fig, ax = plt.subplots()
    ax.plot(time, magnitude)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Magnitude")
    ax.set_title("Signal Magnitude vs. Time")
    ax.grid(True)
    plt.tight_layout()
    plt.close(fig) # Close the figure to avoid displaying it
    return fig

def get_power_spectrogram_db(stft_matrix: np.ndarray,
                                window_size: int = FFT_SIZE, overlap: int = 128, step_size: int = None, eps: float = 1e-12) -> np.ndarray:
    """
    Compute power spectrogram in dB from STFT matrix.
    
    Parameters:
    -----------
    stft_matrix : np.ndarray
        Complex STFT matrix with shape (window_size, num_time_bins)
    sampling_rate : float, default=SAMPLING_RATE
        Sampling rate in Hz
    center_freq : float, default=CENTER_FREQ
        Center frequency for frequency axis
    window_size : int, default=FFT_SIZE
        FFT window size
    overlap : int, default=128
        Number of overlapping samples between windows
    step_size : Optional[int], optional
        Step size between windows (for time axis calculation)
    eps : float, default=1e-12
        Small value to avoid log(0)
        
    Returns:
    --------
    spectrum_db : np.ndarray
        Power spectrogram in dB scale
    """
    if step_size is None:
        step_size = window_size - overlap  # Default overlap of 128
    
    # Calculate power spectral density (magnitude squared)
    spectrum: np.ndarray = np.abs(stft_matrix) ** 2
    
    # Convert to dB for plotting
    spectrum_db: np.ndarray = 10 * np.log10(spectrum + eps)

    return spectrum_db

def get_color_img(spectrum_db: np.ndarray, colormap: str = 'viridis') -> np.ndarray:
    """
    Convert power spectrogram in dB to color image.
    
    Parameters:
    -----------
    spectrum_db : np.ndarray
        Power spectrogram in dB scale
    colormap : str, default='viridis'
        Colormap to use for visualization
        
    Returns:
    --------
    color_img : np.ndarray
        Color image representation of the spectrogram
    """
    # Normalize the dB values to [0, 1] range for colormap
    norm_spectrum: np.ndarray = (spectrum_db - np.min(spectrum_db)) / (np.max(spectrum_db) - np.min(spectrum_db))
    
    # Apply colormap RGBA values ()
    rgba_img: np.ndarray = plt.get_cmap(colormap)(norm_spectrum)

    # Convert to RGB by removing the alpha channel
    alpha = rgba_img[..., 3:]       # shape (H, W, 1)
    rgb_channels = rgba_img[..., :3]     # shape (H, W, 3), floats in [0,1]
    bg_color = np.ones_like(rgb_channels)    # white background (1,1,1)

    # composite
    color_img = rgb_channels * alpha + bg_color * (1 - alpha)

    return color_img