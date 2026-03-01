"""Temporal denoising and smoothing for intensity data.

Operates on (num_frames, 88) float32 arrays produced by wvlt.
"""

import numpy as np


def denoise(
    data: np.ndarray,
    reach: int = 8,
    cutoff: float = 0.4,
) -> np.ndarray:
    """Remove transient noise by requiring persistence above threshold.

    For each frame, checks whether note intensities persist above a
    statistical threshold over a window of ``reach`` frames in both
    the forward and backward directions. Notes that fail either check
    are zeroed out.

    Args:
        data: Array of shape (num_frames, num_keys) with intensity values.
        reach: Number of frames to look ahead/behind for persistence.
        cutoff: Number of standard deviations above the mean a note's
            windowed average must exceed to be kept.

    Returns:
        Denoised copy of the input array.
    """
    num_frames = data.shape[0]
    forward_mask = np.zeros(data.shape, dtype=bool)
    backward_mask = np.zeros(data.shape, dtype=bool)

    # Forward sweep: check if note persists in [idx, idx + reach).
    for idx in range(num_frames - reach):
        window_mean = data[idx:idx + reach].mean(axis=0)
        threshold = window_mean.mean() + cutoff * window_mean.std()
        forward_mask[idx] = window_mean > threshold

    # Backward sweep: check if note persists in (idx - reach, idx].
    for idx in range(reach, num_frames):
        window_mean = data[idx - reach:idx].mean(axis=0)
        threshold = window_mean.mean() + cutoff * window_mean.std()
        backward_mask[idx] = window_mean > threshold

    return data * (forward_mask & backward_mask)


def smooth(
    data: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    """Apply temporal moving-average smoothing along the frame axis.

    Args:
        data: Array of shape (num_frames, num_keys) with intensity values.
        kernel_size: Width of the averaging window (must be odd).

    Returns:
        Smoothed copy of the input array.

    Raises:
        ValueError: If kernel_size is not a positive odd integer.
    """
    if kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError(
            f"kernel_size must be a positive odd integer, got {kernel_size}"
        )

    half = kernel_size // 2
    num_frames, num_keys = data.shape
    result = np.zeros_like(data)

    # Compute cumulative sum for efficient windowed averages.
    cumsum = np.zeros((num_frames + 1, num_keys), dtype=data.dtype)
    cumsum[1:] = np.cumsum(data, axis=0)

    for idx in range(num_frames):
        lo = max(0, idx - half)
        hi = min(num_frames, idx + half + 1)
        result[idx] = (cumsum[hi] - cumsum[lo]) / (hi - lo)

    return result
