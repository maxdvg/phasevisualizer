"""Frequency-domain equalization for CWT intensity data.

Compensates for scale-dependent energy differences across piano keys
and suppresses spectral bleed between adjacent keys.
"""

import numpy as np


def normalize_keys(data: np.ndarray) -> np.ndarray:
    """Normalize each key's intensity independently to [0, 1].

    CWT intensities fall off with frequency because lower-frequency
    wavelets have wider convolution windows that accumulate more energy.
    Per-key normalization compensates for this by scaling each key's
    time series to its own maximum.

    Args:
        data: Array of shape (num_frames, num_keys) with intensity values.

    Returns:
        Normalized copy where each key spans [0, 1] independently.
    """
    key_max = data.max(axis=0)  # (num_keys,)
    key_max[key_max == 0] = 1.0
    return data / key_max[np.newaxis, :]


def suppress_bleed(
    data: np.ndarray,
    threshold: float = 0.3,
) -> np.ndarray:
    """Suppress spectral bleed by keeping only local maxima along the key axis.

    The Morlet wavelet's frequency bandwidth causes adjacent keys to
    activate even when only one note is played. This function identifies
    spectral peaks (local maxima) and zeros out non-peak keys.

    Args:
        data: Array of shape (num_frames, num_keys) with intensity values.
        threshold: Relative threshold for peak significance. A local
            maximum is only kept if it exceeds this fraction of the
            frame's global peak intensity.

    Returns:
        Filtered copy with non-peak keys zeroed out.
    """
    num_frames, num_keys = data.shape
    result = np.zeros_like(data)

    for frame_idx in range(num_frames):
        row = data[frame_idx]
        frame_max = row.max()
        if frame_max == 0:
            continue

        min_intensity = threshold * frame_max

        # Find local maxima: intensity[k] >= both neighbors.
        for k in range(num_keys):
            left = row[k - 1] if k > 0 else 0.0
            right = row[k + 1] if k < num_keys - 1 else 0.0
            if row[k] >= left and row[k] >= right and row[k] >= min_intensity:
                result[frame_idx, k] = row[k]

    return result
