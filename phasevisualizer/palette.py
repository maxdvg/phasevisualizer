"""Scriabin synesthesia color palette for 88 piano keys.

Maps each of the 12 chromatic note classes to an RGB color based on
Alexander Scriabin's sound-to-color associations. The chromatic index
follows wvlt's ordering: 0=A, 1=A#, 2=B, 3=C, ..., 11=G#.
"""

import numpy as np

# Scriabin's synesthesia palette indexed by chromatic class.
# Chromatic index = (piano_key - 1) % 12, matching wvlt._NOTE_NAMES.
SCRIABIN_PALETTE: np.ndarray = np.array([
    [51, 204, 51],      # 0  A
    [169, 103, 124],    # 1  A#
    [142, 201, 255],    # 2  B
    [255, 0, 0],        # 3  C
    [144, 0, 255],      # 4  C#
    [255, 255, 0],      # 5  D
    [183, 70, 139],     # 6  D#
    [195, 242, 255],    # 7  E
    [171, 0, 52],       # 8  F
    [127, 139, 253],    # 9  F#
    [255, 127, 0],      # 10 G
    [187, 117, 252],    # 11 G#
], dtype=np.uint8)


def key_colors() -> np.ndarray:
    """Return an (88, 3) uint8 array mapping each piano key to its RGB color.

    Returns:
        Array of shape (88, 3) with dtype uint8, where row i corresponds
        to piano key i+1 (0-indexed array, 1-indexed keys).
    """
    indices = np.arange(88) % 12
    return SCRIABIN_PALETTE[indices]
