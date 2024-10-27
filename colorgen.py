from pydantic import BaseModel
from dataclasses import dataclass
import numpy as np
from typing import Dict

class Palette(BaseModel):
    """
    Default palette is Scriabin's synesthesia palette
    """
    C: str = "rgba(255,0,0,255)"
    G: str = "rgba(255,127,0,255)"
    D: str = "rgba(255,255,0,255)"
    A: str = "rgba(51,204,51,255)"
    E: str = "rgba(195,242,255,255)"
    B: str = "rgba(142,201,255,255)"
    Gb: str = "rgba(127,139,253,255)"
    Db: str = "rgba(144,0,255,255)"
    Ab: str = "rgba(187,117,252,255)"
    Eb: str = "rgba(183,70,139,255)"
    Bb: str = "rgba(183,70,139,255)"
    F: str = "rgba(171,0,52,255)"

def freqs_to_notes(base_freq_a: float, min_freq: int = 15, max_freq: int = 18000) -> dict:
    """Calculates the frequencies of notes in a 12 tone equal temperment scale with a relative tuning of base_freq_a

    Args:
        base_freq_a: The frequency of the note A (e.g. 440Hz in modern compositions).
        min_freq: lowest frequency to which a note should be assigned (Hz)
        max_freq: highest frequency to which a note will be assigned (Hz)
    Returns:
        A dictionary mapping frequencies to the corresponding 12 tone equal temperment note name
    """
    notes = ["A", "Bb", "B", "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab"]
    cur_freq = base_freq_a
    cur_note_idx = 0
    # Search down to the lowest note in our range of notes
    while cur_freq > min_freq:
        cur_freq = cur_freq * (2 ** (-1.0/12)) # move down a semitone
        cur_note_idx = (cur_note_idx - 1) % len(notes) # move index down one note in scale (scale repeats cyclically)
    
    # Get the name and frequency of the lowest note within our defined range
    next_freq_up = lambda freq: freq * (2 ** (1.0/12))
    next_note_idx_up = lambda idx: (idx + 1) % len(notes)
    cur_freq = next_freq_up(cur_freq)
    cur_note_idx = next_note_idx_up(cur_note_idx)
    # Get the name of all the notes in our frequency range
    freq_to_note = {}
    while cur_freq < max_freq:
        freq_to_note[cur_freq] = notes[cur_note_idx]
        cur_note_idx = next_note_idx_up(cur_note_idx)
        cur_freq = next_freq_up(cur_freq)

    return freq_to_note


def find_closest_note(frequency: float, freq_to_note: dict[float, str]):
    """Finds the closest note to a given frequency.

    Args:
        frequency: The frequency to match.
        note_freqs: A NumPy array containing the frequencies of all notes.

    Returns:
        The index of the closest note in the `note_freqs` array.
    """
    note_freqs = list(freq_to_note.keys())
    # Find the index of the first frequency greater than or equal to the given frequency
    index = np.searchsorted(note_freqs, frequency, side="left")
    if index == 0:
        return 0
    prev_diff = abs(note_freqs[index - 1] - frequency)
    next_diff = abs(note_freqs[index] - frequency)
    # index of the note with the smallest difference
    closest_index = index - 1 if prev_diff < next_diff else index
    return freq_to_note[note_freqs[closest_index]]


def rgba_to_ndarray(rgba_str: str) -> np.ndarray:
    """Converts a string of the form "rgba(r,g,b,a)" to a numpy ndarray.
    Args:
        rgba_str: The string containing the RGBA values.

    Returns:
        umpy ndarray of type int u8 with the RGBA values [r, g, b, a]
    """
    rgba_str = rgba_str.replace("rgba(", "").replace(")", "")
    # Split the string into individual values
    rgba_values = rgba_str.split(",")
    # Convert the values to integers and create a numpy array
    rgba_array = np.array(rgba_values, dtype=np.uint8)
    return rgba_array

class ColorPallateSampler:
    def __init__(self, base_freq: float, palette: Palette, color_snapping: bool = True) -> None:
        self.base_freq = base_freq
        self.palette = palette
        self.color_snapping = color_snapping
        self.freq_to_notes = freqs_to_notes(base_freq)

    def color_for_freq(self, freq: float):
        if self.color_snapping:
            nearest_note = find_closest_note(freq, self.freq_to_notes)
            nearest_note_color: str = getattr(self.palette, nearest_note)
            return rgba_to_ndarray(nearest_note_color)
        else:
            raise NotImplementedError("color_for_freq not yet defined when color_snapping not enabled")