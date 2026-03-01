"""Video rendering via vectorized numpy frames piped to ffmpeg."""

import subprocess

import numpy as np
from tqdm import tqdm

from phasevisualizer.palette import key_colors


def render_video(
    intensities: np.ndarray,
    input_wav: str,
    output_mp4: str,
    *,
    fps: float,
    width: int = 1920,
    height: int = 1080,
) -> None:
    """Render a histogram visualization video from intensity data.

    Each frame contains 88 vertical bars (one per piano key) whose
    brightness corresponds to the key's intensity. Colors follow
    Scriabin's synesthesia palette. Audio from the original WAV is
    muxed in a single ffmpeg pass.

    Args:
        intensities: Array of shape (num_frames, 88) with float intensity
            values (non-negative).
        input_wav: Path to the original WAV file for audio muxing.
        output_mp4: Path for the output MP4 video.
        fps: Video frame rate (should match analysis fps).
        width: Video width in pixels.
        height: Video height in pixels.
    """
    num_frames, num_keys = intensities.shape
    colors = key_colors()  # (88, 3) uint8

    # Bar layout: center 88 bars within the frame width.
    bar_width = width // num_keys
    total_bar_width = bar_width * num_keys
    margin_left = (width - total_bar_width) // 2

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-i", input_wav,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "-loglevel", "warning",
        output_mp4,
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    try:
        for frame_idx in tqdm(range(num_frames), desc="Rendering", unit="frame"):
            frame = _build_frame(
                intensities[frame_idx], colors, width, height,
                bar_width, margin_left, total_bar_width,
            )
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")


def _build_frame(
    frame_intensities: np.ndarray,
    colors: np.ndarray,
    width: int,
    height: int,
    bar_width: int,
    margin_left: int,
    total_bar_width: int,
) -> np.ndarray:
    """Construct a single RGB frame as a numpy array.

    Args:
        frame_intensities: Array of shape (88,) with intensity values,
            expected to be pre-normalized to [0, 1] by equalization.
        colors: Array of shape (88, 3) uint8 with per-key RGB colors.
        width: Frame width in pixels.
        height: Frame height in pixels.
        bar_width: Width of each bar in pixels.
        margin_left: Left margin offset in pixels.
        total_bar_width: Total width used by all bars.

    Returns:
        Array of shape (height, width, 3) with dtype uint8.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Clip to [0, 1] (data should already be normalized by equalization).
    normalized = np.clip(frame_intensities, 0.0, 1.0)

    # Scale colors by intensity: (88, 3) float -> uint8.
    bar_colors = (colors.astype(np.float32) * normalized[:, np.newaxis]).astype(
        np.uint8
    )

    # Expand each bar's color across bar_width pixels: (88, 3) -> (total_bar_width, 3).
    bar_row = np.repeat(bar_colors, bar_width, axis=0)

    # Fill every row of the frame with the same bar pattern.
    frame[:, margin_left:margin_left + total_bar_width] = bar_row

    return frame
