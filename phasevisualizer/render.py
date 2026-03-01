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

    All bar pixel data is pre-computed with vectorized numpy operations
    before the render loop begins. The loop body is a single array
    assignment and a zero-copy pipe write, eliminating per-frame
    allocation and serialization overhead.

    Args:
        intensities: Array of shape (num_frames, 88) with float intensity
            values, expected to be pre-normalized to [0, 1].
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

    # Pre-compute bar pixel rows for all frames at once.
    # bar_rows_all: (num_frames, total_bar_width, 3) uint8
    colors_float = colors.astype(np.float32)
    normalized_all = np.clip(intensities, 0.0, 1.0)
    bar_colors_all = (
        colors_float * normalized_all[:, :, np.newaxis]
    ).astype(np.uint8)
    bar_rows_all = np.repeat(bar_colors_all, bar_width, axis=1)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-i", input_wav,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "-loglevel", "warning",
        output_mp4,
    ]

    # Single reusable frame buffer; black margins stay zero throughout.
    frame_buf = np.zeros((height, width, 3), dtype=np.uint8)

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    try:
        for i in tqdm(range(num_frames), desc="Rendering", unit="frame"):
            frame_buf[:, margin_left:margin_left + total_bar_width] = bar_rows_all[i]
            proc.stdin.write(memoryview(frame_buf))
    finally:
        proc.stdin.close()
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")
