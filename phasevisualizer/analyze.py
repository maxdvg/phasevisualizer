"""Subprocess wrapper for the wav_analyze C utility."""

import shutil
import subprocess


def run(
    input_wav: str,
    output_wvlt: str,
    *,
    fps: float = 60.0,
    omega: float = 6.0,
    sigma: float = 4.0,
    wav_analyze_bin: str = "wav_analyze",
) -> None:
    """Run wav_analyze to produce a .wvlt file from a WAV input.

    Args:
        input_wav: Path to input WAV file.
        output_wvlt: Path for output .wvlt file.
        fps: Analysis frames per second.
        omega: Morlet wavelet omega0 parameter.
        sigma: Window truncation in standard deviations.
        wav_analyze_bin: Path to the wav_analyze binary.

    Raises:
        FileNotFoundError: If wav_analyze binary is not found.
        subprocess.CalledProcessError: If wav_analyze exits with non-zero status.
    """
    if not shutil.which(wav_analyze_bin):
        raise FileNotFoundError(
            f"wav_analyze binary not found: {wav_analyze_bin!r}. "
            "Ensure it is built and on your PATH."
        )

    cmd = [
        wav_analyze_bin,
        "-n", str(fps),
        "-w", str(omega),
        "-m", str(sigma),
        input_wav,
        output_wvlt,
    ]
    subprocess.run(cmd, check=True)
