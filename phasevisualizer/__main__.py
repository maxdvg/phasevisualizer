"""CLI entry point for phasevisualizer.

Usage:
    python -m phasevisualizer input.wav output.mp4 [OPTIONS]
"""

import argparse
import os
import sys
import tempfile

import numpy as np
from wvlt import WvltFile

from phasevisualizer import analyze
from phasevisualizer import denoise as denoise_mod
from phasevisualizer import equalize
from phasevisualizer import render


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="phasevisualizer",
        description=(
            "Generate Scriabin-palette music visualization videos "
            "using Morlet CWT analysis."
        ),
    )
    parser.add_argument("input_wav", help="Path to input WAV file.")
    parser.add_argument("output_mp4", help="Path for output MP4 video.")

    # Analysis options.
    analysis = parser.add_argument_group("analysis")
    analysis.add_argument(
        "--fps", type=float, default=60.0, help="Frames per second (default: 60)."
    )
    analysis.add_argument(
        "--omega", type=float, default=6.0,
        help="Morlet omega0 parameter (default: 6.0).",
    )
    analysis.add_argument(
        "--sigma", type=float, default=4.0,
        help="Window truncation in sigma (default: 4.0).",
    )
    analysis.add_argument(
        "--wav-analyze", default="wav_analyze", dest="wav_analyze_bin",
        help="Path to wav_analyze binary (default: searches PATH).",
    )

    # Video options.
    video = parser.add_argument_group("video")
    video.add_argument(
        "--width", type=int, default=1920, help="Video width (default: 1920)."
    )
    video.add_argument(
        "--height", type=int, default=1080, help="Video height (default: 1080)."
    )

    # Processing options.
    processing = parser.add_argument_group("processing")
    processing.add_argument(
        "--denoise-reach", type=int, default=8,
        help="Denoising window size in frames (default: 8).",
    )
    processing.add_argument(
        "--denoise-cutoff", type=float, default=0.4,
        help="Denoising std-dev cutoff (default: 0.4).",
    )
    processing.add_argument(
        "--smooth-kernel", type=int, default=5,
        help="Smoothing kernel size, must be odd (default: 5).",
    )
    processing.add_argument(
        "--no-denoise", action="store_true", help="Skip denoising step."
    )
    processing.add_argument(
        "--no-smooth", action="store_true", help="Skip smoothing step."
    )
    processing.add_argument(
        "--bleed-threshold", type=float, default=0.3,
        help="Relative threshold for spectral bleed suppression (default: 0.3).",
    )
    processing.add_argument(
        "--no-equalize", action="store_true",
        help="Skip frequency equalization and bleed suppression.",
    )

    # Debug options.
    debug = parser.add_argument_group("debug")
    debug.add_argument(
        "--keep-wvlt", action="store_true",
        help="Do not delete intermediate .wvlt file.",
    )
    debug.add_argument(
        "--wvlt-file", default=None,
        help="Use existing .wvlt file instead of running wav_analyze.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the phasevisualizer pipeline."""
    args = _parse_args()

    # Validate input file.
    if not args.wvlt_file and not os.path.isfile(args.input_wav):
        print(f"Error: input file not found: {args.input_wav}", file=sys.stderr)
        sys.exit(1)

    # Step 1: Produce .wvlt file.
    wvlt_path = args.wvlt_file
    temp_wvlt = None

    if wvlt_path is None:
        temp_wvlt = tempfile.NamedTemporaryFile(
            suffix=".wvlt", delete=False
        )
        wvlt_path = temp_wvlt.name
        temp_wvlt.close()

        print(f"Analyzing {args.input_wav}...")
        analyze.run(
            args.input_wav,
            wvlt_path,
            fps=args.fps,
            omega=args.omega,
            sigma=args.sigma,
            wav_analyze_bin=args.wav_analyze_bin,
        )

    try:
        # Step 2: Load intensity data.
        print("Loading analysis data...")
        with WvltFile(wvlt_path) as wf:
            intensities = wf.to_numpy()  # (num_frames, 88) float32
            fps = wf.fps

        # Step 3: Per-key normalization — equalizes frequency-dependent CWT energy.
        if not args.no_equalize:
            print("Equalizing...")
            intensities = equalize.normalize_keys(intensities)

        # Step 4: Denoise — operates on continuous normalized values before bleed
        # suppression makes the signal binary.
        if not args.no_denoise:
            print("Denoising...")
            intensities = denoise_mod.denoise(
                intensities, reach=args.denoise_reach, cutoff=args.denoise_cutoff
            )

        # Step 5: Spectral bleed suppression — zero out non-peak keys per frame.
        if not args.no_equalize:
            print("Suppressing spectral bleed...")
            intensities = equalize.suppress_bleed(
                intensities, threshold=args.bleed_threshold
            )

        # Step 6: Smooth.
        if not args.no_smooth:
            print("Smoothing...")
            intensities = denoise_mod.smooth(
                intensities, kernel_size=args.smooth_kernel
            )

        # Step 7: Render video.
        print(f"Rendering {intensities.shape[0]} frames at {fps} fps...")
        render.render_video(
            intensities,
            args.input_wav,
            args.output_mp4,
            fps=fps,
            width=args.width,
            height=args.height,
        )

        print(f"Done: {args.output_mp4}")

    finally:
        # Cleanup temp .wvlt unless user wants to keep it.
        if temp_wvlt is not None and not args.keep_wvlt:
            try:
                os.unlink(wvlt_path)
            except OSError:
                pass
        elif temp_wvlt is not None and args.keep_wvlt:
            print(f"Kept intermediate file: {wvlt_path}")


if __name__ == "__main__":
    main()
