# phasevisualizer

Music visualization software inspired by composer [Alexander Scriabin's](https://en.wikipedia.org/wiki/Alexander_Scriabin) synesthesia. Generates MP4 videos that display an 88-key piano histogram, coloring each key according to Scriabin's sound-to-color associations and lighting it in proportion to its instantaneous intensity in the audio.

**Example visualizations:**
- [Mndsgn — Camelblues](https://www.youtube.com/watch?v=2O-x44-ozGY)
- [Scriabin — Étude Op. 65 No. 3](https://youtu.be/BjpzHwe377s?feature=shared)

## How It Works

phasevisualizer uses Morlet Continuous Wavelet Transform (CWT) analysis from the companion
[music-visualizer](../music-visualizer) project to extract per-key intensities across all
88 standard piano keys (A0–C8). The analysis is performed by `wav_analyze`, a C utility that
produces a `.wvlt` binary file, which is then read via the `wvlt` Python package.

The processing pipeline is:

```
WAV → wav_analyze → .wvlt → normalize → denoise → bleed suppression → smooth → MP4
```

1. **wav_analyze** runs Morlet CWT at 88 piano-key frequencies, writing a `.wvlt` file.
2. **normalize** scales each key's intensity to [0, 1] using its per-key maximum,
   compensating for the scale-dependent energy falloff inherent to CWT.
3. **denoise** removes transient noise by requiring temporal persistence across frames.
4. **bleed suppression** zeroes non-peak keys per frame, eliminating spectral bleed
   from the wavelet's finite frequency bandwidth.
5. **smooth** applies a temporal moving average to reduce flicker.
6. **render** pipes raw RGB frames to ffmpeg, muxing in the original audio in one pass.

## Requirements

**Python:** 3.10 or later

**External tools (must be on `PATH` or specified via `--wav-analyze`):**
- `wav_analyze` — build from [music-visualizer](https://codeberg.org/zfzfpk/music-visualizer): `cd ../music-visualizer && make`
- `ffmpeg` — available from [ffmpeg.org](https://ffmpeg.org) or your system package manager

**Python dependencies** (managed by uv):
- `numpy`, `pydantic`, `tqdm`, `wvlt[numpy]` (contained in [music-visualizer](https://codeberg.org/zfzfpk/music-visualizer))

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```sh
# Build the wav_analyze binary (required)
cd ../music-visualizer && make && cd ../phasevisualizer

# Create the virtual environment and install dependencies
uv sync
```

## Usage

```sh
uv run python -m phasevisualizer INPUT_WAV OUTPUT_MP4 [OPTIONS]
```

### Positional arguments

| Argument | Description |
|---|---|
| `INPUT_WAV` | Path to the input WAV file. |
| `OUTPUT_MP4` | Path for the output MP4 video. |

### Analysis options

| Flag | Default | Description |
|---|---|---|
| `--fps FPS` | `60` | Frames per second for both analysis and video. |
| `--omega OMEGA` | `6.0` | Morlet ω₀ parameter. Higher values give sharper frequency resolution at the cost of time resolution. Valid range: [4.0, 20.0]. |
| `--sigma SIGMA` | `4.0` | Wavelet window truncation in standard deviations. |
| `--wav-analyze PATH` | `wav_analyze` | Path to the `wav_analyze` binary if not on `PATH`. |

### Video options

| Flag | Default | Description |
|---|---|---|
| `--width WIDTH` | `1920` | Output video width in pixels. |
| `--height HEIGHT` | `1080` | Output video height in pixels. |

### Processing options

| Flag | Default | Description |
|---|---|---|
| `--denoise-reach N` | `8` | Number of frames to look ahead/behind for temporal persistence check. |
| `--denoise-cutoff F` | `0.4` | Std-dev multiplier for the denoising threshold. Lower values are more lenient. |
| `--smooth-kernel N` | `5` | Temporal smoothing window size (must be odd). |
| `--bleed-threshold F` | `0.3` | Minimum intensity relative to the frame peak for a key to survive bleed suppression. |
| `--no-denoise` | — | Skip the denoising step. |
| `--no-smooth` | — | Skip the temporal smoothing step. |
| `--no-equalize` | — | Skip both normalization and bleed suppression. |

### Debug options

| Flag | Description |
|---|---|
| `--keep-wvlt` | Retain the intermediate `.wvlt` file instead of deleting it after rendering. |
| `--wvlt-file PATH` | Use a pre-existing `.wvlt` file, skipping the `wav_analyze` analysis step. |

### Examples

```sh
# Basic usage at default 1080p/60fps
uv run python -m phasevisualizer song.wav visualization.mp4

# Faster preview at 720p/30fps using a pre-analyzed .wvlt file
uv run python -m phasevisualizer song.wav preview.mp4 \
    --fps 30 --width 1280 --height 720 \
    --wvlt-file song.wvlt

# High frequency-resolution analysis (sharper note discrimination, longer kernels)
uv run python -m phasevisualizer song.wav visualization.mp4 --omega 12.0

# Keep the intermediate .wvlt for inspection or reuse
uv run python -m phasevisualizer song.wav visualization.mp4 --keep-wvlt
```

## Color Palette

Key colors follow Scriabin's synesthesia associations, mapped to the 12 chromatic note classes:

| Note | Color |
|---|---|
| A | Green |
| A# | Mauve |
| B | Light blue |
| C | Red |
| C# | Violet |
| D | Yellow |
| D# | Magenta |
| E | Ice blue |
| F | Dark red |
| F# | Blue-violet |
| G | Orange |
| G# | Lavender |

## Project Structure

```
phasevisualizer/
  __main__.py    CLI entry point and pipeline orchestration
  analyze.py     Subprocess wrapper for wav_analyze
  equalize.py    Per-key normalization and spectral bleed suppression
  denoise.py     Temporal denoising and smoothing
  palette.py     Scriabin color palette (88-key numpy array)
  render.py      Vectorized frame construction; ffmpeg pipe encoding
  config.py      Pydantic configuration models
```

## License

Mozilla Public License 2.0. See [LICENSE](LICENSE).
