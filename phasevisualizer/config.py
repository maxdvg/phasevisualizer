"""Configuration models for phasevisualizer."""

from pydantic import BaseModel, Field


class AnalysisConfig(BaseModel):
    """Parameters for wav_analyze CWT analysis."""

    fps: float = Field(default=60.0, gt=0, description="Analysis frames per second.")
    omega: float = Field(default=6.0, ge=4.0, le=20.0, description="Morlet omega0.")
    sigma: float = Field(default=4.0, ge=1.0, description="Window truncation sigma.")
    wav_analyze_bin: str = Field(
        default="wav_analyze", description="Path to wav_analyze binary."
    )


class VideoConfig(BaseModel):
    """Video output dimensions."""

    width: int = Field(default=1920, gt=0, description="Video width in pixels.")
    height: int = Field(default=1080, gt=0, description="Video height in pixels.")


class EqualizeConfig(BaseModel):
    """Frequency equalization and spectral bleed suppression."""

    enabled: bool = True
    bleed_threshold: float = Field(
        default=0.3, ge=0, le=1.0,
        description="Relative threshold for spectral bleed suppression.",
    )


class DenoiseConfig(BaseModel):
    """Temporal denoising parameters."""

    enabled: bool = True
    reach: int = Field(default=8, gt=0, description="Forward/backward window size.")
    cutoff: float = Field(
        default=0.4, gt=0, description="Std-dev threshold for persistence."
    )


class SmoothConfig(BaseModel):
    """Temporal smoothing parameters."""

    enabled: bool = True
    kernel_size: int = Field(
        default=5, gt=0, description="Smoothing kernel width (must be odd)."
    )


class Config(BaseModel):
    """Top-level configuration."""

    analysis: AnalysisConfig = AnalysisConfig()
    video: VideoConfig = VideoConfig()
    equalize: EqualizeConfig = EqualizeConfig()
    denoise: DenoiseConfig = DenoiseConfig()
    smooth: SmoothConfig = SmoothConfig()
