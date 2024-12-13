from pydantic import BaseModel
from enum import Enum

class AudioInput(BaseModel):
    filename: str
    a_freq: float
    low_freq: int
    high_freq: int
    stretch: int
    duration: int

class VideoProperties(BaseModel):
    output_filename: str
    framerate: int
    resolution_width: int
    resolution_height: int

class ExtractionType(Enum):
    FFT = 'FFT'
    CROSS_COR = 'CROSS-COR'

class IntermediateFile(BaseModel):
    filename: str
    extraction_type: ExtractionType

class Config(BaseModel):
    audio_input: AudioInput
    video_properties: VideoProperties
    intermediate_file: IntermediateFile
