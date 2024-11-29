from pydantic import BaseModel

class AudioInput(BaseModel):
    filename: str
    a_freq: float
    low_freq: int
    stretch: int
    duration: int

class VideoProperties(BaseModel):
    output_filename: str
    framerate: int
    resolution_width: int
    resolution_height: int

class IntermediateFile(BaseModel):
    filename: str

class Config(BaseModel):
    audio_input: AudioInput
    video_properties: VideoProperties
    intermediate_file: IntermediateFile
