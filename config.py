from pydantic import BaseModel

class AudioInput(BaseModel):
    filename: str
    a_freq: float
    low_freq: int

class VideoProperties(BaseModel):
    output_filename: str
    framerate: int
    resolution_width: int
    resolution_height: int

class Config(BaseModel):
    audio_input: AudioInput
    video_properties: VideoProperties
