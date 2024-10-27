from pydantic import BaseModel

class AudioInput(BaseModel):
    filename: str
    a_freq: float

class VideoProperties(BaseModel):
    output_filename: str
    framerate: int

class Config(BaseModel):
    audio_input: AudioInput
    video_properties: VideoProperties