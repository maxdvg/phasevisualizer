import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from config import Config
import yaml


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config_data = yaml.safe_load(f)
    config = Config.model_validate(config_data)

    DURATION = 5 # seconds
    START_TIME = 14 # seconds

    num_frames = DURATION * config.video_properties.framerate
    # Read in the data from our audio file and cut it to the length/times we want
    sample_rate, data = wavfile.read(config.audio_input.filename)
    window_len = int(1 / config.audio_input.low_freq * 5 * sample_rate) # technically we might be able to get away with one cycle, but 5 is nice to have
    x = np.linspace(START_TIME, START_TIME + DURATION, DURATION * sample_rate)
    left_channel = data[:,1]
    vec = left_channel[START_TIME * sample_rate:sample_rate * (START_TIME + DURATION)]

    # Since we're dealing with real numbers we can use real fft
    fourier_transform = np.fft.rfft(vec)

    # not concerned with phase, so get magnitude only
    # np.abs automatically calculates the magnitude of complex numbers
    magnitudes = np.abs(fourier_transform)
    max_index = np.argmax(magnitudes)
    # get frequency bin centers
    frequency_resolution = np.fft.rfftfreq(len(vec), 1 / sample_rate)
    dominant_frequency = frequency_resolution[max_index]
    second_dominant_frequency = frequency_resolution[np.argpartition(magnitudes, -2)[-2]]
    print(f"Dominant frequency: {dominant_frequency}")
    print(f"Secondary frequency: {second_dominant_frequency}")

    print("hi!")