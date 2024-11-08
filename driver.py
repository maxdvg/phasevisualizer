from config import Config
from colorgen import ColorPallateSampler
import yaml
import cv2
from colorgen import Palette
import numpy as np
from scipy.io import wavfile
from math import floor
from tqdm import tqdm

if __name__ == "__main__":
    # Load in the config file
    with open("config.yaml", "r") as f:
        config_data = yaml.safe_load(f)
    config = Config.model_validate(config_data)

    # Initialize ColorPaletteSampler
    palette_sampler = ColorPallateSampler(config.audio_input.a_freq, Palette(), color_snapping=True)

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(config.video_properties.output_filename,
                                    fourcc,
                                    config.video_properties.framerate,
                                    (config.video_properties.resolution_width, config.video_properties.resolution_height))
    
    # start getting the dominant frequencies and writing the frame
    DURATION = 55 # seconds
    START_TIME = 0 # seconds

    num_frames = DURATION * config.video_properties.framerate
    # Read in the data from our audio file and cut it to the length/times we want
    sample_rate, data = wavfile.read(config.audio_input.filename)
    stretch = 9
    window_len = int(1 / config.audio_input.low_freq * stretch * sample_rate)
    # Make the window length even to simplify math
    window_len = window_len + 1 if window_len % 1 == 1 else window_len
    left_channel = data[:,1]
    vec = left_channel[sample_rate * START_TIME:sample_rate * (START_TIME + DURATION) + window_len]

    def normal_gaussian_window(num_std_devs: int, num_samples: int) -> np.ndarray:
        xs = np.linspace(-1 * num_std_devs, num_std_devs, num_samples)
        standard_dev = 1.0
        return 1.0 / (np.sqrt(2 * np.pi * standard_dev)) * np.exp(-1 * ((xs) ** 2.0) / (2 * standard_dev ** 2))

    window_fn = normal_gaussian_window(3, window_len)

    for frame_idx in tqdm(range(num_frames), desc="Processing frames", unit="frame"):
        samples_from_start_pos = floor(frame_idx * (1 / config.video_properties.framerate) * sample_rate)
        left_gaussian_frame = min(window_len // 2, samples_from_start_pos)
        right_gaussian_frame = min(window_len // 2, len(vec) - samples_from_start_pos)
        active_gaussian = window_fn[window_len // 2 - left_gaussian_frame: window_len // 2 + right_gaussian_frame]
        # Since we're dealing with real numbers we can use real fft
        fourier_transform = np.fft.rfft(vec[samples_from_start_pos - left_gaussian_frame:samples_from_start_pos + right_gaussian_frame] * active_gaussian)

        # not concerned with phase, so get magnitude only
        # np.abs automatically calculates the magnitude of complex numbers
        magnitudes = np.abs(fourier_transform)
        max_index = np.argmax(magnitudes)
        # get frequency bin centers
        frequency_resolution = np.fft.rfftfreq(len(vec), 1 / sample_rate)
        dominant_frequency = frequency_resolution[max_index]
        i = -2
        while dominant_frequency < config.audio_input.low_freq:
            dominant_frequency = frequency_resolution[np.argpartition(magnitudes, i)[i]]
            i -= 1
        
        frame = np.full((1080, 1920, 3), palette_sampler.color_for_freq(dominant_frequency), dtype=np.uint8)
        video_writer.write(frame)
    
    video_writer.release()