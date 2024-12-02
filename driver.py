from config import Config
from colorgen import ColorPallateSampler, NOTE_ORDER
import yaml
from colorgen import Palette
import numpy as np
from scipy.io import wavfile
from math import floor
from tqdm import tqdm
from math import pow
import operator

if __name__ == "__main__":
    # Load in the config file
    with open("config.yaml", "r") as f:
        config_data = yaml.safe_load(f)
    config = Config.model_validate(config_data)

    # Initialize ColorPaletteSampler
    palette_sampler = ColorPallateSampler(config.audio_input.a_freq, Palette(), color_snapping=True)

    # start getting the dominant frequencies and writing the frame
    DURATION = config.audio_input.duration
    START_TIME = 0 # seconds

    num_frames = DURATION * config.video_properties.framerate
    # Read in the data from our audio file and cut it to the length/times we want
    sample_rate, data = wavfile.read(config.audio_input.filename)
    window_len = int(1 / config.audio_input.low_freq * config.audio_input.stretch * sample_rate)
    # Make the window length even to simplify math
    window_len = window_len + 1 if window_len % 1 == 1 else window_len
    left_channel = data[:,1]
    vec = left_channel[sample_rate * START_TIME:sample_rate * (START_TIME + DURATION) + window_len]

    def normal_gaussian_window(num_std_devs: int, num_samples: int) -> np.ndarray:
        xs = np.linspace(-1 * num_std_devs, num_std_devs, num_samples)
        standard_dev = 1.0
        return 1.0 / (np.sqrt(2 * np.pi * standard_dev)) * np.exp(-1 * ((xs) ** 2.0) / (2 * standard_dev ** 2))

    window_fn = normal_gaussian_window(3, window_len)

    def generate_note_to_freq(freq_low: float, freq_high: float):
        note_to_freq: dict[str, float] = {'An4': config.audio_input.a_freq}
        cur_freq = config.audio_input.a_freq
        cur_note = 'An4'
        def create_adder(n):
            def subtract(x):
                return x + n
            return subtract

        def helper(climb_descend: int, cur_freq: float, cur_note: str):
            crementer = create_adder(climb_descend)
            if climb_descend == 1:
                comperator = operator.lt
                compval = freq_high
            elif climb_descend == -1:
                comperator = operator.gt
                compval = freq_low
            else:
                raise ValueError("Only defined for -1 and 1")
            while comperator(cur_freq, compval):
                cur_note_name = cur_note[:2]
                cur_note_octave = int(cur_note[2])
                cur_note_idx = NOTE_ORDER.index(cur_note_name)
                if cur_note_idx == len(NOTE_ORDER) - 1:
                    next_note_octave = crementer(cur_note_octave)
                else:
                    next_note_octave = cur_note_octave
                next_note_name = NOTE_ORDER[(crementer(cur_note_idx)) % len(NOTE_ORDER)]
                next_note = next_note_name + str(next_note_octave)
                next_freq = cur_freq * pow(2, climb_descend / 12.0)
                note_to_freq[next_note] = next_freq
                cur_freq = next_freq
                cur_note = next_note
        
        helper(1, cur_freq, cur_note)
        helper(-1, cur_freq, cur_note)
        return note_to_freq
    
    note_to_freq = generate_note_to_freq(30, 5000)

    note_strengths: np.ndarray = np.empty([num_frames, len(NOTE_ORDER)])
    for frame_idx in tqdm(range(num_frames), desc="Processing frames", unit="frame"):
        samples_from_start_pos = floor(frame_idx * (1 / config.video_properties.framerate) * sample_rate)
        left_gaussian_frame = min(window_len // 2, samples_from_start_pos)
        right_gaussian_frame = min(window_len // 2, len(vec) - samples_from_start_pos)
        active_gaussian = window_fn[window_len // 2 - left_gaussian_frame: window_len // 2 + right_gaussian_frame]
        signal = vec[samples_from_start_pos - left_gaussian_frame:samples_from_start_pos + right_gaussian_frame]
        # Since we're dealing with real numbers we can use real fft
        fourier_transform = np.fft.rfft(signal * active_gaussian)

        # not concerned with phase, so get magnitude only
        # np.abs automatically calculates the magnitude of complex numbers
        magnitudes = np.abs(fourier_transform)
        max_index = np.argmax(magnitudes)
        # get frequency bin centers
        frequency_resolution = np.fft.rfftfreq(len(signal), 1.0 / sample_rate)
        note_intensities: dict[str, float] = {}


        for idx, freq in enumerate(frequency_resolution):
            # What is the closest note? Throw out high and low
            if freq > 30 and freq < 5000:
                note_intensities[palette_sampler.closest_note(freq)] = note_intensities.get(palette_sampler.closest_note(freq), 0.0) + magnitudes[idx]
        for note, intensity in note_intensities.items():
            note_strengths[frame_idx][NOTE_ORDER.index(note)] = intensity
    
    with open(config.intermediate_file.filename, 'wb') as f:
        np.save(f, note_strengths)