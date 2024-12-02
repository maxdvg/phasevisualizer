import yaml
import numpy as np
from config import Config
from colorgen import ColorPallateSampler, NOTE_ORDER, Palette
import cv2
from tqdm import tqdm
import subprocess
from typing import Optional
import pickle

NUM_NOTES = len(NOTE_ORDER)

def write_histogram_frame(hist_data: np.ndarray, video_writer: cv2.VideoWriter, freq_array: np.ndarray,
                           freq_to_note: dict[float, str], global_max: Optional[float] = None,
                           flicker_reduction_data: Optional[np.ndarray] = None):
        max_height = max(hist_data)
        frame = np.zeros((config.video_properties.resolution_height, config.video_properties.resolution_width, 3), dtype=np.uint8)
        if max_height != 0.0:
            min_height = min(hist_data)
            for note_idx, note_stren in enumerate(hist_data):
                bar_left = (frame.shape[1] // hist_data.shape[0]) * note_idx
                bar_right = (frame.shape[1] // hist_data.shape[0]) * (note_idx + 1)
                normalized_strength = (note_stren - min_height) / (max_height - min_height)
                # if global_max is None:
                #     normalized_strength = note_stren / max_height
                # else:
                #     normalized_strength = note_stren / global_max
                # height_in_px = min(config.video_properties.resolution_height, int(normalized_strength * config.video_properties.resolution_height))
                if normalized_strength > 0:
                    if flicker_reduction_data is not None:
                        if flicker_reduction_data[:,note_idx].mean() > note_stren * .5:
                            frame[:,bar_left:bar_right,] = palette_sampler.color_for_note(freq_to_note[freq_array[note_idx]][:2]) * normalized_strength
                    else:
                            frame[:,bar_left:bar_right,] = palette_sampler.color_for_note(freq_to_note[freq_array[note_idx]][:2]) * normalized_strength
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

def denoise(full_data: np.ndarray, smoothing_reach: int = 6, cutoff: float = .35) -> np.ndarray:
    cleaned_data_mask: np.ndarray = np.zeros(full_data.shape)
    # Sweep forwards, checking if the note persists above median for durating of smoothing-reach
    for idx in range(len(full_data) - smoothing_reach):
        forward_look = full_data[idx:idx + smoothing_reach]
        medians = np.mean(forward_look, axis=0)
        # Calculate the mean and standard deviation of means lol
        mean = np.mean(medians)
        std_dev = np.std(medians)
        # Find indices where elements are cutoff std dev above the mean
        indices = np.where(medians > mean + cutoff * std_dev)[0]
        cleaned_data_mask[idx][indices] = 1
    # Sweep backwards
    for idx in range(smoothing_reach, len(full_data)):
        backwards_look = full_data[idx - smoothing_reach:idx]
        medians = np.mean(backwards_look, axis=0)
        # Calculate the mean and standard deviation of means lol
        mean = np.mean(medians)
        std_dev = np.std(medians)
        # Find indices where elements are cutoff std dev above the mean
        indices = np.where(medians > mean + cutoff * std_dev)[0]
        cleaned_data_mask[idx][indices] *= 1

    cleaned_data = full_data * cleaned_data_mask
    return cleaned_data

def smoother(full_data: np.ndarray, smoothing_kernel_len: int = 5) -> np.ndarray:
     if smoothing_kernel_len % 2 != 1:
          raise ValueError("I didn't implement smoother for non-odd kernel lengths yet")
     smoothed_data = np.empty(full_data.shape)
     one_side_kernel_len = smoothing_kernel_len // 2 + 1
     start_idx = one_side_kernel_len
     end_idx = len(full_data) - one_side_kernel_len
     smoothed_data[:start_idx] = 0 # Change later if you want it to actually be good
     smoothed_data[end_idx:] = 0
     for idx in range(start_idx, end_idx):
          smoothed_data[idx] = full_data[idx - one_side_kernel_len:idx + one_side_kernel_len].sum(axis=0) / float(smoothing_kernel_len)
     return smoothed_data

if __name__ == "__main__":
    # Load in the config file
    with open("config.yaml", "r") as f:
        config_data = yaml.safe_load(f)
    config = Config.model_validate(config_data)

    # Load histograms generated by driver
    with open(config.intermediate_file.filename, 'rb') as f:
        note_intensities = np.load(f)
    with open('data.pickle', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        note_to_freq: dict[str, float] = pickle.load(f)
    freq_to_note = {note: freq for freq, note in note_to_freq.items()}
    freq_array = np.sort(np.array(list(freq_to_note.keys())))

    # Initialize ColorPaletteSampler
    palette_sampler = ColorPallateSampler(config.audio_input.a_freq, Palette(), color_snapping=True)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(config.video_properties.output_filename,
                                    fourcc,
                                    config.video_properties.framerate,
                                    (config.video_properties.resolution_width, config.video_properties.resolution_height))
    
    num_frames = len(note_intensities)

    cleaned = denoise(note_intensities, smoothing_reach=8, cutoff=.4)
    cleaned = smoother(cleaned, smoothing_kernel_len=5)
    for frame_idx in tqdm(range(num_frames), desc="Generating video", unit="frame"):
        cur_frame_hist = cleaned[frame_idx]
        write_histogram_frame(cur_frame_hist, video_writer, freq_array, freq_to_note)

    video_writer.release()

    subprocess.call(f"ffmpeg -i {config.video_properties.output_filename} -i {config.audio_input.filename} -y -c:v copy -c:a aac output.mp4", shell=True)


