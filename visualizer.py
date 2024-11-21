# Copyright Max Van Gelder 2024
# This code is garbage idgaf I just want to vibe with pretty colors and stuff
# Maybe I'll make it actually good code later maybe not

import pydub
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

MUSIC_FILE = "/home/mvg/Music/sweep.mp3"
FRAME_RATE = 30 # fps for animaton
WINDOW_LEN = 300 # window duration in MS
NUM_SECS = 30 # how many seconds to generate animation for
SMOOTHING_KERNAL_LEN = 5 # for finding peaks and valley in the trigger detection using uniform smoothing

song = pydub.AudioSegment.from_mp3(MUSIC_FILE)

# I ripped this function straight from a stackoverflow answer
def pydub_to_np(audio: pydub.AudioSegment) -> tuple[np.ndarray, int]:
    """
    Converts pydub audio segment into np.float32 of shape [duration_in_seconds*sample_rate, channels],
    where each value is in range [-1.0, 1.0]. 
    Returns tuple (audio_np_array, sample_rate).
    """
    return np.array(audio.get_array_of_samples(), dtype=np.float32).reshape((-1, audio.channels)) / (
            1 << (8 * audio.sample_width - 1)), audio.frame_rate

song = pydub_to_np(song)
#TODO: dual channels
song_array: np.ndarray = (song[0][:, 0] + song[0][:, 1]) / 2 # Average stereo channels into mono
song_array = song_array[:song[1] * 5] # grab first five seconds

smoothing_kernel = np.ones(SMOOTHING_KERNAL_LEN) / SMOOTHING_KERNAL_LEN
smoothed_song = np.convolve(song_array, smoothing_kernel, mode='same')


# TODO: Use fourier transform to find strongest frequency, so that the oscilloscope can display a stable waveform
def update(frame: int):
    # Find nearest trigger point (looking backwards only)
    frame_to_time = lambda f: int(1000 * (f / FRAME_RATE))
    frame_start_time = frame_to_time(frame)
    prev_frame_start_time = frame_to_time(frame - 1)
    if prev_frame_start_time < 0:
        line.set_ydata(song_array[frame_start_time:frame_start_time + WINDOW_LEN])
        return line,
    trigger = 0 # Trigger point for taking the image
    for sample_time in range(prev_frame_start_time, frame_start_time):
        if song_array[sample_time] > trigger and song_array[sample_time - 1] < trigger:
                line.set_ydata(song_array[frame_start_time:frame_start_time + WINDOW_LEN])
    return line,

fig, ax = plt.subplots()

# Initialize the plot (empty line)
line, = ax.plot(np.zeros(WINDOW_LEN))
ax.set_ylim(-1, 1) # Signal is normalized -1.0 to 1.0 so this is sufficient

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=FRAME_RATE * NUM_SECS, interval=1000 / FRAME_RATE)

# Save the animation as a video
ani.save('animation.mp4', writer='ffmpeg')