import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


if __name__ == "__main__":
    WAV_FNAME = "/home/mvg/Music/bon.wav"
    FREQ_LOW = 15 # Hz is about as low as humans can hear
    FRAME_RATE = 24 # FPS
    DURATION = 5 # seconds
    START_TIME = 14 # seconds
    NUM_FRAMES = DURATION * FRAME_RATE

    # Read in the data from our audio file and cut it to the length/times we want
    SAMPLERATE, data = wavfile.read(WAV_FNAME)
    WIN_LEN = int(1 / FREQ_LOW * 5 * SAMPLERATE) # technically we might be able to get away with one cycle, but 5 is nice to have
    x = np.linspace(START_TIME, START_TIME + DURATION, DURATION * SAMPLERATE)
    left_channel = data[:,1]
    vec = left_channel[START_TIME * SAMPLERATE:SAMPLERATE * (START_TIME + DURATION)]

    # # Plot the whole audio sample PCM as a sanity check
    # plt.figure(figsize=(10, 6))  # Set the figure size
    # plt.plot(x, vec)
    # plt.xlabel("Seconds (in recording)")
    # plt.ylabel("PCM")
    # plt.title("Audio PCM for Snippet")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    for frame_idx in range(NUM_FRAMES):
        # window looks forwards
        frame_start_time = frame_idx * (1 / FRAME_RATE) # seconds from beginning of snippet
        frame_start_pos = int(frame_start_time * SAMPLERATE)
        # TODO: actually deal with overrun errors at some point instead of ignoring
        if frame_start_pos + WIN_LEN < len(vec):
            window = vec[frame_start_pos:frame_start_pos+WIN_LEN]
            window_fft = np.fft.rfft(window)
            frequency_resolution = np.fft.rfftfreq(len(vec), 1 / SAMPLERATE)
            # Plot the frequency spectrum
            plt.plot(list(range(len(window_fft))), np.abs(window_fft))

            # Set plot labels and title
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude")
            plt.title("Frequency Spectrum")

            # Show the plot
            plt.show()

    # Since we're dealing with real numbers we can use real fft
    fourier_transform = np.fft.rfft(vec)

    # not concerned with phase, so get magnitude only
    # np.abs automatically calculates the magnitude of complex numbers
    magnitudes = np.abs(fourier_transform)
    max_index = np.argmax(magnitudes)
    # get frequency bin centers
    frequency_resolution = np.fft.rfftfreq(len(vec), 1 / SAMPLERATE)
    dominant_frequency = frequency_resolution[max_index]
    second_dominant_frequency = frequency_resolution[np.argpartition(magnitudes, -2)[-2]]
    print(f"Dominant frequency: {dominant_frequency}")
    print(f"Secondary frequency: {second_dominant_frequency}")

    print("hi!")