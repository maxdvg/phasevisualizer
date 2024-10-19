import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

if __name__ == "__main__":
    WAV_FNAME = "/home/mvg/Music/bon.wav"
    samplerate, data = wavfile.read(WAV_FNAME)
    DURATION = 2 # seconds
    START_TIME = 14 # seconds
    x = np.linspace(START_TIME, START_TIME + DURATION, DURATION * samplerate)
    left_channel = data[:,1]
    vec = left_channel[START_TIME * samplerate:samplerate * (START_TIME + DURATION)]

    # Configure the plot
    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.plot(x, vec, label="zigzaggy sine")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Function that has two frequencies")
    plt.legend()
    # plt.xlim(0, DURATION)
    plt.grid(True)
    plt.show()

    # Since we're dealing with real numbers we can use real fft
    fourier_transform = np.fft.rfft(vec)

    # not concerned with phase, so get magnitude only
    # np.abs automatically calculates the magnitude of complex numbers
    magnitudes = np.abs(fourier_transform)
    max_index = np.argmax(magnitudes)
    # get frequency bin centers
    frequency_resolution = np.fft.rfftfreq(len(vec), 1 / samplerate)
    dominant_frequency = frequency_resolution[max_index]
    second_dominant_frequency = frequency_resolution[np.argpartition(magnitudes, -2)[-2]]
    print(f"Dominant frequency: {dominant_frequency}")
    print(f"Secondary frequency: {second_dominant_frequency}")

    print("hi!")