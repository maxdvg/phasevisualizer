import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Two sin waves on top of each other of different frequencies
    # We will pretend this is a musical recording that lasts 2 seconds
    SAMPLING_RATE = 1000 # Hz, this is arbitrary make-believe
    DURATION = 2 # seconds
    x = np.linspace(0, DURATION, DURATION * SAMPLING_RATE) # make a 2 second long "recording"
    sin = np.sin(5 * np.pi * x) # make a 2.5Hz sine wave w/ magnitude 1
    quick_sin = .15 * np.sin(50 * np.pi * x) # make a 25Hz sine wave w/ magnitude .15
    vec = sin + quick_sin

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
    frequency_resolution = np.fft.rfftfreq(len(vec), 1 / SAMPLING_RATE)
    dominant_frequency = frequency_resolution[max_index]
    print("Dominant frequency:", dominant_frequency)

    # # Plot the Fourier transform as a histogram
    # plt.figure(figsize=(10, 6))
    # plt.hist(np.abs(fourier_transform), bins=1000, color='blue', alpha=0.7)
    # plt.xlim(1, 20)  # Set the x-axis limits to 1 and 20
    # plt.xlabel('Frequency')
    # plt.ylabel('Magnitude')
    # plt.title('Fourier Transform Histogram')
    # plt.grid(True)
    # plt.show()


    # # Show the plot
    # plt.show()

    print("hi!")