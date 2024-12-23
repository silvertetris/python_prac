import matplotlib.pyplot as plt
import numpy as np
import librosa
from scipy.fft import fft, fftfreq

# Load audio file
audio_path = "filtered_audio.wav"
y, sr = librosa.load(audio_path, sr=None)
y = y / np.max(np.abs(y))  # Normalize the signal

# Time domain (amplitude vs. time)
time = np.linspace(0, len(y) / sr, len(y))  # Time array

# FFT Analysis (frequency domain)
N = len(y)
T = 1 / sr
frequencies = fftfreq(N, T)[:N // 2]
fft_spectrum = np.abs(fft(y))[:N // 2]

# Plotting
plt.figure(figsize=(12, 8))

# Plot 1: Amplitude vs. Time
plt.subplot(2, 1, 1)
plt.plot(time, y, color='blue')
plt.title("Amplitude Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

# Plot 2: Frequency Spectrum
plt.subplot(2, 1, 2)
plt.plot(frequencies, fft_spectrum, color='red')
plt.title("Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 10000)  # Focus on 0-10,000 Hz range
plt.grid(True)

# Show plots
plt.tight_layout()
plt.show()
