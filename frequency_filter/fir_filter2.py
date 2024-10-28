import librosa
import numpy as np
from scipy.signal import firwin, lfilter
import matplotlib.pyplot as plt
import ffmpeg

# Define input and output file paths
input_file = '1015.mp4'
output_file = '1015.wav'

# Convert .mp4 to .wav (uncomment if needed)
# ffmpeg.input(input_file).output(output_file, format='wav', acodec='pcm_s16le', ar=44100, ac=2).run()

# Load the audio file
y, sr = librosa.load(output_file, sr=None)

# Define FIR bandpass filter parameters
lowcut = 300.0
highcut = 3000.0
numtaps = 101

# Function to design an FIR filter
def fir_bandpass(lowcut, highcut, sr, numtaps=101):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    taps = firwin(numtaps, [low, high], pass_zero=False)
    return taps

# Apply the FIR bandpass filter to isolate the target frequency range
def fir_bandpass_filter(data, lowcut, highcut, sr, numtaps=101):
    taps = fir_bandpass(lowcut, highcut, sr, numtaps)
    filtered_data = lfilter(taps, 1.0, data)
    return filtered_data

# Filter the signal with the FIR bandpass filter
filtered_audio = fir_bandpass_filter(y, lowcut, highcut, sr, numtaps=numtaps)

# Zero-out parts of the signal outside the frequency range based on RMS energy threshold
frame_length = 1024
hop_length = 512
energy_threshold = 0.01  # Adjust this threshold as needed

# Calculate RMS energy of the filtered signal in frames
energy = librosa.feature.rms(y=filtered_audio, frame_length=frame_length, hop_length=hop_length).flatten()

# Generate time stamps for frames
time_stamps = librosa.times_like(energy, sr=sr, hop_length=hop_length)

# Initialize a zero array for output where we’ll keep only parts in the frequency range
output_audio = np.zeros_like(y)

# Fill only the segments that meet the energy threshold in the output array
for i, e in enumerate(energy):
    if e > energy_threshold:  # If energy is above threshold, retain the segment
        start = i * hop_length
        end = min(start + frame_length, len(y))
        output_audio[start:end] = filtered_audio[start:end]

# Generate time array for plotting
time = np.linspace(0, len(y) / sr, len(y))

# Plotting
plt.figure(figsize=(12, 6))

# Original Signal Plot
plt.subplot(3, 1, 1)
plt.plot(time, y, label='Original Signal')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Original Signal")
plt.legend()
plt.grid(True)

# Filtered Signal Plot
plt.subplot(3, 1, 2)
plt.plot(time, filtered_audio, label='Filtered Signal', color='orange')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Filtered Signal (Without Zeroing)")
plt.legend()
plt.grid(True)

# Final Output Signal with Zeroed Segments Outside Frequency Range
plt.subplot(3, 1, 1)
plt.plot(time, output_audio, label='Filtered Signal with Zeroed Out Segments', color='green')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Filtered Signal (Zeroed Out Segments Outside Frequency Range)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
