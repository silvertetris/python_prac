import librosa
import numpy as np
from scipy.signal import firwin, lfilter
import matplotlib.pyplot as plt
import ffmpeg

input_file = '2024-10-23 09-17-18.mkv'
output_file = '2024-10-23 09-17-18.wav'
# Use ffmpeg to extract and convert audio
ffmpeg.input(input_file).output(output_file, format='wav', acodec='pcm_s16le', ar=44100, ac=2).run()


# Function to design an FIR filter
def fir_bandpass(lowcut, highcut, sr, numtaps=101):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    taps = firwin(numtaps, [low, high], pass_zero=False)
    return taps

# Function to apply the FIR filter to a signal
def fir_bandpass_filter(data, lowcut, highcut, sr, numtaps=101):
    taps = fir_bandpass(lowcut, highcut, sr, numtaps)
    filtered_data = lfilter(taps, 1.0, data)
    return filtered_data

# Load an example audio file using librosa
y, sr = librosa.load(output_file, sr=None)

# Define the frequency range you want to detect (e.g., 500 Hz to 1500 Hz)
lowcut = 300.0
highcut = 3000.0
numtaps = 101

# Apply the FIR bandpass filter to the audio signal
filtered_audio = fir_bandpass_filter(y, lowcut, highcut, sr, numtaps=numtaps)

# Parameters for detecting energy in frames
hop_length = 512
frame_length = 1024
energy_threshold = 0.01  # Adjust based on signal strength

# Calculate the energy in each frame
energy = librosa.feature.rms(y=filtered_audio, frame_length=frame_length, hop_length=hop_length).flatten()
time_stamps = librosa.times_like(energy, sr=sr, hop_length=hop_length)

# Detect times where the energy indicates presence of the frequency range
presence = energy > energy_threshold
filtered_time_stamps = time_stamps[presence]

# Output the times where the specific frequency is detected
if filtered_time_stamps.size > 0:
    print("Times containing the specified frequency range:")
    for start, end in zip(filtered_time_stamps, filtered_time_stamps[1:]):
        if end - start > (hop_length / sr):  # Check for continuous segments
            print(f"{start:.2f} s to {end:.2f} s")
else:
    print("No segments contain the specified frequency range.")

# Plot original and filtered audio signals with detected segments
plt.figure(figsize=(12, 8))

# Original Signal
plt.subplot(2, 1, 1)
plt.plot(y, label="Original Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.title("Original Audio Signal")
plt.legend()

# Filtered Signal with Detected Segments
plt.subplot(2, 1, 2)
plt.plot(filtered_audio, label="Filtered Signal", color='orange')
plt.plot(np.arange(len(presence)) * hop_length, presence * np.max(filtered_audio), 'r', alpha=0.5, label="Detected Segments")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.title("Filtered Audio Signal with Detected Segments")
plt.legend()

plt.tight_layout()
plt.show()
