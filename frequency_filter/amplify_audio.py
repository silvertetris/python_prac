import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

# Load your private audio file
audio_path = "2024-09-12 09-14-19.wav"  # Replace with your file
y, sr = librosa.load(audio_path, sr=None)  # Load audio with original sampling rate

# Amplify the audio
amplified_audio = y * 10
#amplified_audio = np.clip(amplified_audio, -1, 1)

# Save the amplified audio
output_path = "amplified_audio1.wav"
sf.write(output_path, amplified_audio, sr)

# Plot original and amplified waveforms
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(y, label="Original Audio", color='blue')
plt.title("Original Audio Waveform")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(amplified_audio, label="Amplified Audio", color='red')
plt.title("Amplified Audio Waveform")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
