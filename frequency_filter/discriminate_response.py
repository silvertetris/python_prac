import librosa
import numpy as np
import matplotlib.pyplot as plt


def classify_audio(audio, sr, threshold=0.01):
    frame_length = 1024
    hop_length = 512
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length).flatten()

    # Normalize energy to range [0, 1]
    energy /= np.max(energy)

    # Check if energy decays below the threshold
    long_decay = np.any(energy > threshold)

    if long_decay:
        return "IIR-Suitable (Butterworth)"
    else:
        return "FIR-Suitable"


# Load the audio
audio_path = "2024-10-23 09-17-18.wav"
y, sr = librosa.load(audio_path, sr=None)

# Classify the audio
result = classify_audio(y, sr)
print("Classification Result:", result)


def classify_impulse_response(audio, sr, tail_samples=1024, decay_threshold=0.05):
    # Detect envelope of the audio signal
    envelope = np.abs(audio)

    # Normalize the envelope
    envelope /= np.max(envelope)

    # Check the decay in the tail of the envelope
    tail_energy = np.mean(envelope[-tail_samples:])

    if tail_energy > decay_threshold:
        return "IIR-Suitable (Butterworth)"
    else:
        return "FIR-Suitable"


# Classify based on impulse response-like characteristics
result = classify_impulse_response(y, sr)
print("Impulse Response Classification:", result)


def classify_audio_combined(audio, sr, energy_threshold=0.01, decay_threshold=0.05, tail_samples=1024):
    # Short-Time Energy Analysis
    frame_length = 1024
    hop_length = 512
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length).flatten()
    energy /= np.max(energy)

    # Long decay detection
    long_decay = np.any(energy > energy_threshold)

    # Impulse Response Tail Analysis
    envelope = np.abs(audio)
    envelope /= np.max(envelope)
    tail_energy = np.mean(envelope[-tail_samples:])

    # Combine criteria
    if long_decay or tail_energy > decay_threshold:
        return "IIR-Suitable (Butterworth)"
    else:
        return "FIR-Suitable"


# Classify audio using combined approach
result = classify_audio_combined(y, sr)
print("Combined Classification Result:", result)

plt.figure(figsize=(12, 6))

# Plot original audio waveform
plt.subplot(2, 1, 1)
plt.plot(y, label="Audio Signal")
plt.title("Audio Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()

# Plot short-time energy
frame_length = 1024
hop_length = 512
energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length).flatten()
plt.subplot(2, 1, 2)
plt.plot(energy, label="Short-Time Energy")
plt.title("Short-Time Energy (Decay Characteristics)")
plt.xlabel("Frames")
plt.ylabel("Energy")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
