import numpy as np
import librosa
import soundfile as sf

# Load the two audio files
audio1, sr1 = librosa.load("amplified_audio1.wav", sr=None)  # Load with original sampling rate
audio2, sr2 = librosa.load("amplified_audio2.wav", sr=None)

# Ensure both audio files have the same sampling rate
if sr1 != sr2:
    raise ValueError("Sampling rates of the audio files do not match.")

# Append the audio arrays
combined_audio = np.concatenate((audio1, audio2))

# Save the combined audio
sf.write("combined_audio.wav", combined_audio, sr1)
