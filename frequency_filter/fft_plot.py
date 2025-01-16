import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.signal import butter, lfilter
from scipy.fft import fft, fftfreq, ifft

audio_path = "2024-10-23 09-17-18.wav"  # 오디오 파일
y, sr = librosa.load(audio_path, sr=None)
y = y / np.max(np.abs(y))

# Step 1: FFT Analysis
N = len(y)  # time series length => 샘플 수 (쪼개진 시간 배열 총 크기)
T = 1 / sr  # Sample rate 역수 => 주기
frequencies = fftfreq(N, T)[:N // 2]  # Positive frequencies
fft_spectrum = np.abs(fft(y))[:N // 2]  # Compute FFT and take positive half

# Plot
plt.figure(figsize=(12, 6))
plt.plot(frequencies, fft_spectrum, label="FFT Spectrum")
plt.title("Frequency Spectrum of the Audio")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 10000)
plt.grid()
plt.legend()
plt.show()
