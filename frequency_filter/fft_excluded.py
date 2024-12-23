import numpy as np
import librosa
from scipy.fft import fft, fftfreq
from scipy.signal import butter, lfilter
import soundfile as sf

audio_path = "filtered_audio_no_speech.wav"
y, sr = librosa.load(audio_path, sr=None)
y = y / np.max(np.abs(y)) #여기서부터 정상화 코드 추가 요망

# FFT Analysis
N = len(y)
T = 1 / sr
frequencies = fftfreq(N, T)[:N // 2]
fft_spectrum = np.abs(fft(y))[:N // 2]

freq_limit = 5000 #0부터 5천주파수 까지
freq_mask = frequencies <= freq_limit
frequencies = frequencies[freq_mask]
fft_spectrum = fft_spectrum[freq_mask]

magnitude_threshold = np.percentile(fft_spectrum, 0.01) # 자를 수치 조정 임계값 (각 주파수 마다의 magnitude에 대한)
low_sound_indices = np.where(fft_spectrum < magnitude_threshold)[0] #임계값 보다 작은 값들 수 (indices)
print(f"FFT Magnitude Spectrum: {fft_spectrum}")
print(f"Magnitude Threshold: {magnitude_threshold}")
print(f"Low Sound Indices: {low_sound_indices}")


low_sound_ranges = []
#임계값보다 낮은 애들 범위 산출
if len(low_sound_indices) > 0:
    start = frequencies[low_sound_indices[0]]
    for i in range(1, len(low_sound_indices)):
        if low_sound_indices[i] != low_sound_indices[i - 1] + 1:
            end = frequencies[low_sound_indices[i - 1]]
            low_sound_ranges.append((start, end))
            start = frequencies[low_sound_indices[i]]
    end = frequencies[low_sound_indices[-1]]
    low_sound_ranges.append((start, end))

print("Low Sound Ranges :", low_sound_ranges)

attach_ranges = []
for start, end in low_sound_ranges:
    if not attach_ranges or start > attach_ranges[-1][1]:
        attach_ranges.append((start, end))
    else:
        attach_ranges[-1] = (attach_ranges[-1][0], max(attach_ranges[-1][1], end))

print("Attached Ranges :", attach_ranges)