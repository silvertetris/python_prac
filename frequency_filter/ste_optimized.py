import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
import librosa
import soundfile as sf
import librosa

audio_file = "combined_audio.wav"
signal, sr = librosa.load(audio_file, sr=16000)  # 16kHz로 로드

# 오디오 정보 출력
print(f"오디오 샘플 길이: {len(signal)} 샘플")
print(f"샘플링 레이트: {sr} Hz")

# 오디오 신호 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.plot(signal, label="Audio Signal")
plt.title("Audio Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.show()


def enhanced_short_time_energy(signal, frame_size, frame_step, pre_emphasis=0.97, use_hamming=True):
    import numpy as np
    from scipy.signal import lfilter

    emphasized_signal = lfilter([1, -pre_emphasis], 1, signal)
    num_samples = len(emphasized_signal)
    num_frames = 1 + int((num_samples - frame_size) / frame_step)
    ste = np.zeros(num_frames)

    for i in range(num_frames):
        start_idx = i * frame_step
        end_idx = start_idx + frame_size
        frame = emphasized_signal[start_idx:end_idx]
        if use_hamming:
            frame *= np.hamming(frame_size)
        ste[i] = np.sum(frame ** 2)

    # 스무딩 윈도우 크기 증가 (더 부드러운 에너지 곡선을 위해)
    smoothing_window = 15  # 150ms (10ms * 15)
    smoothed_ste = np.convolve(ste, np.ones(smoothing_window) / smoothing_window, mode="same")

    # 동적 임계값 설정
    threshold = np.mean(smoothed_ste) * 0.1  # 임계값을 평균의 10%로 설정

    return ste, smoothed_ste, threshold


def detect_silence_regions(ste, threshold, frame_step, sr, min_silence_duration=1.0):
    """
    무음 구간 탐지 함수
    min_silence_duration: 최소 무음 구간 길이 (초 단위)
    """
    import numpy as np

    # 최소 무음 구간 프레임 수 계산
    min_silence_frames = int(min_silence_duration * sr / frame_step)

    # 무음 구간 탐지
    is_silence = ste < threshold
    silence_regions = []
    silence_start = None

    for i in range(len(is_silence)):
        if is_silence[i] and silence_start is None:
            silence_start = i
        elif not is_silence[i] and silence_start is not None:
            if (i - silence_start) >= min_silence_frames:
                # 시간으로 변환 (초 단위)
                start_time = silence_start * frame_step / sr
                end_time = i * frame_step / sr
                silence_regions.append((start_time, end_time))
            silence_start = None

    # 마지막 무음 구간 처리
    if silence_start is not None:
        if (len(is_silence) - silence_start) >= min_silence_frames:
            start_time = silence_start * frame_step / sr
            end_time = len(is_silence) * frame_step / sr
            silence_regions.append((start_time, end_time))

    return silence_regions


# STE 계산
frame_size = 400  # 25ms @ 16kHz
frame_step = 160  # 10ms @ 16kHz
ste, smoothed_ste, threshold = enhanced_short_time_energy(signal, frame_size, frame_step)

# 무음 구간 탐지
silence_regions = detect_silence_regions(smoothed_ste, threshold, frame_step, sr, min_silence_duration=1.0)

# 무음 구간 시각화
plt.figure(figsize=(15, 6))
plt.subplot(211)
plt.plot(np.arange(len(signal)) / sr, signal)
plt.title("Audio Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# 무음 구간 표시
for start, end in silence_regions:
    plt.axvspan(start, end, color='red', alpha=0.3)

plt.subplot(212)
plt.plot(np.arange(len(smoothed_ste)) * frame_step / sr, smoothed_ste, label="Smoothed STE")
plt.axhline(y=threshold, color='r', linestyle='--', label="Threshold")
plt.title("Short-Time Energy")
plt.xlabel("Time (s)")
plt.ylabel("Energy")
plt.legend()

plt.tight_layout()
plt.show()

frame_size = 400  # 25ms @ 16kHz
frame_step = 160  # 10ms @ 16kHz
ste, smoothed_ste, threshold = enhanced_short_time_energy(signal, frame_size, frame_step)


def enhanced_short_time_energy(signal, frame_size, frame_step, pre_emphasis=0.97, use_hamming=True):
    emphasized_signal = lfilter([1, -pre_emphasis], 1, signal)
    num_samples = len(emphasized_signal)
    num_frames = 1 + int((num_samples - frame_size) / frame_step)
    ste = np.zeros(num_frames)

    for i in range(num_frames):
        start_idx = i * frame_step
        end_idx = start_idx + frame_size
        frame = emphasized_signal[start_idx:end_idx]
        if use_hamming:
            frame *= np.hamming(frame_size)
        ste[i] = np.sum(frame ** 2)

    smoothing_window = 15
    smoothed_ste = np.convolve(ste, np.ones(smoothing_window) / smoothing_window, mode="same")
    threshold = np.mean(smoothed_ste) * 0.1

    return ste, smoothed_ste, threshold


def detect_silence_regions(ste, threshold, frame_step, sr, min_silence_duration=1.0):
    min_silence_frames = int(min_silence_duration * sr / frame_step)
    is_silence = ste < threshold
    silence_regions = []
    silence_start = None

    for i in range(len(is_silence)):
        if is_silence[i] and silence_start is None:
            silence_start = i
        elif not is_silence[i] and silence_start is not None:
            if (i - silence_start) >= min_silence_frames:
                start_time = silence_start * frame_step / sr
                end_time = i * frame_step / sr
                silence_regions.append((start_time, end_time))
            silence_start = None

    if silence_start is not None:
        if (len(is_silence) - silence_start) >= min_silence_frames:
            start_time = silence_start * frame_step / sr
            end_time = len(is_silence) * frame_step / sr
            silence_regions.append((start_time, end_time))

    return silence_regions


def remove_silence(signal, silence_regions, sr):
    """무음 구간을 제거하고 오디오를 연결하는 함수"""
    # 시간을 샘플 인덱스로 변환
    silence_samples = [(int(start * sr), int(end * sr)) for start, end in silence_regions]

    # 유지할 구간 찾기
    keep_regions = []
    current_pos = 0

    for start, end in silence_samples:
        if start > current_pos:
            keep_regions.append((current_pos, start))
        current_pos = end

    # 마지막 구간 처리
    if current_pos < len(signal):
        keep_regions.append((current_pos, len(signal)))

    # 오디오 구간 연결
    output_signal = []
    for start, end in keep_regions:
        # 페이드 인/아웃을 위한 설정
        fade_samples = int(0.01 * sr)  # 10ms
        chunk = signal[start:end]

        if len(chunk) > 2 * fade_samples:
            # 페이드 인
            chunk[:fade_samples] *= np.linspace(0, 1, fade_samples)
            # 페이드 아웃
            chunk[-fade_samples:] *= np.linspace(1, 0, fade_samples)

        output_signal.extend(chunk)

    return np.array(output_signal)


# 메인 프로세스
frame_size = 400  # 25ms @ 16kHz
frame_step = 160  # 10ms @ 16kHz

# STE 계산
ste, smoothed_ste, threshold = enhanced_short_time_energy(signal, frame_size, frame_step)

# 무음 구간 탐지
silence_regions = detect_silence_regions(smoothed_ste, threshold, frame_step, sr, min_silence_duration=1.0)

# 무음 구간 제거
output_signal = remove_silence(signal, silence_regions, sr)

# 결과 저장
output_file = 'output_without_silence.wav'
sf.write(output_file, output_signal, sr)

# 결과 출력
print(f"\n원본 오디오 길이: {len(signal) / sr:.2f}초")
print(f"처리된 오디오 길이: {len(output_signal) / sr:.2f}초")

# 시각화
plt.figure(figsize=(15, 8))

# 원본 신호
plt.subplot(211)
plt.plot(np.arange(len(signal)) / sr, signal)
plt.title("Original Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# 무음 제거된 신호
plt.subplot(212)
plt.plot(np.arange(len(output_signal)) / sr, output_signal)
plt.title("Signal without Silence")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()
