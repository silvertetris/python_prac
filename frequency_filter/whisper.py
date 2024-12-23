import numpy as np
import soundfile as sf
from scipy.signal import firwin, lfilter
import librosa
from yt_dlp import YoutubeDL


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
audio_path = "2024-09-11 08-13-07.mkv"
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


def fir_bandpass_filter(data, lowcut, highcut, sr, numtaps=400):
    # FIR 필터 설계 및 적용
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    taps = firwin(numtaps, [low, high], pass_zero=False)
    filtered_data = lfilter(taps, 1.0, data)
    return filtered_data


def calculate_short_time_energy(y, frame_length, hop_length):
    # Short-Time Energy (STE) 계산
    ste = np.array([
        np.sum(y[i:i + frame_length] ** 2)
        for i in range(0, len(y) - frame_length + 1, hop_length)
    ])
    return ste


def calculate_rms(y, frame_length, hop_length):
    # Root Mean Square (RMS) 계산
    rms = np.array([
        np.sqrt(np.mean(y[i:i + frame_length] ** 2))
        for i in range(0, len(y) - frame_length + 1, hop_length)
    ])
    return rms


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


# percentile 함수를 이용한 임계값 동적 변환 함수
def calculate_dynamic_thresholds_with_percentiles(ste, rms, ste_percentile=55, rms_percentile=55):
    # Use percentiles to set thresholds dynamically
    ste_threshold = np.percentile(ste, ste_percentile)
    rms_threshold = np.percentile(rms, rms_percentile)
    return ste_threshold, rms_threshold


def discriminate_audio(audio, sr, ste_threshold, rms_threshold, tail_samples=1024):
    frame_length = 1024
    hop_length = 512
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length).flatten()

    # Normalize energy to range [0, 1]
    energy /= np.max(energy)

    # Check if energy decays below the threshold
    tail_energy = np.any(energy > ste_threshold)
    long_decay = np.any(energy > rms_threshold)

    if long_decay or tail_energy:
        return "IIR-Suitable (Butterworth)"
    else:
        return "FIR-Suitable"


def fir_ste_rms_pipeline(input_file, output_file, lowcut=300, highcut=3000, frame_length=1024, hop_length=512):
    # 1. 변환된 wav 파일 로드
    y, sr = librosa.load(input_file, sr=None)

    # 2. FIR 필터 적용
    filtered_audio = fir_bandpass_filter(y, lowcut, highcut, sr)
    sf.write("filtered_audio.wav", filtered_audio, sr)

    # 3. Short-Time Energy (STE)와 RMS 계산
    ste = calculate_short_time_energy(filtered_audio, frame_length, hop_length)
    rms = calculate_rms(filtered_audio, frame_length, hop_length)
    # 4. STE와 RMS 기반 음성 활성 구간 탐지
    voice_segments = []
    start = None

    # 동적 임계값 할당
    ste_threshold, rms_threshold = calculate_dynamic_thresholds_with_percentiles(ste, rms)

    for i, (energy, rms_val) in enumerate(zip(ste, rms)):
        is_voice = energy > ste_threshold and rms_val > rms_threshold
        if is_voice and start is None:
            start = i * hop_length
        elif not is_voice and start is not None:
            end = i * hop_length + frame_length
            voice_segments.append((start, end))
            start = None
    if start is not None:
        voice_segments.append((start, len(filtered_audio)))

    # STE, RMS 에 걸리는 시간대 추출
    # time_code는 리스트 [(start_time, end_time)] 형식
    time_codes = [(start / sr, end / sr) for start, end in voice_segments]  # 각 길이 시점대(start, end) 나누기 전체길이 => 시간대
    print("time_code: ", time_codes)

    for idx, (start_time, end_time) in enumerate(time_codes):
        # segment 단위로 변환해서 original audio에서 짜르기
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

    revised_segment = np.concatenate([y[start_sample:end_sample] for start_sample, end_sample in voice_segments])

    # original segment 조합된 영상 추출
    sf.write(output_file, revised_segment, sr)

    # 5. 최종 음성 데이터 조합
    final_data = np.concatenate([filtered_audio[start:end] for start, end in voice_segments])

    # 6. 결과 저장 (Google Drive 내 경로)
    sf.write("filtered_revised_audio.wav", final_data, sr)

    # 로그 출력
    print(f"Processing complete. Output saved to {output_file}")
    print(f"Original duration: {len(y) / sr:.2f}s")
    print(f"Processed duration: {len(final_data) / sr:.2f}s")
    print(f"Number of voice segments: {len(voice_segments)}")
