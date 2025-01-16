import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
import librosa
from yt_dlp import YoutubeDL


def butter_bandpass_filter(data, lowcut, highcut, sr, order=4):
    # Butterworth 필터 설계 및 적용
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = lfilter(b, a, data)
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

def fir_ste_rms_pipeline(input_file, output_file, lowcut=300, highcut=3000, frame_length=1024, hop_length=512):
    # 1. 변환된 wav 파일 로드
    y, sr = librosa.load(input_file, sr=None)

    # 2. Butterworth 필터 적용
    filtered_audio = butter_bandpass_filter(y, lowcut, highcut, sr)
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
    time_codes = [(start / sr, end / sr) for start, end in voice_segments]
    print("time_code: ", time_codes)

    revised_segment = np.concatenate([y[start:end] for start, end in voice_segments])

    # Original segment 조합된 영상 추출
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

# 유튜브 영상 추출 함수
def extract_audio_from_youtube(URLS, title):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': title,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'max_filesize': 10 * 1024 * 1024
    }
    with YoutubeDL(ydl_opts) as ydl:
        error_code = ydl.download(URLS)

# Percentile 함수를 이용한 임계값 동적 변환 함수
def calculate_dynamic_thresholds_with_percentiles(ste, rms, ste_percentile=55, rms_percentile=55):
    ste_threshold = np.percentile(ste, ste_percentile)
    rms_threshold = np.percentile(rms, rms_percentile)
    return ste_threshold, rms_threshold

URLS = 'https://www.youtube.com/watch?v=-vjv7lI7UVE'  # 오디오 추출 유튜브 링크
title = "yt_extracted_audio"  # 생성될 제목
# extract_audio_from_youtube(URLS, title)

# 사용 예시 (Google Drive 경로 설정)
input_path = '2024-09-12 09-14-19.wav'
output_path = 'ste_filtered_result_5.wav'
fir_ste_rms_pipeline(input_path, output_path, lowcut=300, highcut=3000)
