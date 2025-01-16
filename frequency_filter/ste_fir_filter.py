import numpy as np
import soundfile as sf
from scipy.signal import firwin, lfilter
import librosa
from yt_dlp import YoutubeDL


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


def fir_ste_rms_pipeline(input_file, output_file, lowcut=300, highcut=3000, ste_threshold=0.01, rms_threshold=0.01,
                         frame_length=1024, hop_length=512):
    # 1. 변환된 wav 파일 로드
    y, sr = librosa.load(input_file, sr=None)

    # 2. FIR 필터 적용
    filtered_audio = fir_bandpass_filter(y, lowcut, highcut, sr)

    # 3. Short-Time Energy (STE)와 RMS 계산
    ste = calculate_short_time_energy(filtered_audio, frame_length, hop_length)
    rms = calculate_rms(filtered_audio, frame_length, hop_length)

    # 4. STE와 RMS 기반 음성 활성 구간 탐지
    voice_segments = []
    start = None
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

    # 5. 최종 음성 데이터 조합
    final_data = np.concatenate([filtered_audio[start:end] for start, end in voice_segments])

    # 6. 결과 저장 (Google Drive 내 경로)
    sf.write(output_file, final_data, sr)

    # 로그 출력
    print(f"Processing complete. Output saved to {output_file}")
    print(f"Original duration: {len(y) / sr:.2f}s")
    print(f"Processed duration: {len(final_data) / sr:.2f}s")
    print(f"Number of voice segments: {len(voice_segments)}")


def extract_audio_from_youtube(URLS, title):
    #URLS = ['https://www.youtube.com/watch?v=jWOw_PXAzm0']
    ydl_opts = {  # YoutubeDL parameter 정보 => https://github.com/yt-dlp/yt-dlp/blob/master/yt_dlp/YoutubeDL.py#L183
        'format': 'bestaudio/best',
        'outtmpl': title,  # 영상 제목
        # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
        'postprocessors': [{  # Extract audio using ffmpeg
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }]
    }
    with YoutubeDL(ydl_opts) as ydl:
        error_code = ydl.download(URLS)

URLS='https://www.youtube.com/watch?v=jWOw_PXAzm0' # 여기다가 오디오 추출 유튜브 링크 넣어주세요
extract_audio_from_youtube(URLS, "yt_extracted_audio")

# 사용 예시 (Google Drive 경로 설정)
input_path = '2024-10-23 09-17-18.wav'  # 변환된 wav 파일 경로
output_path = 'ste_filtered_result.wav'
fir_ste_rms_pipeline(input_path, output_path, lowcut=300, highcut=3000, ste_threshold=0.006, rms_threshold=0.006)
