import whisper
import librosa
import numpy as np
import json

# ======================================
# 1. Whisper Large-V2 모델 로드
#    (GPU 사용 시 device="cuda" 권장)
# ======================================
model = whisper.load_model("turbo")
# 만약 GPU가 없다면 아래처럼 CPU 사용:
# model = whisper.load_model("large-v2", device="cpu")

# ======================================
# 2. 오디오 파일 로드
#    - Spleeter 등으로 분리된 voice 파일 사용
#    - sr=16000으로 리샘플링 (Whisper 기본 샘플레이트)
# ======================================
voice_file ="../whisper/ste_filtered_result_4.wav"
audio, sr = librosa.load(voice_file, sr=16000)

# ======================================
# 3. Whisper 모델로 텍스트 + 타임스탬프 추출
#    - 정확도를 높이기 위해 beam_size, best_of 등 추가
#    - temperature=0.0로 탐색 폭 줄여 결정적 결과 유도
# ======================================
result = model.transcribe(
    voice_file,            # 파일 경로 (또는 audio 배열도 가능)
    language="ko",         # 한국어 지정
    task="transcribe",     # 음성 → 텍스트 변환
    word_timestamps=True,  # 단어별(혹은 토큰별) 타임스탬프
    beam_size=15,          # 빔서치 크기
    best_of=5,            # 후보 중 최적 결과 선택
    temperature=0.0,       # 탐색 온도 (0에 가까울수록 결정적 결과)
    # fp16=False           # CPU 사용 시 fp16=False 권장
    # patience=0.1         # 필요 시 beam search 내 인내치 설정
)
print(result["segments"])