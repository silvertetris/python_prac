import ffmpeg
input_path = '2024-09-12 09-14-19.mkv'
output_path = '2024-09-12 09-14-19.wav'

# 오류 처리와 함께 ffmpeg를 사용하여 변환
try:
    (
        ffmpeg
        .input(input_path)
        .output(output_path, format='wav', acodec='pcm_s16le', ar=44100, ac=2)
        .run(capture_stdout=True, capture_stderr=True) # 디버깅을 위해 출력을 캡처합니다.
    )
    print("변환 완료, 결과 파일:", output_path)
except ffmpeg.Error as e:
    print(f"오류 발생: {e.stderr.decode()}") # 진단을 위해 오류를 출력합니다.