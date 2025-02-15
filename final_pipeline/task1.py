import os
import tempfile

import ffmpeg
from spleeter.separator import Separator

# 초기 파일 경로
input_path = '../frequency_filter/2024-10-23 09-17-18.mkv'

'''
1. 파일 경로를 받음 (영상)
2. 영상을 오디오로 변환하고 파일로 write 함 (임시 파일 , 삭제할 예정)
3. 파일을 가져와서 spleeter로 넣고 임시 파일 삭제
'''


def video_to_temp_wav(video_path):
    try:
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio_path = temp_audio.name
        temp_audio.close()

        ffmpeg.input(video_path).output(temp_audio_path, format='wav', acodec='pcm_s16le', ar=44100, ac=2).run(
            quiet=True, capture_stdout=True, capture_stderr=True)
        print("변환 완료, 결과 파일:", temp_audio_path)
        return temp_audio_path
    except ffmpeg.Error as e:
        print(f"오류 발생: {e.stderr.decode()}")
        return None


def separate_audio_from_temp_wav(temp_audio_path):
    try:
        output_dir = tempfile.mkdtemp()

        separator = Separator('spleeter:2stems')
        separator.separate_to_file(temp_audio_path, output_dir)
        vocals_path = os.path.join(output_dir, "your_audio_file/vocals.wav")
        # 임시파일 삭제
        os.remove(temp_audio_path)
        return vocals_path
    except Exception as e:
        print(f"Error during separation: {e}")
        return None


temp_wav_path = video_to_temp_wav(input_path)

if temp_wav_path:
    vocals_path = separate_audio_from_temp_wav(temp_wav_path)
    if vocals_path:
        print(f"목소리 파일 저장 위치: {vocals_path}")
    else:
        print("경로 없음")
else:
    print("Failed to extract audio.")
