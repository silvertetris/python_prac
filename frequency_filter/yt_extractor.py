from yt_dlp import YoutubeDL

URLS = ['https://www.youtube.com/watch?v=jWOw_PXAzm0'] #여기다가 오디오 추출 유튜브 링크 넣어주세요

ydl_opts = {  # YoutubeDL parameter 정보 => https://github.com/yt-dlp/yt-dlp/blob/master/yt_dlp/YoutubeDL.py#L183
    'format': 'bestaudio/best',
    'outtmpl': "yt_extracted_audio",  # 영상 제목
    # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
    'postprocessors': [{  # Extract audio using ffmpeg
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
    }]
}

with YoutubeDL(ydl_opts) as ydl:
    error_code = ydl.download(URLS)
