import librosa
import numpy as np
from scipy.signal import firwin, lfilter
import matplotlib.pyplot as plt
import ffmpeg

input_file = '1015.mp4'
output_file = '1015.wav'

#ffmpeg로 mkv를 wav로 변경
#ffmpeg.input(input_file).output(output_file, format='wav', acodec='pcm_s16le', ar=44100, ac=2).run()
y, sr = librosa.load(output_file, sr=None) #time series, fr(샘플링 속도 (rate) ) None=native sampling rate

# FIR 통과 주파수 설정 (nyquist를 통해 normalize)
def fir_bandpass(lowcut, highcut, sr, numtaps):
    half = 0.5 * sr #normalize 하기 위한 nyquist
    low = lowcut / half
    high = highcut / half
    taps = firwin(numtaps, [low, high], pass_zero=False)
    return taps

#FIR 필터링
def fir_bandpass_filter(data, lowcut, highcut, sr, numtaps):
    taps = fir_bandpass(lowcut, highcut, sr, numtaps)
    filtered_data = lfilter(taps, 1.0, data) #필터에 대역 통과 (분자, 분모, input array (coefficient) )
    return filtered_data

lowcut = 300.0
highcut = 3000.0
numtaps = 300

#최종 필터링된 오디오
filtered_audio = fir_bandpass_filter(y, lowcut, highcut, sr, numtaps=numtaps) #numtaps = filter 길이

#총 시간 길이 / 속도 -> 시간
time = np.linspace(0, len(y) / sr, len(y))


#plot 설정
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.xlabel("t(sec)")
plt.ylabel("amplitude")
plt.plot(time, y, label='Original Signal')
plt.title("Original Signal")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time, filtered_audio, label='Filtered Signal')
plt.xlabel("t(sec)")
plt.ylabel("amplitude")
plt.title("Filtered Signal")
plt.legend()
plt.grid(True)


plt.show()