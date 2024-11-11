import numpy as np
import scipy.io.wavfile as wav
from scipy.fft import fft, fftfreq, ifft
import matplotlib.pyplot as plt

# 오디오 파일 로드 함수
def load_audio(filename):
    rate, data = wav.read(filename)
    if data.ndim > 1:  # 스테레오 오디오일 경우, 첫 번째 채널 사용
        data = data[:, 0]
    return rate, data

# FFT를 사용하여 사람 목소리 주파수 대역 필터링 함수 (300Hz ~ 3000Hz)
def voice_frequency_filter(data, rate, low_freq=300, high_freq=3000):
    N = len(data)
    yf = fft(data)
    xf = fftfreq(N, 1 / rate)

    # 사람 목소리 대역 내 주파수 필터링 (300Hz ~ 3000Hz)
    filtered_yf = np.where((xf >= low_freq) & (xf <= high_freq), yf, 0)

    # 역 FFT를 통해 필터링된 신호 복원
    filtered_data = ifft(filtered_yf).real
    return xf, np.abs(filtered_yf), filtered_data

# 필터링된 오디오 데이터를 파일로 저장하는 함수
def save_filtered_audio(filename, rate, data):
    # 데이터를 int16 형식으로 변환
    data = np.int16(data / np.max(np.abs(data)) * 32767)
    wav.write(filename, rate, data)

# 스펙트럼과 필터링된 신호 그래프 시각화 함수
def plot_results(xf, yf, original_data, filtered_data, rate):
    # 주파수 스펙트럼 시각화
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(xf, yf)
    plt.xlim(0, 5000)  # 사람 목소리 범위 표시 (300Hz ~ 3000Hz)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Frequency Spectrum with Voice Range Filtered")

    # 원본 신호와 필터링된 신호 비교
    time = np.linspace(0, len(original_data) / rate, num=len(original_data))
    plt.subplot(2, 1, 2)
    plt.plot(time, original_data, label="Original Audio")
    plt.plot(time, filtered_data, label="Filtered Voice", color="orange")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("Original vs Filtered Voice Signal")
    plt.tight_layout()
    plt.show()

# 메인 함수
def main():
    filename = "1015.wav"  # 분석할 오디오 파일 경로
    rate, data = load_audio(filename)

    # 사람 목소리 대역 필터링 (300Hz ~ 3000Hz)
    xf, yf, filtered_data = voice_frequency_filter(data, rate)

    # 필터링된 오디오 파일 저장
    save_filtered_audio("filtered_voice.wav", rate, filtered_data)
    print("Filtered audio saved as 'filtered_voice.wav'")

    # 결과 시각화
    plot_results(xf, yf, data, filtered_data, rate)

if __name__ == "__main__":
    main()
