import whisper

model = whisper.load_model("turbo")
result = model.transcribe(audio="ste_filtered_result_4.wav", word_timestamps=True)
print(result["segments"])