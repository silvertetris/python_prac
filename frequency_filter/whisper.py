import whisper

model = whisper.load_model("turbo")
result = model.transcribe('ste_filtered_result_4.wav')
text = result["text"]
segments = result["segments"]
print(segments[0])