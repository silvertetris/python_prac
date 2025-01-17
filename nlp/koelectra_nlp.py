from transformers import ElectraTokenizer
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
with open("data.json", "r", encoding="utf-8") as file:
    data = json.load(file)

print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize("오라클이 11% 급등... 오라클은 11% 급등을 보여줬는데요.")))