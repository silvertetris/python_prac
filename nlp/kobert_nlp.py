import json
from kobert_tokenizer import KoBERTTokenizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Load KoBERT tokenizer
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

with open("data.json", "r", encoding="utf-8") as file:
    data = json.load(file)

#timestamp 제외하고 텍스트 뽑아오기
all_text = " ".join(entry.split(']: ')[1].strip() for entry in data if ']: ' in entry and entry.split(']: ')[1].strip())

#토큰으로 전처리
tokens = tokenizer.tokenize(all_text)

print(tokens)
#Find repetitive words
repetitive_words = []
for i in range(len(tokens) - 1):
    if tokens[i] == tokens[i + 1]:
        repetitive_words.append(tokens[i])

#scikitlearn
vectorizer = CountVectorizer()
token_counts = vectorizer.fit_transform([" ".join(tokens)])
token_freq = dict(zip(vectorizer.get_feature_names_out(), np.array(token_counts.sum(axis=0)).flatten()))

print("Repetitive Words (Consecutive):", set(repetitive_words))
print("Token Frequencies:")
for token, freq in sorted(token_freq.items(), key=lambda x: x[1], reverse=True):
    print(f"{token}: {freq}")


#문제점:
# kobert가 토큰화를 할 때, 한 단어에 대해 제대로 인식하지 못해 토큰화가 제대로 진행되지 않을 때가 있음
# consecutive 단어들을 찾아낼 때, 토큰화가 제대로 되지 않아 한글자 반복에 대한 토큰만 찾아냄
