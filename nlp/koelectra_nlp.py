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

#timestamp 제외하고 텍스트 뽑아오기
all_text = " ".join(entry.split(']: ')[1].strip() for entry in data if ']: ' in entry and entry.split(']: ')[1].strip())
tokens=tokenizer.tokenize(all_text)
token_ids= tokenizer.convert_tokens_to_ids(tokens)

# Label repetitive subsequences
labels = []
subsequence_length = 15  # Length of subsequence to check for repetition

#여기 for문을 다시 짜야함 (2중 for문으로)
for i in range(len(token_ids)):
    if i >= subsequence_length and token_ids[i-subsequence_length:i] == token_ids[i:i+subsequence_length]:
        labels.append(1)  # Subsequence repetition detected
    else:
        labels.append(0)

# Prepare sequences for LSTM
sequence_length = 15
X, y = [], []
for i in range(len(token_ids) - sequence_length):
    X.append(token_ids[i:i + sequence_length])
    y.append(labels[i + sequence_length - 1])

X = np.array(X)
y = np.array(y)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential([
    Embedding(input_dim=tokenizer.vocab_size, output_dim=50, input_length=sequence_length),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=32)

# Predict repetitive subsequences
test_text = all_text
test_tokens = tokenizer.tokenize(test_text)
test_token_ids = tokenizer.convert_tokens_to_ids(test_tokens)

# Prepare test sequences
test_X = []
for i in range(len(test_token_ids) - sequence_length):
    test_X.append(test_token_ids[i:i + sequence_length])

test_X = np.array(test_X)

# Predict
predictions = model.predict(test_X)

# Interpret predictions
decoded_tokens = test_tokens[sequence_length:]
predicted_labels = (predictions > 0.5).astype(int).flatten()

print("Tokenized Text:", test_tokens)
for token, label in zip(decoded_tokens, predicted_labels):
    print(f"Token: {token}, Repetitive: {bool(label)}")