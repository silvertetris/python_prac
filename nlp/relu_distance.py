import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Prepare Dataset
def compute_distances(tokens):
    """
    Compute the distance between repetitive words in a tokenized sequence.
    Returns a list of distances where -1 indicates no repetition.
    """
    distances = [-1] * len(tokens)
    last_seen = {}
    for i, token in enumerate(tokens):
        if token in last_seen:
            distances[last_seen[token]] = i - last_seen[token]
        last_seen[token] = i
    return distances

# Example data
data = [
    "apple banana apple orange banana",
    "grape apple orange apple grape grape",
    "banana orange orange apple"
]

# Tokenize the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)

# Compute distances for each sequence
distances = [compute_distances(seq) for seq in sequences]

# Pad sequences and distances
max_sequence_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
padded_distances = pad_sequences(distances, maxlen=max_sequence_length, padding='post', value=-1)

# Convert distances to a NumPy array for training
padded_distances = np.array(padded_distances)

# 2. Build the Model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length),
    LSTM(128, return_sequences=True),
    TimeDistributed(Dense(1, activation='relu'))  # Predict distance for each word
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# 3. Train the Model
# Split into training and validation data
split_idx = int(len(padded_sequences) * 0.8)
x_train, x_val = padded_sequences[:split_idx], padded_sequences[split_idx:]
y_train, y_val = padded_distances[:split_idx], padded_distances[split_idx:]

model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32)

# 4. Predict on new data
def predict_distances(model, tokenizer, text, max_sequence_length):
    """
    Predict distances for a new sequence of text using the trained model.
    """
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=max_sequence_length, padding='post')
    predictions = model.predict(padded_seq)
    return predictions[0]

# Example prediction
new_text = "apple orange apple banana grape"
predicted_distances = predict_distances(model, tokenizer, new_text, max_sequence_length)
print("Input:", new_text)
print("Predicted distances:", np.round(predicted_distances).astype(int))
