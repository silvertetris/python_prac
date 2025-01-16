from sklearn.neighbors import NearestNeighbors
import numpy as np
from transformers import ElectraTokenizer
import json

tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

with open("data.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# timestamp 제외하고 텍스트 뽑아오기
all_text = " ".join(entry.split(']: ')[1].strip() for entry in data if ']: ' in entry and entry.split(']: ')[1].strip())
tokens = tokenizer.tokenize(all_text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(token_ids)

def knn_repeating_subsequences(lst, k, n_neighbors=2, distance_threshold=5.0):
    subsequences = [lst[i:i + k] for i in range(len(lst) - k + 1)] #기준 subsequence 길이 만큼 iterate
    subsequences_array = np.array(subsequences)

    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean') #apply KNN
    knn.fit(subsequences_array)

    distances, indices = knn.kneighbors(subsequences_array)

    repeating_patterns = []

    for idx, (dist, neighbor_indices) in enumerate(zip(distances, indices)): #임계값, 거리에 따라 repeating_patterns에 append
        pattern = tuple(subsequences[idx])
        for j, i in enumerate(neighbor_indices):
            if i != idx and dist[j] <= distance_threshold:
                if pattern not in repeating_patterns:
                    repeating_patterns.append(pattern)

    return repeating_patterns


k = 10  # length of subsequence
distance_threshold = 5  # threshold
repeating_patterns = knn_repeating_subsequences(token_ids, k, n_neighbors=2, distance_threshold=distance_threshold)

# 결과 출력
for pattern in repeating_patterns:
    print(f"패턴 (토큰 ID): {pattern}")
    print(f"패턴 (디코딩): {tokenizer.decode(pattern)}")
