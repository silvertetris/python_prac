import re
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from konlpy.tag import Kkma
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from KoBERTScore import BERTScore

# ======================================
# 1. BERT 모델 및 토크나이저 로드
# ======================================
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# GPU 사용 가능하면 모델을 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

bert_scorer = BERTScore("beomi/kcbert-base", best_layer=4)  # Use the same model for consistency

# ======================================
# 2. JSON 데이터를 파싱하여 타임스탬프와 텍스트 추출
# ======================================
def parse_segments(segments):
    """
    JSON 파일의 문자열 데이터를 파싱하여 타임스탬프와 텍스트를 분리
    """
    parsed_segments = []
    for segment in segments:
        match = re.match(r"\[(\d+\.?\d*)s ~ (\d+\.?\d*)s\]:\s*(.*)", segment)
        if match:
            start = float(match.group(1))
            end = float(match.group(2))
            text = match.group(3).strip()
            if text:  # 빈 텍스트 제거
                parsed_segments.append({"start": start, "end": end, "text": text})
    return parsed_segments

# ======================================
# 3. 문장 임베딩 계산 함수
# ======================================
def get_sentence_embedding(sentence):
    """
    BERT 모델을 사용하여 문장의 임베딩 벡터를 추출
    """
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return cls_embedding

# ======================================
# 4. 형태소 분석 및 벡터 계산
# ======================================
kkma = Kkma()

def extract_morphemes(sentence):
    """
    Konlpy의 Kkma를 사용하여 형태소를 추출
    """
    return " ".join([word for word, tag in kkma.pos(sentence)])

def calculate_euclidean_distance(sentences):
    """
    형태소 기반 문장 유사도를 계산
    """
    morphemes = [extract_morphemes(sent["text"]) for sent in sentences]
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(morphemes).toarray()
    distances = euclidean_distances(vectors, vectors)
    return distances

# ======================================
# 5. 버벅거림 제거 함수 (BERTScore 추가)
# ======================================
def remove_repetitive_sentences(segments, similarity_threshold=0.9):
    """
    코사인 유사도, TF-IDF, 형태소 기반 벡터 거리 계산, BERTScore를 활용하여 중복 및 버벅거림 제거
    """
    filtered_segments = []
    prev_embedding = None

    tfidf_texts = [seg["text"] for seg in segments if seg["text"].strip()]
    tfidf_vectorizer = TfidfVectorizer().fit_transform(tfidf_texts)
    cosine_similarities = sklearn_cosine_similarity(tfidf_vectorizer, tfidf_vectorizer)
    morpheme_distances = calculate_euclidean_distance(segments)

    # Precompute BERTScore for all pairs
    all_texts = [seg["text"] for seg in segments]
    bert_scores = bert_scorer.score(all_texts, all_texts, batch_size=128)  # P, R, F1 scores

    for i, segment in enumerate(segments):
        text = segment["text"]
        embedding = get_sentence_embedding(text)

        # 코사인 유사도 체크
        is_similar_cosine = False
        if prev_embedding is not None:
            cosine_similarity_score = np.dot(prev_embedding, embedding) / (np.linalg.norm(prev_embedding) * np.linalg.norm(embedding))
            is_similar_cosine = cosine_similarity_score >= similarity_threshold

        # TF-IDF 유사도 체크
        is_similar_tfidf = any(cosine_similarities[i, j] >= similarity_threshold for j in range(i))

        # 형태소 기반 벡터 거리 계산
        is_close_euclidean = any(morpheme_distances[i][j] < 1.0 for j in range(i))

        # BERTScore 체크
        is_similar_bertscore = any(bert_scores[2][i, j] >= similarity_threshold for j in range(i))  # F1 score

        # 비교 로직
        if not (is_similar_cosine or is_similar_tfidf or is_close_euclidean or is_similar_bertscore):
            # 이전 문장과 충분히 다르면 추가
            if len(filtered_segments) == 0 or text != filtered_segments[-1]["text"]:
                filtered_segments.append(segment)
                prev_embedding = embedding

    return filtered_segments

# ======================================
# 6. JSON 파일 불러오기
# ======================================
input_file_path = "/nlp/data.json"

with open(input_file_path, "r", encoding="utf-8") as f:
    raw_segments = json.load(f)

# JSON 데이터 파싱
parsed_segments = parse_segments(raw_segments)

# ======================================
# 7. 중복 제거 수행
# ======================================
filtered_segments = remove_repetitive_sentences(parsed_segments)

# ======================================
# 8. 결과 출력 및 저장
# ======================================
def format_segments(segments):
    """
    필터링된 데이터를 문자열 형식으로 반환
    """
    return "\n".join([f"[{seg['start']:.2f}s ~ {seg['end']:.2f}s]: {seg['text']}" for seg in segments])

print("=== 중복 제거 후 결과 ===")
print(format_segments(filtered_segments))