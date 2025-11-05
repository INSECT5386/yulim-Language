!pip install gensim sentencepiece scikit-learn tqdm

import json
import os
import sentencepiece as spm
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from tqdm import tqdm
import random

# ⬇️ 파일 다운로드 함수
def download_file(url, save_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✅ 파일 저장됨: {save_path}")

# ✅ 데이터 다운로드
download_file('https://huggingface.co/datasets/Yuchan5386/SFT/resolve/main/data_shuffled_1.jsonl?download=true', 'faq_dataset.jsonl')
download_file('https://huggingface.co/Yuchan5386/inlam-100m/resolve/main/ko_unigram.model?download=true', 'spm.model')

# ✅ SentencePiece 모델 로드 (이미 학습된 모델 사용)
sp = spm.SentencePieceProcessor()
sp.load("spm.model")

# ✅ Word2Vec 모델 초기화
model = Word2Vec(vector_size=200, window=7, min_count=2, workers=4)

def generate_sentences(file_path, sample_rate=0.05):
    """JSONL을 스트리밍으로 읽고, 일정 확률로 샘플링해서 문장 토큰 반환"""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if random.random() > sample_rate:
                continue
            try:
                item = json.loads(line)
                conv = item.get("conversations", [])
                for turn in conv:
                    if turn["from"] == "human":
                        tokens = sp.encode_as_pieces(turn["value"])
                        yield tokens
            except:
                continue

# ✅ Word2Vec 점진 학습 준비 (리스트로 데이터를 미리 로드)
print("📖 Word2Vec 학습용 데이터 로딩 중...")
sentences = list(generate_sentences("faq_dataset.jsonl", sample_rate=0.02))  # 전체 데이터를 리스트로 저장

# ✅ 어휘 빌드 (어휘 사전 생성)
print("📖 어휘 빌드 중...")
model.build_vocab(sentences, progress_per=100000)
print("✅ 어휘 수:", len(model.wv))

# ✅ 전체 코퍼스에서 점진 학습
print("⚙️ Word2Vec 학습 중...")
model.train(sentences, total_examples=model.corpus_count, epochs=3)
model.save("respiso.model")
print("✅ Word2Vec 학습 완료")

# ✅ FAQ 샘플 데이터 로딩 (limit 제거)
def load_faq_subset(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            conv = item.get("conversations", [])
            if len(conv) >= 2:
                q = [c["value"] for c in conv if c["from"] == "human"]
                a = [c["value"] for c in conv if c["from"] == "gpt"]
                if q and a:
                    data.append({"question": q[0], "answer": a[0]})
    return data

faq_data = load_faq_subset("faq_dataset.jsonl")
print(f"✅ FAQ 샘플 {len(faq_data)}개 로드 완료")

# ✅ 문장 벡터 계산 함수
def sentence_vector(sentence, model, sp):
    tokens = sp.encode_as_pieces(sentence)
    vecs = [model.wv[token] for token in tokens if token in model.wv]
    if len(vecs) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vecs, axis=0)

faq_vectors = [sentence_vector(faq["question"], model, sp) for faq in faq_data]

# ✅ 챗봇 응답 함수
from sklearn.metrics.pairwise import cosine_similarity

def chatbot_response(user_input):
    user_vec = sentence_vector(user_input, model, sp).reshape(1, -1)
    sims = cosine_similarity(user_vec, faq_vectors)
    idx = sims.argmax()
    return faq_data[idx]["answer"]

# ✅ 챗봇 시작
print("챗봇 시작! (종료하려면 '종료', 'exit', 'quit')")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["종료", "exit", "quit"]:
        print("챗봇 종료!")
        break
    print("Bot:", chatbot_response(user_input))
