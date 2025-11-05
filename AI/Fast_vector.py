!pip install faiss-cpu tqdm gensim sentencepiece scikit-learn

import json
import numpy as np
import faiss
from tqdm import tqdm
from gensim.models import Word2Vec
import sentencepiece as spm
import os

# ✅ 모델 로드
sp = spm.SentencePieceProcessor()
sp.load("spm.model")
model = Word2Vec.load("respiso.model")

# ✅ 문장 → 벡터 변환
def sentence_vector(sentence, model, sp):
    tokens = sp.encode_as_pieces(sentence)
    vecs = [model.wv[tok] for tok in tokens if tok in model.wv]
    if not vecs:
        return np.zeros(model.vector_size, dtype="float32")
    return np.mean(vecs, axis=0).astype("float32")

# ✅ FAQ JSONL 로드 (answers만 메모리에 보관)
def load_answers(jsonl_path, limit=None):
    answers = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            item = json.loads(line)
            conv = item.get("conversations", [])
            if len(conv) >= 2:
                a = [c["value"] for c in conv if c["from"] == "gpt"]
                if a:
                    answers.append(a[0])
    print(f"✅ {len(answers)}개의 답변 로드 완료!")
    return answers

# ✅ 벡터 memmap 로드
def load_vectors(memmap_path, dim, limit=None):
    vecs = np.memmap(memmap_path, dtype="float32", mode="r")
    total = vecs.size // dim
    if limit:
        total = min(limit, total)
    return vecs[:total * dim].reshape(total, dim)

# ✅ FAISS 인덱스 생성
def build_faiss_index(vectors):
    index = faiss.IndexFlatL2(vectors.shape[1])  # L2 거리 기반
    index.add(vectors)
    print(f"✅ {vectors.shape[0]}개 벡터 인덱스 구축 완료!")
    return index

# ✅ 챗봇 응답
def chatbot_response(user_input, index, vectors, answers, model, sp):
    user_vec = sentence_vector(user_input, model, sp).reshape(1, -1)
    D, I = index.search(user_vec, 1)  # top-1 검색
    sim = 1 / (1 + D[0][0])  # 거리 → 유사도 변환
    return answers[I[0][0]], sim

# ✅ 실행
LIMIT = 100000
VEC_DIM = model.vector_size

print(f"📦 {LIMIT}개 데이터 로드 및 인덱스 구성 중...")

faq_vectors = load_vectors("faq_vectors_100k.memmap", VEC_DIM, limit=LIMIT)
answers = load_answers("faq_dataset.jsonl", limit=LIMIT)
index = build_faiss_index(faq_vectors)

print("🤖 Respiso FAISS 챗봇 시작! (종료하려면 '종료', 'exit', 'quit')")
while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["종료", "exit", "quit"]:
        print("👋 종료!")
        break
    answer, sim = chatbot_response(user_input, index, faq_vectors, answers, model, sp)
    print(f"Bot ({sim:.3f}): {answer}")
