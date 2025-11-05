import json
import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec
import sentencepiece as spm
from sklearn.metrics.pairwise import cosine_similarity
import os

# 모델 로드
sp = spm.SentencePieceProcessor()
sp.load("spm.model")
model = Word2Vec.load("respiso.model")

# 문장 → 벡터
def sentence_vector(sentence, model, sp):
    tokens = sp.encode_as_pieces(sentence)
    vecs = [model.wv[tok] for tok in tokens if tok in model.wv]
    if not vecs:
        return np.zeros(model.vector_size)
    return np.mean(vecs, axis=0)

# FAQ 데이터를 제너레이터로 읽기
def stream_faq(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            conv = item.get("conversations", [])
            if len(conv) >= 2:
                q = [c["value"] for c in conv if c["from"] == "human"]
                a = [c["value"] for c in conv if c["from"] == "gpt"]
                if q and a:
                    yield {"question": q[0], "answer": a[0]}

# 전체 라인 수 세기 (memmap 크기 설정용)
def count_faq_lines(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

# 스트리밍 저장
def build_memmap_vectors(jsonl_path, output_path, model, sp):
    total = count_faq_lines(jsonl_path)
    dim = model.vector_size

    print(f"📦 총 {total}개의 FAQ 문장 처리 예정 (벡터 차원: {dim})")

    vectors = np.memmap(output_path, dtype="float32", mode="w+", shape=(total, dim))

    for i, faq in enumerate(tqdm(stream_faq(jsonl_path), total=total, desc="벡터화 중")):
        vec = sentence_vector(faq["question"], model, sp)
        vectors[i] = vec

    del vectors  # flush to disk
    print(f"✅ {output_path} 저장 완료!")

# 챗봇 응답 함수 (memmap 기반)
def chatbot_response(user_input, model, sp, jsonl_path, vec_path):
    user_vec = sentence_vector(user_input, model, sp).reshape(1, -1)
    vectors = np.memmap(vec_path, dtype="float32", mode="r").reshape(-1, model.vector_size)
    best_sim = -1
    best_answer = None

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            conv = data.get("conversations", [])
            if len(conv) < 2:
                continue
            a = [c["value"] for c in conv if c["from"] == "gpt"]
            if not a:
                continue

            sim = cosine_similarity(user_vec, vectors[i].reshape(1, -1))[0, 0]
            if sim > best_sim:
                best_sim = sim
                best_answer = a[0]

    return best_answer, best_sim

# 메인
if not os.path.exists("faq_vectors.memmap"):
    build_memmap_vectors("faq_dataset.jsonl", "faq_vectors.memmap", model, sp)

print("🤖 Respiso 완전 스트리밍 챗봇 시작! (종료하려면 '종료')")
while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["종료", "exit", "quit"]:
        print("👋 종료!")
        break
    answer, sim = chatbot_response(user_input, model, sp, "faq_dataset.jsonl", "faq_vectors.memmap")
    print(f"Bot ({sim:.3f}): {answer}")
