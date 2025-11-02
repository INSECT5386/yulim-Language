# seprod_train_multiturn.py
# 실행 환경: Python 3.8+, TensorFlow 2.x, sentencepiece
# 준비물: output.jsonl, ko_unigram.model

import json
import random
import sentencepiece as spm
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import requests
# -----------------------------
# 설정
# -----------------------------
DATA_PATH = "output.jsonl"
SP_MODEL = "ko_unigram.model"
MAX_ENC_LEN = 128
MAX_DEC_LEN = 128
BATCH_SIZE = 64
D_MODEL = 256
EPOCHS = 1
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✅ 파일 저장됨: {save_path}")

# ⬇️ 데이터와 토크나이저 다운로드
download_file('https://huggingface.co/datasets/Yuchan5386/TinyInst/resolve/main/output.jsonl?download=true', DATA_PATH)
download_file('https://huggingface.co/datasets/Yuchan5386/TinyInst/resolve/main/ko_unigram.model?download=true', SP_MODEL)

# -----------------------------
# JSONL -> (human,gpt) pair 추출
# -----------------------------
def load_pairs_from_jsonl(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "conversations" in obj and isinstance(obj["conversations"], list):
                convs = obj["conversations"]
                for i in range(0, max(0, len(convs)-1), 2):
                    a, b = convs[i], convs[i+1]
                    if a.get("from") == "human" and b.get("from") in ("gpt", "assistant"):
                        q = a.get("value", "").strip().replace("\n", " ")
                        r = b.get("value", "").strip().replace("\n", " ")
                        if q and r:
                            pairs.append((q, r))
    return pairs

pairs = load_pairs_from_jsonl(DATA_PATH)
print(f"Loaded pairs: {len(pairs)}")
if len(pairs) == 0:
    raise SystemExit("output.jsonl에서 (human,gpt) 페어를 찾지 못했음.")

# -----------------------------
# SentencePiece 로드
# -----------------------------
sp = spm.SentencePieceProcessor()
sp.load(SP_MODEL)

def safe_id(piece, default=0):
    pid = sp.piece_to_id(piece)
    return pid if pid != -1 else default

PAD_ID = safe_id("<pad>", 0)
START_ID = safe_id("<start>", None)
SEP_ID = safe_id("<sep>", None)
END_ID = safe_id("<end>", None)

if START_ID is None or SEP_ID is None or END_ID is None:
    raise SystemExit("토크나이저에 <start>, <sep>, <end> 등 스페셜 토큰 확인 필요.")

VOCAB_SIZE = sp.get_piece_size()
print(f"Vocab size: {VOCAB_SIZE}, PAD_ID={PAD_ID}, START={START_ID}, SEP={SEP_ID}, END={END_ID}")

# -----------------------------
# 데이터 전처리
# -----------------------------
def make_example(q, r, max_enc=MAX_ENC_LEN, max_dec=MAX_DEC_LEN):
    q = q.replace("\n", " ").strip()
    r = r.replace("\n", " ").strip()

    enc_tokens = sp.encode(q + " <sep>")[:max_enc]
    enc_p = enc_tokens + [PAD_ID] * (max_enc - len(enc_tokens))

    r_ids = sp.encode(r)[:(max_dec-1)]
    dec_in = [START_ID] + r_ids
    dec_in = dec_in[:max_dec]
    dec_in_p = dec_in + [PAD_ID] * (max_dec - len(dec_in))

    tgt = r_ids + [END_ID]
    tgt = tgt[:max_dec]
    tgt_p = tgt + [PAD_ID] * (max_dec - len(tgt))
    
    return np.array(enc_p, dtype=np.int32), np.array(dec_in_p, dtype=np.int32), np.array(tgt_p, dtype=np.int32)

examples = [make_example(q, r) for q, r in pairs]
encs = np.stack([e[0] for e in examples])
decs = np.stack([e[1] for e in examples])
tgts = np.stack([e[2] for e in examples])
print("Example shapes:", encs.shape, decs.shape, tgts.shape)

dataset = tf.data.Dataset.from_tensor_slices(((encs, decs), tgts))
dataset = dataset.shuffle(buffer_size=len(encs), seed=SEED).batch(BATCH_SIZE).prefetch(AUTOTUNE)

# -----------------------------
# 모델 구성
# -----------------------------
class LearnablePositionalEmbedding(layers.Layer):
    def __init__(self, max_length, d_model):
        super().__init__()
        self.pos_emb = self.add_weight(
            name="pos_emb", shape=(max_length, d_model),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02), trainable=True)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_emb[tf.newaxis, :seq_len, :]

class Adapter(layers.Layer):
    def __init__(self, d_model, bottleneck=64):
        super().__init__()
        self.w_down = layers.Dense(bottleneck, activation='silu')
        self.w_up = layers.Dense(d_model)
        self.norm = layers.LayerNormalization(epsilon=1e-5, dtype='float32')

    def call(self, x):
        residual = x
        x = self.w_down(x)
        x = self.w_up(x)
        return self.norm(x + residual)

class Block(layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.norm = layers.LayerNormalization(epsilon=1e-5, dtype='float32')
        self.conv = layers.Conv1D(filters=d_model, kernel_size=3, padding='same', activation='relu')
        self.gap = layers.GlobalAveragePooling1D()
        self.w2 = layers.Dense(d_model, activation='silu')
        self.w3 = layers.Dense(d_model)
        self.adapter = Adapter(d_model)
        self.out_norm = layers.LayerNormalization(epsilon=1e-5, dtype='float32')

    def call(self, x):
        x_norm = self.norm(x)
        x_conv = self.conv(x_norm)
        pooled = self.gap(x_conv)
        a = self.w3(pooled) * self.w2(pooled)
        a = self.adapter(a)
        a = self.out_norm(a)
        return a

class D(layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.norm = layers.LayerNormalization(epsilon=1e-5, dtype='float32')
        self.attn = tf.keras.layers.Attention(use_scale=True)
        self.adapter = Adapter(d_model)

    def call(self, x):
        x_norm = self.norm(x)
        v = self.adapter(x_norm)
        out = self.attn([x_norm, x_norm, v], use_causal_mask=True)
        return out

class CrossBlock(layers.Layer):
    def __init__(self, dim, dropout_rate=0.1):
        super().__init__()
        self.dense1 = layers.Dense(dim)
        self.dense2 = layers.Dense(dim)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout_rate)
        self.dim = dim

    def call(self, x, z, training=None):
        seq_len = tf.shape(x)[1]
        z_exp = tf.expand_dims(z, axis=1)
        z_rep = tf.tile(z_exp, [1, seq_len, 1])
        gate = self.dense2(z_rep)
        z_th = tf.nn.sigmoid(x * gate) * gate
        z_th = self.norm1(z_th)
        x = self.dense1(z_th)
        x = self.norm2(x)
        if self.dim % 2 != 0:
            x = layers.Dense(self.dim + 1)(x)
        f, ft = tf.split(x, num_or_size_splits=2, axis=-1)
        f = tf.nn.silu(f)
        out = f * ft
        return out

encoder_input = Input(shape=(MAX_ENC_LEN,), dtype=tf.int32, name='encoder_input')
decoder_input = Input(shape=(MAX_DEC_LEN,), dtype=tf.int32, name='decoder_input')

enc_emb = layers.Embedding(input_dim=VOCAB_SIZE, output_dim=D_MODEL)(encoder_input)
enc_pos = LearnablePositionalEmbedding(MAX_ENC_LEN, D_MODEL)(enc_emb)
context_vector = Block(D_MODEL)(enc_pos)

dec_emb = layers.Embedding(input_dim=VOCAB_SIZE, output_dim=D_MODEL)(decoder_input)
dec_pos = LearnablePositionalEmbedding(MAX_DEC_LEN, D_MODEL)(dec_emb)
dec_post = D(D_MODEL)(dec_pos)

cross_out = CrossBlock(D_MODEL)(dec_post, context_vector)
logits = layers.Dense(VOCAB_SIZE)(cross_out)

model = Model(inputs=[encoder_input, decoder_input], outputs=logits)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
def masked_loss(y_true, y_pred):
    loss = loss_fn(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, PAD_ID), tf.float32)
    loss = loss * mask
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-8)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=masked_loss)
model.summary()

# -----------------------------
# 학습
# -----------------------------
try:
    model.fit(dataset, epochs=EPOCHS)
    model.save('tf_seprod_multiturn.h5')
    print("모델 저장 완료")
except Exception as e:
    print("학습 중 오류:", str(e))

# -----------------------------
# 멀티턴 생성 함수
# -----------------------------
def merge_history(history, sep_token="<sep>"):
    merged = ""
    for i, turn in enumerate(history):
        prefix = "User: " if i % 2 == 0 else "Assistant: "
        merged += prefix + turn + " "
    return merged.strip()

def generate_multiturn(model, sp, history, max_dec_len=MAX_DEC_LEN, temperature=0.7):
    merged_input = merge_history(history)
    enc_ids = sp.encode(merged_input + " <sep>")[:MAX_ENC_LEN]
    enc_ids += [PAD_ID] * (MAX_ENC_LEN - len(enc_ids))
    enc_tensor = tf.constant([enc_ids], dtype=tf.int32)
    
    dec_sequence = [START_ID]
    for step in range(max_dec_len):
        dec_input = dec_sequence + [PAD_ID] * (max_dec_len - len(dec_sequence))
        dec_tensor = tf.constant([dec_input], dtype=tf.int32)
        
        logits = model([enc_tensor, dec_tensor], training=False)
        step_logits = logits[:, len(dec_sequence)-1, :]
        
        if temperature == 0.0:
            next_id = int(tf.argmax(step_logits, axis=-1).numpy()[0])
        else:
            scaled = step_logits / max(1e-8, temperature)
            sampled = tf.random.categorical(scaled, num_samples=1)
            next_id = int(tf.squeeze(sampled, axis=-1).numpy()[0])
        
        if next_id == END_ID:
            break
        dec_sequence.append(next_id)
    
    return sp.decode(dec_sequence[1:])

# -----------------------------
# 간단 테스트
# -----------------------------
history = ["안녕하세요! 저는 호주 여행을 가고 싶어요."]
resp = generate_multiturn(model, sp, history, temperature=0.7)
print("AI:", resp)

# 다음 턴 추가 예시
history.append(resp)
history.append("호주에서 어떤 스포츠 경기를 볼 수 있을까요?")
resp2 = generate_multiturn(model, sp, history, temperature=0.7)
print("AI:", resp2)
