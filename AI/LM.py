import json
import numpy as np
import tensorflow as tf
import sentencepiece as spm
import requests
from tensorflow.keras import layers

sp = spm.SentencePieceProcessor()
# 'ko_unigram.model' 파일이 현재 경로에 있어야 합니다.
sp.load("ko_unigram.model") 

pad_id = sp.piece_to_id("<pad>") if sp.piece_to_id("<pad>") != -1 else 0  
start_token = sp.id_to_piece(sp.piece_to_id("<start>")) if sp.piece_to_id("<start>") != -1 else "<start>"
end_token = sp.id_to_piece(sp.piece_to_id("<end>")) if sp.piece_to_id("<end>") != -1 else "<end>"
sep_token = sp.id_to_piece(sp.piece_to_id("<sep>")) if sp.piece_to_id("<sep>") != -1 else "<sep>"
vocab_size = sp.get_piece_size()

print(f"Special Tokens: <start>={start_token}, <end>={end_token}, <sep>={sep_token}")

# ⬇️ JSONL 데이터 불러오기
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

jsonl_data = load_jsonl("output.jsonl")
print(f"✅ JSONL 데이터 로드 완료: {len(jsonl_data)}개의 대화")

# ⬇️ 멀티턴 대화를 토큰 ID 시퀀스로 변환
# 학습 데이터는 문자열 리스트 대신 토큰 ID 리스트의 리스트가 됩니다.
all_token_sequences = [] # 변수명을 변경하여 최종 데이터 형태를 명확히 했습니다.

for item in jsonl_data:
    conversations = item.get("conversations", [])
    
    # dialogue_parts는 전체 대화를 구성하는 문자열 조각들의 리스트입니다.
    dialogue_parts = [start_token]
    
    for msg in conversations:
        # 텍스트 전처리: 공백 제거 및 줄바꿈 공백으로 대체
        text = msg.get("value", "").strip().replace("\n", " ") 
        
        if msg.get("from") == "human":
            # "[사용자 발화]" 형태
            dialogue_parts.append(f"{start_token} {text}")
        elif msg.get("from") == "gpt":
            # "[sep] [GPT 응답]" 형태
            dialogue_parts.append(f"{sep_token} {text}")
            
    dialogue_parts.append(end_token)
    
    # 1. 모든 문자열 조각을 공백 한 칸으로 연결하여 하나의 긴 문자열 생성
    full_dialogue_text = " ".join(dialogue_parts)
    
    # 2. SentencePiece 모델을 사용하여 긴 문자열을 토큰 ID 시퀀스로 변환
    full_dialogue_ids = sp.encode_as_ids(full_dialogue_text)
    
    # 토큰 ID 시퀀스를 최종 리스트에 추가
    all_token_sequences.append(full_dialogue_ids) # 변수명 변경 적용

print(f"✅ 토큰 ID 시퀀스 변환 완료: {len(all_token_sequences)}개의 시퀀스")
print("\n--- 첫 번째 변환 결과 미리보기 ---")
# 첫 번째 시퀀스의 길이와 토큰 ID를 출력
print(f"길이: {len(all_token_sequences[0])}")
print(f"토큰 ID 시퀀스 (일부): {all_token_sequences[0][:20]}...")
# ID 시퀀스를 다시 텍스트로 디코딩하여 확인
print(f"디코딩된 텍스트 (일부): {sp.decode_ids(all_token_sequences[0])[:100]}...")

# ----------------------------------------------------------------------
# ********** 여기서부터 오류 수정 적용 **********
# ----------------------------------------------------------------------

print(f"✅ Vocabulary size: {vocab_size}")

# 이제 이 함수들은 토큰 ID를 받지 않고, 문자열 <-> ID 변환만 담당
def text_to_ids(text):
    # 이 함수는 문자열을 기대하므로, list를 전달하면 안 됩니다.
    return sp.encode(text, out_type=int)

def ids_to_text(ids):
    return sp.decode(ids)

# ⬇️ 전처리 하이퍼파라미터
max_len = 300
batch_size = 8

encoded_inputs = []
targets = []

# train_sentences 대신 all_token_sequences 사용 (데이터 일관성 유지)
for ids in all_token_sequences: # ids는 이미 정수 ID 리스트입니다.
    # ids = text_to_ids(sentence) <-- 이 줄을 제거하여 TypeError를 방지

    if len(ids) < 2:
        continue
    
    # 입력과 타겟 생성 (정확히 max_len으로)
    if len(ids) >= max_len + 1:
        # 길면 자르기
        input_ids = ids[:max_len]
        target_ids = ids[1:max_len+1]
    else:
        # 짧으면 패딩
        input_ids = ids[:-1]  # 마지막 제외
        target_ids = ids[1:]   # 첫 번째 제외
        
        # max_len에 맞게 패딩
        pad_len = max_len - len(input_ids)
        input_ids = input_ids + [pad_id] * pad_len
        target_ids = target_ids + [pad_id] * pad_len
    
    # 정확히 max_len인지 확인
    assert len(input_ids) == max_len, f"Input length mismatch: {len(input_ids)}"
    assert len(target_ids) == max_len, f"Target length mismatch: {len(target_ids)}"
    
    encoded_inputs.append(input_ids)
    targets.append(target_ids)

# 이제 numpy 변환이 안전함
encoded_inputs = np.array(encoded_inputs, dtype=np.int32)
targets = np.array(targets, dtype=np.int32)

print(f"✅ 인코딩 완료: {encoded_inputs.shape}, {targets.shape}")

# ⬇️ TensorFlow Dataset 생성
dataset = tf.data.Dataset.from_tensor_slices((encoded_inputs, targets))
dataset = dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

print("✅ TF Dataset 생성 완료!")

# ⬇️ 샘플 확인
for input_batch, target_batch in dataset.take(1):
    print(f"\nBatch shapes: {input_batch.shape}, {target_batch.shape}")
    print(f"\n첫 번째 대화 샘플 (처음 200자):")
    decoded = ids_to_text(input_batch[0].numpy().tolist())
    print(decoded[:200])

# 이하 클래스 및 모델 코드는 변경 없음

class Lo(layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        # 내부 계산은 float32로 유지
        self.proj = layers.Dense(d_model, use_bias=True, dtype='float32')
        self.p = layers.Dense(128, use_bias=True, dtype='float32')
        self._out_dtype = 'float32'

    def call(self, x):
        # x may be bfloat16; cast to float32 for stable intermediate computation
        x_f32 = tf.cast(x, tf.float32)
        x = self.proj(x_f32)
        x = tf.nn.gelu(x)
        x = self.p(x)
        # cast back to model dtype for consistency
        return tf.cast(x, self._out_dtype)

class LoSoU(layers.Layer):
    """
    안정화된 LoSoU 레이어
    - 누적합 대신 지수이동평균(EMA) 사용 (alpha: smoothing factor)
    - 내부 계산은 float32로 수행 (TPU bfloat16 안정성 향상)
    - EMA 결과 클리핑 및 작은 epsilon 적용
    - 안전한 split 처리 (짝수 차원 가정; 아니라면 마지막 차원 pad 필요)
    """
    def __init__(self, d_model, alpha=0.15, clip_value=5.0, eps=1e-6):
        super().__init__()
        # 대부분 연산을 float32로 수행
        self.d_model = d_model
        self.alpha = float(alpha)
        self.clip_value = float(clip_value)
        self.eps = float(eps)

        # projection / gating layers in float32
        self.Q = layers.Dense(128, dtype='float32')
        self.K = layers.Dense(128, dtype='float32')
        # V produces d_model so keep it float32 internally
        self.V = Lo(d_model)  # Lo already handles casting to model dtype; we'll cast back to float32
        self.proj = layers.Dense(d_model, use_bias=True, dtype='float32')
        self.O = layers.Dense(d_model, dtype='float32')
        self.norm = layers.LayerNormalization(epsilon=1e-5, dtype='float32')

    def _ema_over_time(self, score):
        # score: (B, L, D) float32 in [0,1] roughly
        alpha = tf.constant(self.alpha, dtype=score.dtype)

        # transpose to (L, B, D) to scan over time steps
        seq = tf.transpose(score, perm=[1, 0, 2])

        def step(prev_ema, x_t):
            # prev_ema: (B, D), x_t: (B, D)
            new = alpha * x_t + (1.0 - alpha) * prev_ema
            return new

        # 초기값을 첫 step 값으로 설정
        init = seq[0]  

        ema_seq = tf.scan(fn=step, elems=seq[1:], initializer=init)
        ema_seq = tf.concat([tf.expand_dims(init, 0), ema_seq], axis=0)  # (L, B, D)

        # transpose back to (B, L, D)
        ema = tf.transpose(ema_seq, perm=[1, 0, 2])
        return ema


    def call(self, x):
        # x: (B, L, d_model) maybe bfloat16 or float32
        # cast to float32 for all internal computations
        x_f32 = tf.cast(x, tf.float32)
        residual = x_f32

        # Q, K, V
        q = self.Q(x_f32)   # (B, L, 128)
        k = self.K(x_f32)   # (B, L, 128)
        V = tf.cast(self.V(x), tf.float32)  # ensure V's output is float32

        # gating signals in (0,1)
        g_q = tf.nn.sigmoid(q)
        g_k = tf.nn.sigmoid(k)

        # elementwise product -> bounded roughly [0,1]
        score = g_q * g_k

        # EMA across time (stable alternative to cumsum)
        score_ema = self._ema_over_time(score)

        # optionally normalize by (mean + eps) across last dim to reduce scale variations
        mean_last = tf.reduce_mean(score_ema, axis=-1, keepdims=True)  # (B, L, 1)
        denom = tf.maximum(mean_last, self.eps)
        score_norm = score_ema / denom

        # clip to avoid extremes
        score_clipped = tf.clip_by_value(score_norm, -self.clip_value, self.clip_value)

        # combine with V
        x_comb = score_clipped * V  # (B, L, d_model)

        out = self.proj(x_comb)  # (B, L, d_model)

        # ensure out dim even for split
        d = out.shape[-1]  # this is an int (static shape)
        if d is not None and d % 2 == 1:
            out = tf.pad(out, [[0,0],[0,0],[0,1]])


        a, b = tf.split(out, 2, axis=-1)
        gated = tf.nn.silu(a) * b
        out = self.O(gated)

        out = self.norm(out + residual)

        # cast back to original dtype for downstream layers
        return tf.cast(out, x.dtype)

class Block(layers.Layer):
    def __init__(self, d_model, r, hyper_n, num_heads, num_groups):
        super().__init__()
        self.losou = [LoSoU(d_model) for _ in range(hyper_n)]

    def call(self, x):
        for losou in self.losou:
            x = losou(x)
        return x

class ReLaM(tf.keras.Model):
    def __init__(self, vocab_size, max_seq_len, d_model, n_layers, dropout_rate=0.1):
        super().__init__()
        self.token_embedding = layers.Embedding(vocab_size, d_model)
        self.pos_embedding = layers.Embedding(max_seq_len, d_model)
        self.blocks = [Block(d_model, r=204, hyper_n=3, num_heads=8, num_groups=2) for _ in range(n_layers)]

        # LayerNormalization은 float32로 해서 정밀도 문제 방지
        self.ln_f = layers.LayerNormalization(epsilon=1e-5, dtype="float32")

    def call(self, x, training=False):
        batch_size, seq_len = tf.shape(x)[0], tf.shape(x)[1]
        positions = tf.range(seq_len)[tf.newaxis, :]

        x = self.token_embedding(x) + self.pos_embedding(positions)
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        embedding_matrix = tf.cast(self.token_embedding.embeddings, x.dtype)
        logits = tf.matmul(x, embedding_matrix, transpose_b=True)
        return tf.cast(logits, tf.float32)

# 손실 함수 및 메트릭 정의
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def masked_loss(y_true, y_pred):
    loss = loss_fn(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
    masked_loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
    return masked_loss

def masked_accuracy(y_true, y_pred):
    preds = tf.argmax(y_pred, axis=-1, output_type=y_true.dtype)
    matches = tf.cast(tf.equal(y_true, preds), tf.float32)
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
    return tf.reduce_sum(matches * mask) / tf.reduce_sum(mask)

def masked_perplexity(y_true, y_pred):
    loss = loss_fn(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
    avg_loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
    return tf.exp(tf.minimum(avg_loss, 10.0))  # 수치 안정성 확보

def masked_top5_accuracy(y_true, y_pred):
    top5_preds = tf.nn.top_k(y_pred, k=5).indices
    top5_preds = tf.cast(top5_preds, dtype=y_true.dtype)  # <-- 이 줄 추가
    y_true_expanded = tf.expand_dims(y_true, axis=-1)
    matches = tf.reduce_any(tf.equal(y_true_expanded, top5_preds), axis=-1)
    matches = tf.cast(matches, tf.float32)
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
    return tf.reduce_sum(matches * mask) / tf.reduce_sum(mask)


def token_level_loss(y_true, y_pred):
    loss = loss_fn(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
    return tf.reduce_mean(loss * mask)

def create_lr_schedule(initial_lr=5e-5, decay_steps=10000, decay_rate=0.9):
    return tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=False
    )


# 모델 생성
model = ReLaM(
    vocab_size=vocab_size,
    max_seq_len=max_len,
    d_model=256,
    n_layers=4
)

# 옵티마이저 설정
optimizer = tf.keras.optimizers.Adam(
    learning_rate=create_lr_schedule(),
    beta_1=0.9,
    beta_2=0.95,
    epsilon=1e-8,
    clipnorm=1.0
)

# 모델 컴파일
model.compile(
    optimizer=optimizer,
    loss=masked_loss,
    metrics=[
        masked_accuracy,
        masked_perplexity,
        masked_top5_accuracy,
        token_level_loss
    ]
)

# 더미 인풋으로 모델 초기화
dummy_input = np.zeros((1, max_len), dtype=np.int32)
model(dummy_input)
model.summary()

# 학습 시작
history = model.fit(
    dataset,
    epochs=1,
    steps_per_epoch = encoded_inputs.shape[0] // batch_size,
    verbose=1
)

# 가중치 저장
model.save_weights("Cobra.weights.h5")
print("모델 가중치 저장 완료!")

def generate_text_topp(model, prompt, max_len=100, max_gen=98, p=0.9, temperature=0.8, min_len=20):
    model_input = text_to_ids(f"<start> {prompt} <sep>")
    model_input = model_input[:max_len]
    generated = list(model_input)
    for step in range(max_gen):
        if len(generated) > max_len:
            input_seq = generated[-max_len:]
        else:
            input_seq = generated
        input_padded = np.pad(input_seq, (0, max_len - len(input_seq)), constant_values=pad_id)
        input_tensor = tf.convert_to_tensor([input_padded])
        logits = model(input_tensor, training=False)
        next_token_logits = logits[0, len(input_seq) - 1].numpy()
        next_token_logits[end_token] -= 5.0
        next_token_logits[pad_id] -= 10.0
        probs = tf.nn.softmax(next_token_logits / temperature).numpy()
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumulative_probs, p)
        top_indices = sorted_indices[:cutoff + 1]
        top_probs = sorted_probs[:cutoff + 1]
        top_probs /= np.sum(top_probs)
        next_token_id = np.random.choice(top_indices, p=top_probs)
        if next_token_id == end_id and len(generated) >= min_len:
            break
        generated.append(int(next_token_id))
    return ids_to_text(generated)

print("\n\n===== 생성 결과 =====")  
print(generate_text_topp(model, "안녕", p=0.9))
