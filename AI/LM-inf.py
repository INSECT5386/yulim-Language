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
pad_id = sp.piece_to_id("<pad>") if sp.piece_to_id("<pad>") != -1 else 0  
end_id = sp.piece_to_id("<end>") # 코드가 정상적으로 실행되기 위해 end_id도 필요
print(f"Special Tokens: <start>={start_token}, <end>={end_token}, <sep>={sep_token}")


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



# 모델 생성

model = ReLaM(vocab_size, max_seq_len=max_len, d_model=256, n_layers=4, dropout_rate=0.1)
dummy_input = tf.zeros((1, max_len), dtype=tf.int32)  # 배치1, 시퀀스길이 max_len  
_ = model(dummy_input)  # 모델이 빌드됨  
model.load_weights('/content/Cobra.weights.h5')  
print("모델 가중치 로드 완료!")  



# 더미 인풋으로 모델 초기화
dummy_input = np.zeros((1, max_len), dtype=np.int32)
model(dummy_input)
model.summary()


def generate_text_topp(model, prompt, max_len=100, max_gen=98, p=0.9, temperature=0.8, min_len=20):
    # 필요한 ID 변수들을 함수 내부에서 참조
    global pad_id, end_id, start_token, sep_token # 전역 변수 참조

    # 시작 프롬프트 인코딩: <start> 프롬프트 <sep>
    model_input = text_to_ids(f"{start_token} {prompt} {sep_token}")
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
        
        # 현재 생성 중인 토큰의 로짓을 가져옵니다.
        # len(input_seq) - 1 은 시퀀스의 마지막 위치입니다.
        next_token_logits = logits[0, len(input_seq) - 1].numpy()
        
        # === 오류 수정: 토큰 문자열 대신 토큰 ID(정수) 사용 ===
        # <end> 토큰의 생성 확률을 낮춥니다.
        next_token_logits[end_id] -= 5.0
        # <pad> 토큰의 생성 확률을 낮춥니다.
        next_token_logits[pad_id] -= 10.0
        # ===================================================

        probs = tf.nn.softmax(next_token_logits / temperature).numpy()
        
        # Top-p (Nucleus) 샘플링
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumulative_probs, p)
        top_indices = sorted_indices[:cutoff + 1]
        top_probs = sorted_probs[:cutoff + 1]
        top_probs /= np.sum(top_probs)
        
        next_token_id = np.random.choice(top_indices, p=top_probs)
        
        # 종료 조건
        if next_token_id == end_id and len(generated) >= min_len:
            break
            
        generated.append(int(next_token_id))
        
    return ids_to_text(generated)

# ----------------------------------------------------------------------------------
# (참고) 코드의 상단에서 이미 pad_id와 end_id는 다음과 같이 정의되어 있어야 합니다.
# pad_id = sp.piece_to_id("<pad>") if sp.piece_to_id("<pad>") != -1 else 0  
# end_id = sp.piece_to_id("<end>") # 코드가 정상적으로 실행되기 위해 end_id도 필요
# ----------------------------------------------------------------------------------

print("\n\n===== 생성 결과 =====")  
print(generate_text_topp(model, "안녕", p=0.9))
