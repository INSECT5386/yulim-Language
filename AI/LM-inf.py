

import json  
import numpy as np  
import pandas as pd
import tensorflow as tf  
from tensorflow.keras import layers 
from tensorflow.keras.initializers import RandomNormal
import sentencepiece as spm  
import requests
from tensorflow.keras.initializers import RandomNormal

import tensorflow as tf
from tensorflow.keras import layers, Model, Input

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.initializers import RandomNormal


import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Dropout

from tensorflow.keras.models import load_model

# ⬇️ 토크나이저 불러오기
sp = spm.SentencePieceProcessor()
sp.load("ko_unigram.model")

# ⬇️ 특수 토큰 ID 추출
pad_id = sp.piece_to_id("<pad>") if sp.piece_to_id("<pad>") != -1 else 0  
start_id = sp.piece_to_id("<start>")  
sep_id = sp.piece_to_id("<sep>")  
end_id = sp.piece_to_id("<end>")  

vocab_size = sp.get_piece_size()
print(f"✅ Vocabulary size: {vocab_size}")

# ⬇️ 전처리 하이퍼파라미터
max_enc_len = 128 # 인코더 최대 길이 (질문 부분)
max_dec_len = 128 # 디코더 최대 길이 (답변 부분)
batch_size = 64

# ===== 1. 가변 위치 인코딩 =====
# ===== 1. 가변 위치 인코딩 =====
class LearnablePositionalEmbedding(layers.Layer):
    def __init__(self, max_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.d_model = d_model
        self.add = layers.Add()
        pos_emb = RandomNormal()(shape=[max_length, d_model])
        self.pos_emb = tf.Variable(
            initial_value=pos_emb,
            trainable=True,
            name='positional_embedding'
        )

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return self.add([inputs, self.pos_emb[tf.newaxis, :seq_len, :]])

class CrossBlock(layers.Layer):
    def __init__(self, dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.dense1 = layers.Dense(dim)
        self.dense2 = layers.Dense(dim)
        self.dense = layers.Dense(dim)
  
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x, z, training=None):
        batch_size, seq_len, d_model = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        x = self.dense(x)

        # ===== Reverse Block (GLU Style) =====
        A2 = self.dense2(z)  # [B, T, D*2]
        splits = tf.split(A2, num_or_size_splits=8, axis=-1)
        a, at, b, bt, c, ct, d, dt = splits
        splits1 = tf.split(x, num_or_size_splits=8, axis=-1)
        A, A1, B, B1, C, C1, D, D1 = splits1

        a = tf.sigmoid(a)
        b = tf.nn.silu(b)
        c = tf.nn.gelu(c)
        d = tf.nn.tanh(d)


        A = tf.sigmoid(A)
        B = tf.nn.silu(B)
        C = tf.nn.gelu(C)
        D = tf.nn.tanh(D)

        Ath = A * A1
        Bth = B * B1
        Cth = C * C1
        Dth = D * D1

        ath = a * at
        bth = b * bt
        cth = c * ct
        dth = d * dt

        z_th = tf.concat([ath, bth, cth, dth, Ath, Bth, Cth, Dth], axis=-1)  # [B, T, D*2]
      
        z_th = self.norm1(z_th)
        x = z_th
        x = self.dense1(x)
        x = self.norm2(x)
        f, ft = tf.split(x, num_or_size_splits=2, axis=-1)
        f = tf.nn.silu(f)
        output = f * ft
        return output

class EnBlock(layers.Layer):
    def __init__(self, dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.c = layers.Conv1D(dim, kernel_size=3, padding='same', dilation_rate=1, activation='relu')
        self.c1 = layers.Conv1D(dim, kernel_size=3, padding='same', dilation_rate=2, activation='relu')
        self.c2 = layers.Conv1D(dim, kernel_size=3, padding='same', dilation_rate=3, activation='relu')
        self.c3 = layers.Conv1D(dim, kernel_size=3, padding='same', dilation_rate=4, activation='relu')
        self.c4 = layers.Conv1D(dim, kernel_size=3, padding='same', dilation_rate=5, activation='relu')
    def call(self, x, training=None):
        x = self.c(x)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        return x

class DeBlock(layers.Layer):
    def __init__(self, dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.c = layers.Conv1D(dim, kernel_size=3, padding='causal', dilation_rate=1, activation='relu')
        self.c1 = layers.Conv1D(dim, kernel_size=3, padding='causal', dilation_rate=2, activation='relu')
        self.c2 = layers.Conv1D(dim, kernel_size=3, padding='causal', dilation_rate=3, activation='relu')
        self.c3 = layers.Conv1D(dim, kernel_size=3, padding='causal', dilation_rate=4, activation='relu')
        self.c4 = layers.Conv1D(dim, kernel_size=3, padding='causal', dilation_rate=5, activation='relu')
    def call(self, x, training=None):
        x = self.c(x)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        return x
        
d_model = 256
dropout_rate = 0.1
# ===== 모델 구성 =====
# 인코더 경로
encoder_input = Input(shape=(max_enc_len,), name='encoder_input')
x_emb = layers.Embedding(input_dim=vocab_size, output_dim=d_model)(encoder_input)
x_pos = LearnablePositionalEmbedding(max_enc_len, d_model)(x_emb)

context_vector = EnBlock(d_model)(x_pos, training=True)

# 디코더 경로
decoder_input = Input(shape=(max_dec_len,), name='decoder_input')
y_emb = layers.Embedding(input_dim=vocab_size, output_dim=d_model)(decoder_input)
y_pos = LearnablePositionalEmbedding(max_dec_len, d_model)(y_emb)
decoder_output = DeBlock(d_model)(y_pos, training=True)
output = CrossBlock(d_model, dropout_rate=dropout_rate)(decoder_output, context_vector)

# 최종 출력
logits = layers.Dense(vocab_size)(output)

model = Model(inputs=[encoder_input, decoder_input], outputs=logits, name='SeProd')
model = load_model('tf_model.h5', custom_objects={"LearnablePositionalEmbedding": LearnablePositionalEmbedding, "SeProdBlock": SeProdBlock, "EnBlock": EnBlock, "DeBlock": DeBlock, "CrossBlock":CrossBlock})


import tensorflow as tf
import numpy as np # For potential numpy array to tensor conversion if needed, though tf is primary

# Note: The original function used 'max_enc_len' in its logic but not in its signature.
# I'm adding 'max_enc_len' to the signature for correctness.
def generate(model, sp, input_text, max_dec_len=128, max_enc_len=128, 
             temperature=0.7, top_k=0, min_dec_len=1, verbose=False):
    """
    Generates a sequence using the given model, incorporating temperature, top-k sampling, 
    and minimum decoding length control.

    :param model: The Seq2Seq or Transformer model (e.g., Keras Model).
    :param sp: SentencePiece processor object.
    :param input_text: The input text for the encoder.
    :param max_dec_len: Maximum length for the decoded sequence.
    :param max_enc_len: Maximum length for the encoder input.
    :param temperature: Sampling temperature (0.0 for greedy decoding).
    :param top_k: If > 0, only sample from the top_k most likely tokens.
    :param min_dec_len: Minimum length of the decoded sequence before the <end> token can be considered.
    :param verbose: If True, prints generation steps.
    :return: The generated text string.
    """
    start_id = sp.piece_to_id("<start>")
    end_id = sp.piece_to_id("<end>")
    
    # 인코더 입력 전처리
    enc_ids = sp.encode(input_text)
    # max_enc_len으로 잘라내고 패딩
    enc_ids = enc_ids[:max_enc_len]
    enc_ids += [sp.pad_id()] * (max_enc_len - len(enc_ids))
    enc_tensor = tf.constant([enc_ids], dtype=tf.int32)

    if verbose:
        print("Encoder Input:", input_text)
        print("Encoded:", enc_ids)

    # 초기 디코더 입력 설정
    dec_input = tf.constant([[start_id]], dtype=tf.int32)
    generated_ids = []

    # 'predict' 호출 최적화를 위해 디코더 입력의 최대 길이 계산
    # (원래 코드는 매 스텝마다 max_dec_len으로 패딩 후 predict를 호출하는 비효율적인 방식이었습니다.
    # 하지만 기존 로직을 유지하면서 top_k와 min_len만 추가하겠습니다.)
    
    for step in range(max_dec_len):
        # 디코더 입력을 max_dec_len으로 패딩
        padded_dec_input = tf.pad(dec_input, [[0, 0], [0, max_dec_len - tf.shape(dec_input)[1]]],
                                  constant_values=sp.pad_id())

        # 전체 모델 예측 (주의: 이 방식은 매 스텝마다 전체 시퀀스를 재계산할 가능성이 높아 비효율적일 수 있습니다.
        # 일반적으로는 트랜스포머의 경우 캐싱을 사용합니다.)
        # 하지만 기존 함수의 구조를 유지합니다.
        # model.predict는 (1, max_dec_len, vocab_size) 형태의 출력을 가정합니다.
        decoder_output = model.predict([enc_tensor, padded_dec_input], verbose=0)
        
        # 현재 스텝의 로짓 추출: (1, vocab_size)
        logits = decoder_output[:, step, :]
        
        # 온도 조절
        logits = logits / temperature if temperature > 0. else logits

        # Top-k 필터링 (temperature > 0. 일 때만 의미 있음)
        if top_k > 0 and temperature > 0.:
            # Top-k 값을 계산하여 가장 낮은 k번째 값보다 작은 로짓은 -inf로 설정
            values, _ = tf.math.top_k(logits, k=top_k)
            min_kth_val = values[:, -1, tf.newaxis] # k번째 로짓 값

            # k번째 로짓 값보다 작은 모든 로짓을 매우 작은 값(실질적으로 -inf)으로 마스킹
            # tf.where를 사용하여 마스킹
            logits = tf.where(logits < min_kth_val, tf.fill(tf.shape(logits), -1e9), logits)


        # 최소 길이 보장: 현재 스텝이 min_dec_len보다 작을 경우 <end> 토큰의 확률을 0으로 만듭니다.
        if step < min_dec_len:
            # <end> 토큰의 로짓을 매우 작은 값(실질적으로 -inf)으로 설정
            end_token_mask = tf.one_hot([end_id], depth=tf.shape(logits)[-1], dtype=tf.float32)
            # logits = logits - end_token_mask * 1e9
            
            # tf.tensor_scatter_nd_update를 사용하여 end_id 위치의 로짓을 마스킹합니다.
            indices = tf.constant([[0, end_id]]) # 배치의 0번째 요소, end_id 인덱스
            updates = tf.constant([-1e9]) # 마스킹 값
            logits = tf.tensor_scatter_nd_update(logits, indices, updates)


        if temperature == 0. or top_k == 0:
            # Greedy 또는 필터링 없는 샘플링 (temperature=0. 일 때 argmax가 됨)
            if temperature == 0.:
                pred_id = tf.argmax(logits, axis=-1, output_type=tf.int32)
                pred_id = pred_id[:, tf.newaxis] # (1, 1) 형태로 변환
            else:
                pred_id = tf.random.categorical(logits, 1, dtype=tf.int32)
        else:
            # Top-k 샘플링 (이미 로짓 필터링 완료)
            pred_id = tf.random.categorical(logits, 1, dtype=tf.int32)

        pred_id = tf.squeeze(pred_id, axis=1)  # (1, )

        # 종료 토큰 체크
        if int(pred_id[0]) == end_id:
            break

        generated_ids.append(int(pred_id[0]))
        # 다음 스텝을 위해 디코더 입력에 추가
        dec_input = tf.concat([dec_input, pred_id[:, tf.newaxis]], axis=1)

        if verbose:
            token_str = sp.decode([int(pred_id[0])])
            print(f"Step {step}: ID={int(pred_id[0])}, Token='{token_str}'")

    decoded_text = sp.decode(generated_ids)
    return decoded_text

# 예시 호출 (실행을 위해서는 model, sp, max_enc_len 등이 정의되어 있어야 합니다.)
input_text = "안녕하세요! 어떻게 지내셨는지 알려주실 수 있나요?"
response = generate(model, sp, input_text, temperature=0.7, top_k=5, min_dec_len=11, verbose=True)
print("AI Response:", response)
