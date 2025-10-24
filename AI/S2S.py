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

# ⬇️ 파일 다운로드 함수
def download_file(url, save_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✅ 파일 저장됨: {save_path}")

# ⬇️ 데이터와 토크나이저 다운로드
download_file('https://huggingface.co/datasets/Yuchan5386/TinyInst/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet?download=true', 'dataset.parquet')
download_file('https://huggingface.co/datasets/Yuchan5386/TinyInst/resolve/main/ko_unigram.model?download=true', 'ko_unigram.model')

# ⬇️ Parquet 데이터 불러오기  
df = pd.read_parquet("dataset.parquet", engine="pyarrow")

# ⬇️ <start> 질문 <sep> 답변 <end> 포맷으로 변환
train_sentences = []

for conversations in df["conversations"]:
    for i in range(0, len(conversations) - 1, 2):
        item1, item2 = conversations[i], conversations[i + 1]
        if item1.get("from") == "human" and item2.get("from") == "gpt":
            prompt = item1.get("value", "").strip().replace("\n", " ")
            response = item2.get("value", "").strip().replace("\n", " ")
            full = f"<start> {prompt} <sep> {response} <end>"
            train_sentences.append(full)
train_sentences = train_sentences # 예제용 소량
print(f"총 문장 개수: {len(train_sentences)}")

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

# ⬇️ 전처리 결과 저장할 리스트
encoder_inputs = []
decoder_inputs = []
targets = []

for sentence in train_sentences:
    if "<sep>" not in sentence:
        continue

    sep_index = sentence.index("<sep>")
    input_text = sentence[:sep_index].strip() # 질문 부분
    target_text = sentence[sep_index + len("<sep>"):].strip() # 답변 부분

    # 인코더 입력: 질문 + <sep>
    enc_ids = sp.encode(input_text + " <sep>")[:max_enc_len]

# 디코더 입력: <start> + 답변[:-1]
    dec_input_ids = [start_id] + sp.encode(target_text)[:max_dec_len - 2]

# 정답 라벨: 답변 + <end>
    target_ids = sp.encode(target_text)[:max_dec_len - 1] + [end_id]
    # 패딩 추가
    enc_padded = enc_ids + [pad_id] * (max_enc_len - len(enc_ids))
    dec_padded = dec_input_ids + [pad_id] * (max_dec_len - len(dec_input_ids))
    target_padded = target_ids + [pad_id] * (max_dec_len - len(target_ids))

    encoder_inputs.append(enc_padded)
    decoder_inputs.append(dec_padded)
    targets.append(target_padded)

# ⬇️ 넘파이 배열로 변환
encoder_inputs = np.array(encoder_inputs, dtype=np.int32)
decoder_inputs = np.array(decoder_inputs, dtype=np.int32)
targets = np.array(targets, dtype=np.int32)

# ⬇️ TensorFlow Dataset 생성
def data_generator():
    for enc, dec, tgt in zip(encoder_inputs, decoder_inputs, targets):
        # 딕셔너리 대신 튜플 형태로 반환
        yield (enc, dec), tgt

output_types = (
    (tf.int32, tf.int32), # 두 개의 입력 텐서에 대한 타입
    tf.int32 # 타겟에 대한 타입
)

output_shapes = (
    (tf.TensorShape([max_enc_len]), tf.TensorShape([max_dec_len])), # 두 개의 입력 텐서에 대한 모양
    tf.TensorShape([max_dec_len]) # 타겟에 대한 모양
)

dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_types=output_types,
    output_shapes=output_shapes
)

dataset = dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
print("dataset ok")


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

class SeProdBlock(layers.Layer):
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

d_model = 256
dropout_rate = 0.1
# ===== 모델 구성 =====
# 인코더 경로
encoder_input = Input(shape=(max_enc_len,), name='encoder_input')
x_emb = layers.Embedding(input_dim=vocab_size, output_dim=d_model)(encoder_input)
x_pos = LearnablePositionalEmbedding(max_enc_len, d_model)(x_emb)

x = layers.Conv1D(d_model, kernel_size=3, padding='same', dilation_rate=1, activation='relu')(x_pos)
x = layers.Conv1D(d_model, kernel_size=3, padding='same', dilation_rate=2, activation='relu')(x)
x = layers.Conv1D(d_model, kernel_size=3, padding='same', dilation_rate=3, activation='relu')(x)
x = layers.Conv1D(d_model, kernel_size=3, padding='same', dilation_rate=4, activation='relu')(x)
context_vector =layers.Conv1D(d_model, kernel_size=3, padding='same', dilation_rate=5, activation='relu')(x)
# 디코더 경로
decoder_input = Input(shape=(max_dec_len,), name='decoder_input')
y_emb = layers.Embedding(input_dim=vocab_size, output_dim=d_model)(decoder_input)
y_pos = LearnablePositionalEmbedding(max_dec_len, d_model)(y_emb)
decoder_output = SeProdBlock(d_model, dropout_rate=dropout_rate)(y_pos, y_pos, training=True)
output = SeProdBlock(d_model, dropout_rate=dropout_rate)(decoder_output, context_vector)

# 최종 출력
logits = layers.Dense(vocab_size)(output)

model = Model(inputs=[encoder_input, decoder_input], outputs=logits, name='SeProd')

# ===== 컴파일 및 학습 =====
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 모델 요약
model.summary()

model.fit(dataset, epochs=1, steps_per_epoch=len(train_sentences) // batch_size)
model.save('tf_model.h5')

def generate(model, sp, input_text, max_dec_len=128, temperature=0.7, verbose=False):
    start_id = sp.piece_to_id("<start>")
    end_id = sp.piece_to_id("<end>")
    
    # 인코더 입력 전처리
    enc_ids = sp.encode(input_text)
    enc_ids = enc_ids[:max_enc_len]
    enc_ids += [sp.pad_id()] * (max_enc_len - len(enc_ids))
    enc_tensor = tf.constant([enc_ids], dtype=tf.int32)

    if verbose:
        print("Encoder Input:", input_text)
        print("Encoded:", enc_ids)

    # 초기 디코더 입력 설정
    dec_input = tf.constant([[start_id]], dtype=tf.int32)
    generated_ids = []

    for step in range(max_dec_len):
        # 디코더 입력을 max_dec_len으로 패딩
        padded_dec_input = tf.pad(dec_input, [[0, 0], [0, max_dec_len - tf.shape(dec_input)[1]]],
                                  constant_values=sp.pad_id())

        # 전체 모델 예측
        decoder_output = model.predict([enc_tensor, padded_dec_input])
        
        # 현재 스텝의 로짓 추출
        logits = decoder_output[:, step, :]  # 현재 step 위치의 로짓

        if temperature == 0.:
            pred_id = tf.argmax(logits, axis=-1, output_type=tf.int32)
        else:
            logits = logits / temperature
            pred_id = tf.random.categorical(logits, 1, dtype=tf.int32)

        pred_id = tf.squeeze(pred_id, axis=1)  # (1, )

        # 종료 토큰 체크
        if int(pred_id[0]) == end_id:
            break

        generated_ids.append(int(pred_id[0]))
        dec_input = tf.concat([dec_input, pred_id[:, tf.newaxis]], axis=1)

        if verbose:
            token_str = sp.decode([int(pred_id[0])])
            print(f"Step {step}: ID={int(pred_id[0])}, Token='{token_str}'")

    decoded_text = sp.decode(generated_ids)
    return decoded_text

input_text = "안녕하세요! 어떻게 지내셨나요?"
response = generate(model, sp, input_text, temperature=0.7, verbose=True)
print("AI Response:", response)
