import json  
import numpy as np  
import pandas as pd
import tensorflow as tf  
from tensorflow.keras import layers 
import sentencepiece as spm  
import requests

# ⬇️ 파일 다운로드 함수
def download_file(url, save_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✅ 파일 저장됨: {save_path}")

# ⬇️ 데이터와 토크나이저 다운로드
download_file('https://huggingface.co/datasets/Yuchan5386/TinyInst/resolve/main/ko_unigram.model?download=true', 'ko_unigram.model')
download_file('https://huggingface.co/datasets/Yuchan5386/TinyInst/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet?download=true', 'dataset.parquet')

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
train_sentences = train_sentences
print(f"총 문장 개수: {len(train_sentences)}")

# ⬇️ 토크나이저 불러오기
sp = spm.SentencePieceProcessor()
sp.load("ko_unigram.model")

# ⬇️ 특수 토큰 ID 추출
pad_id = sp.piece_to_id("<pad>") if sp.piece_to_id("<pad>") != -1 else 0  
start_id = sp.piece_to_id("<start>")  
sep_id = sp.piece_to_id("<sep>")  
end_id = sp.piece_to_id("<end>")  
unk_id = sp.piece_to_id("<unk>")  

vocab_size = sp.get_piece_size()
print(f"✅ Vocabulary size: {vocab_size}")

# ⬇️ 텍스트 <-> ID 변환 함수
def text_to_ids(text):
    return sp.encode(text, out_type=int)

def ids_to_text(ids):
    return sp.decode(ids)

# ⬇️ 전처리 하이퍼파라미터
max_len = 100
batch_size = 128

# ⬇️ 인풋과 타겟 마스킹 포함된 전처리
encoded_inputs = []
targets = []

for sentence in train_sentences:
    if "<sep>" not in sentence:
        continue

    sep_index = sentence.index("<sep>")
    input_text = sentence[:sep_index + len("<sep>")].strip()
    target_text = sentence[sep_index + len("<sep>"):].strip()

    input_ids = text_to_ids(input_text)
    target_ids = text_to_ids(target_text + " <end>")

    full_input = input_ids + target_ids
    full_input = full_input[:max_len]

    target_mask = [0] * len(input_ids) + [1] * len(target_ids)
    target_mask = target_mask[:max_len]

    if len(full_input) < max_len:
        pad_len = max_len - len(full_input)
        full_input += [pad_id] * pad_len
        target_mask += [0] * pad_len

    encoded_inputs.append(full_input)

    target_seq = full_input[1:] + [end_id]
    target_seq = target_seq[:max_len]

    masked_target = [
        t if m == 1 else pad_id
        for t, m in zip(target_seq, target_mask)
    ]

    targets.append(masked_target)

# ⬇️ 넘파이 변환
encoded_inputs = np.array(encoded_inputs)
targets = np.array(targets)

# ⬇️ TensorFlow Dataset 생성
def data_generator():
    for input_seq, target_seq in zip(encoded_inputs, targets):
        yield input_seq, target_seq

dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=(max_len,), dtype=tf.int32),
        tf.TensorSpec(shape=(max_len,), dtype=tf.int32)
    )
)

dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

print("✅ TF Dataset 생성 완료!")

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

class SwiRUCell(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        super(SwiRUCell, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # 🔸 기본 가중치: 입력 → 게이트
        self.W_ih = self.add_weight(
            shape=(input_dim, self.units),
            initializer='glorot_uniform',
            name='W_ih',
            trainable=True
        )
        self.b_h = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            name='b_h',
            trainable=True
        )

        # 🔸 Dense 레이어는 __init__에서 생성해야 하지만, build 시점에 input_dim 알 수 있으므로 여기서 생성 가능
        # 그러나 Keras 호환성을 위해 __init__에서 생성하는 것이 더 안전 → 이를 위해 get_config도 구현 필요
        # 간단히 하기 위해 여기서 생성하되, build 호출 보장 전제
        self.W1 = layers.Dense(self.units, activation='silu', name='W1')
        self.W2 = layers.Dense(self.units, name='W2')  # no activation
        self.ln = layers.LayerNormalization(epsilon=1e-5, dtype="float32", name='ln')

        # 🔸 서브 레이어의 가중치를 build
        self.W1.build(input_shape)
        self.W2.build(input_shape)
        self.ln.build((None, self.units))

        self.built = True

    def call(self, inputs, states):
        h_prev = states[0]
        # 게이트 f ∈ [0,1]
        f = tf.sigmoid(tf.matmul(inputs, self.W_ih) + self.b_h)
        # SwiGLU 스타일: Swish(x) ⊙ x
        x = self.W1(inputs) * self.W2(inputs)
        # RRU 스타일 업데이트 + LayerNorm + residual
        h_raw = f * h_prev + (1.0 - f) * x
        h_norm = self.ln(h_raw)
        h = h_norm + h_prev  # residual connection
        return h, [h]

    @property
    def state_size(self):
        return self.units

    @property
    def output_size(self):
        return self.units

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

class Respiso(tf.keras.Model):
    def __init__(self, vocab_size, max_seq_len, d_model, n_layers, dropout_rate=0.1):
        super().__init__()
        self.token_embedding = layers.Embedding(vocab_size, d_model)
        self.rnn_cell = SwiRUCell(units=d_model)
        self.rnn_layer = tf.keras.layers.RNN(
            self.rnn_cell,
            return_sequences=True,
            return_state=False
        )
        self.initial_state_trainable = self.add_weight(
            shape=(1, rnn_units),
            initializer='zeros',
            trainable=True,
            name='initial_hidden_state'
        )
        # LayerNormalization은 float32로 해서 정밀도 문제 방지
        self.ln_f = layers.LayerNormalization(epsilon=1e-5, dtype="float32")

    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        x = self.token_embedding(x)
        # 배치 크기만큼 initial_state 복제
        initial_state = tf.tile(self.initial_state_trainable, [batch_size, 1])
        x = self.rnn_layer(x, initial_state=initial_state)

        x = self.ln_f(x)

        embedding_matrix = tf.cast(self.token_embedding.embeddings, x.dtype)
        logits = tf.matmul(x, embedding_matrix, transpose_b=True)
        return tf.cast(logits, tf.float32)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def masked_loss(y_true, y_pred):
    loss = loss_fn(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
    masked_loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
    return masked_loss

def masked_perplexity(y_true, y_pred):
    loss = loss_fn(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
    avg_loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
    return tf.exp(tf.minimum(avg_loss, 10.0))  # 수치 안정성 확보

def create_lr_schedule(initial_lr=5e-5, decay_steps=10000, decay_rate=0.9):
    return tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=False
    )

# 모델 생성
model = Respiso(
    vocab_size=vocab_size,
    max_seq_len=max_len,
    d_model=256,
    n_layers=1
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
        masked_perplexity
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
        next_token_logits[end_id] -= 5.0
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
