import json  
import numpy as np  
import pandas as pd
import tensorflow as tf  
from tensorflow.keras import layers 
import sentencepiece as spm  
import requests

class SwiGLUFFN(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.gate_proj = layers.Dense(dim)
        self.up_proj = layers.Dense(dim)
        self.down_proj = layers.Dense(dim)

    def call(self, x):
        gate = tf.nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

# ======================= RoPE ==========================
def apply_rope(x):
    seq_len = tf.shape(x)[1]
    dim = tf.shape(x)[2]
    half_dim = dim // 2

    position = tf.cast(tf.range(seq_len), tf.float32)  # (T,)  <-- 이게 핵심!
    freq = tf.pow(10000.0, -tf.range(half_dim, dtype=tf.float32) / tf.cast(half_dim, tf.float32))  # (D/2,)
    angles = tf.einsum('i,j->ij', position, freq)  # (T, D/2)

    sin = tf.sin(angles)[None, :, :]  # (1, T, D/2)
    cos = tf.cos(angles)[None, :, :]  # (1, T, D/2)

    x1 = x[:, :, :half_dim]
    x2 = x[:, :, half_dim:]
    x_rot = tf.concat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return x_rot

# ======================= Cobrablock ======================
class Cobrablock(tf.keras.layers.Layer):
    def __init__(self, d_model, dropout_rate=0.1):
        super().__init__()
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = layers.MultiHeadAttention(8, 32)
        self.dropout1 = layers.Dropout(dropout_rate)

        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.ffn = SwiGLUFFN(d_model)
        self.w1 = layers.Dense(d_model)
        self.w2 = layers.Dense(d_model)
        self.w3 = layers.Dense(d_model)
    def call(self, x, training=False):
        residual = x
        x = self.norm1(x)
        q = self.w1(x)
        k = self.w2(x)
        v = self.w3(x)
        q = apply_rope(q)
        k = apply_rope(k)
        x = self.attn(q, k, v)
        x = residual + self.dropout1(x, training=training)

        x = self.ffn(x)

        residual = x
        x = self.norm2(x)
        x = self.dropout2(x, training=training)
        x = residual + x

        return x

# ======================= CobraModel ======================
class CobraModel(tf.keras.Model):
    def __init__(self, vocab_size, d_model, n_layers, dropout_rate=0.1):
        super().__init__()
        self.token_embedding = layers.Embedding(vocab_size, d_model)
        self.blocks = [Cobrablock(d_model, dropout_rate) for _ in range(n_layers)]
        self.ln_f = layers.LayerNormalization(epsilon=1e-5)

    def call(self, x, training=False):
        x = self.token_embedding(x)
        for block in self.blocks:
            x = block(x, training=training)
        x = self.ln_f(x)
        logits = tf.matmul(x, self.token_embedding.embeddings, transpose_b=True)
        return logits
