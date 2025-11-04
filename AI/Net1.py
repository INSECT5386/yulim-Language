import tensorflow as tf

class CustomRNNCell(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        super(CustomRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # 가중치: 입력 → 은닉 (W_ih), 은닉 → 은닉 (W_hh), 편향 (b_h)
        self.W_ih = self.add_weight(
            shape=(input_dim, self.units),
            initializer='glorot_uniform',
            name='W_ih',
            trainable=True
        )
        self.W_hh = self.add_weight(
            shape=(self.units, self.units),
            initializer='orthogonal',  # RNN에서는 orthogonal 초기화가 좋음
            name='W_hh',
            trainable=True
        )
        self.b_h = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            name='b_h',
            trainable=True
        )
        self.built = True

    def call(self, inputs, states):
        h_prev = states[0]
        h = tf.nn.tanh(
            tf.matmul(inputs, self.W_ih) +
            tf.matmul(h_prev, self.W_hh) +
            self.b_h
        )
        return h, [h]

    @property
    def state_size(self):
        return self.units

    @property
    def output_size(self):
        return self.units
