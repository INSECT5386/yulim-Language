class CustomRNNModel(tf.keras.Model):
    def __init__(self, rnn_units, output_dim, sequence_length, **kwargs):
        super(CustomRNNModel, self).__init__(**kwargs)
        self.rnn_units = rnn_units
        self.sequence_length = sequence_length

        # 커스텀 셀
        self.rnn_cell = CustomRNNCell(rnn_units)
        self.rnn_layer = tf.keras.layers.RNN(
            self.rnn_cell,
            return_sequences=True,
            return_state=False
        )

        # 학습 가능한 초기 은닉 상태 h0: (batch_size, rnn_units) → 그러나 배치 독립적이어야 하므로 (1, rnn_units)
        # 실제로는 배치 크기와 무관하게 사용되므로 (1, rnn_units)로 만들고, 필요시 tile
        self.initial_state_trainable = self.add_weight(
            shape=(1, rnn_units),
            initializer='zeros',
            trainable=True,
            name='initial_hidden_state'
        )

        # 출력 레이어
        self.dense = tf.keras.layers.Dense(output_dim)

    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        # 배치 크기만큼 initial_state 복제
        initial_state = tf.tile(self.initial_state_trainable, [batch_size, 1])
        rnn_out = self.rnn_layer(inputs, initial_state=initial_state)
        output = self.dense(rnn_out)
        return output
