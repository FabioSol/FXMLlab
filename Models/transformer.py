
from keras.src.layers import GlobalAveragePooling1D


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization,BatchNormalization
from tensorflow.keras.models import Model

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        self.dense = Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output, attention_weights

    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

def transformer_encoder(inputs, num_layers, dff, d_model, num_heads, dropout):
    attention_output = inputs
    mask = None
    for _ in range(num_layers):
        attention_output, _ = MultiHeadSelfAttention(
            d_model, num_heads)(attention_output, attention_output, attention_output, mask)
        attention_output = Dropout(dropout)(attention_output)
        attention_output = BatchNormalization(epsilon=1e-6)(inputs + attention_output)
        ffn_output = Dense(dff, activation='relu')(attention_output)
        ffn_output = Dense(d_model)(ffn_output)
        ffn_output = Dropout(dropout)(ffn_output)
        ffn_output = BatchNormalization(epsilon=1e-6)(attention_output + ffn_output)
    return ffn_output

def transformer_model(input_shape, num_layers, dff, d_model, num_heads, dropout, output_dim):
    inputs = Input(shape=input_shape)
    x = PositionalEncoding(input_shape[0], d_model)(inputs)
    x = transformer_encoder(x, num_layers, dff, d_model, num_heads, dropout)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(output_dim, activation='linear')(x)  # Use linear activation for regression
    model = Model(inputs=inputs, outputs=outputs)
    return model
"""
# Example usage:
input_shape = (X_train.shape[1], X_train.shape[2])  # Sequence length and input dimension
num_layers = 4
dff = 512
d_model = 7
num_heads = 7
dropout = 0.1
output_dim = 1

model = transformer_model(input_shape, num_layers, dff, d_model, num_heads, dropout, output_dim)
model.compile(optimizer='adam', loss='mse')

# Train the model with your data



"""