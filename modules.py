import tensorflow as tf
import math

from tensorflow.keras import layers
from einops import rearrange, repeat


class DoubleConv(layers.Layer):
    def __init__(self, filters, mid_filters=None, residual=False):
        super().__init__()

        self.residual = residual

        if not mid_filters:
            mid_filters = filters

        self.double_conv = tf.keras.Sequential([
            layers.Conv2D(filters=mid_filters, kernel_size=3, padding='same', use_bias=False),
            layers.GroupNormalization(groups=1),
            layers.Activation(activation=tf.keras.activations.gelu),

            layers.Conv2D(filters=filters, kernel_size=3, padding='same', use_bias=False),
            layers.GroupNormalization(groups=1),
        ])

    def call(self, x):
        if self.residual:
            return tf.nn.gelu(x + self.double_conv(x))

        else:
            return self.double_conv(x)


class SinusoidalPositionEmbedding(layers.Layer):
    def __init__(self, dim):
        super().__init__()

        self.half_dim = dim // 2

        embedding = math.sqrt(10000) / (self.half_dim - 1)
        self.embedding = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -embedding)

    def call(self, x):
        x = tf.cast(x, dtype=tf.float32)

        embedding = x[:, None] * self.embedding[None, :]
        embedding = tf.concat([tf.sin(embedding), tf.cos(embedding)], axis=-1)

        return embedding


class Down(layers.Layer):
    def __init__(self, filters):
        super().__init__()

        self.maxpool_conv = tf.keras.Sequential([
            layers.MaxPooling2D(pool_size=2),

            DoubleConv(filters[0], residual=True),
            DoubleConv(filters[1])
        ])

        self.embedding_linear = tf.keras.Sequential([
            layers.Activation(activation=tf.keras.activations.swish),
            layers.Dense(filters[1])
        ])

    def call(self, x, t):
        x = self.maxpool_conv(x)

        embedding = self.embedding_linear(t)

        x_shape = tf.shape(x)
        embedding = repeat(embedding[:, None, None, :], 'b 1 1 c -> b h w c', h=x_shape[-3], w=x_shape[-2])

        return x + embedding


class Up(layers.Layer):
    def __init__(self, filters):
        super().__init__()

        self.up = layers.UpSampling2D(size=2, interpolation='nearest')

        self.conv = tf.keras.Sequential([
            DoubleConv(filters[0], residual=True),
            DoubleConv(filters[1], filters[0] // 2)
        ])

        self.embedding_linear = tf.keras.Sequential([
            layers.Activation(activation=tf.keras.activations.swish),
            layers.Dense(filters[1])
        ])

    def call(self, x, skip_connection, t):
        x = self.up(x)

        x = tf.concat([skip_connection, x], axis=-1)

        x = self.conv(x)

        embedding = self.embedding_linear(t)

        x_shape = tf.shape(x)
        embedding = repeat(embedding[:, None, None, :], 'b 1 1 c -> b h w c', h=x_shape[-3], w=x_shape[-2])

        return x + embedding


class SelfAttention(layers.Layer):
    def __init__(self, units):
        super().__init__()

        self.multi_head_attention = layers.MultiHeadAttention(num_heads=4, key_dim=units)

        self.layer_norm = layers.LayerNormalization()

        self.projection_linear = tf.keras.Sequential([
            layers.LayerNormalization(),
            layers.Dense(units),
            layers.Activation(activation=tf.keras.activations.gelu),

            layers.Dense(units)
        ])

    # def build(self, input_shape):
    #     query_shape = (input_shape[0], input_shape[1] * input_shape[2], input_shape[3])
    #     key_shape = (input_shape[0], input_shape[1] * input_shape[2], input_shape[3])
    #     value_shape = (input_shape[0], input_shape[1] * input_shape[2], input_shape[3])
    #
    #     self.multi_head_attention._build_from_signature(query_shape, value_shape, key_shape)

    def call(self, x):
        x_shape = tf.shape(x)

        x = rearrange(x, 'b h w c -> b (h w) c', h=x_shape[1], w=x_shape[2])

        x_layer_norm = self.layer_norm(x)

        attention_value = self.multi_head_attention(x_layer_norm, x_layer_norm, x_layer_norm)
        attention_value = attention_value + x

        attention_value = self.projection_linear(attention_value) + attention_value
        attention_value = rearrange(attention_value, 'b (h w) c -> b h w c', h=x_shape[1], w=x_shape[2])

        return attention_value
