import tensorflow as tf

from tensorflow.keras import layers

from modules import *


def get_unet(image_input_shape):
    image_input = layers.Input(shape=image_input_shape, name='image_input')
    time_input = layers.Input(shape=(), dtype=tf.int64, name='time_input')

    t = SinusoidalPositionEmbedding(256)(time_input)

    # Contraction
    x1 = DoubleConv(64)(image_input)

    x2 = Down([64, 128])(x1, t)
    x2 = SelfAttention(128)(x2)

    x3 = Down([128, 256])(x2, t)
    x3 = SelfAttention(256)(x3)

    x4 = Down([256, 256])(x3, t)
    x4 = SelfAttention(256)(x4)

    # Bottleneck
    x4 = DoubleConv(512)(x4)
    x4 = DoubleConv(512)(x4)
    x4 = DoubleConv(256)(x4)

    # Expansion
    x = Up([512, 128])(x4, x3, t)
    x = SelfAttention(128)(x)

    x = Up([256, 64])(x, x2, t)
    x = SelfAttention(64)(x)

    x = Up([128, 64])(x, x1, t)
    x = SelfAttention(64)(x)

    # Match channels
    x = layers.Conv2D(filters=image_input_shape[-1], kernel_size=1, padding='same')(x)

    return tf.keras.Model(inputs=[image_input, time_input], outputs=x, name='UNet')
