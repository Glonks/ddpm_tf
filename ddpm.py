import tensorflow as tf
import numpy as np

from tqdm import tqdm

from network import get_unet
from utils import postprocess_images, plot_images


class DiffusionUtility:
    def __init__(self, beta_start=1e-4, beta_end=2e-2, timesteps=1000):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps

        beta = self.prepare_variance_schedule()

        alpha = 1.0 - beta
        alpha_cum_prod = np.cumprod(alpha, axis=0)

        self.beta = tf.constant(beta, dtype=tf.float32)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.alpha_cum_prod = tf.constant(alpha_cum_prod, dtype=tf.float32)

    def prepare_variance_schedule(self):
        return np.linspace(
            start=self.beta_start,
            stop=self.beta_end,
            num=self.timesteps,
            dtype=np.float64
        )

    @staticmethod
    def extract(tensor, indices):
        out = tf.gather(tensor, indices)
        out = out[:, None, None, None]

        return out

    def forward_diffuse(self, x_start, t):
        noise = tf.random.normal(shape=tf.shape(x_start))

        sqrt_alpha_cum_prod = tf.sqrt(self.extract(self.alpha_cum_prod, t))
        sqrt_one_minus_alpha_cum_prod = tf.sqrt(1 - self.extract(self.alpha_cum_prod, t))

        return sqrt_alpha_cum_prod * x_start + sqrt_one_minus_alpha_cum_prod * noise, noise

    def backward_sample(self, samples, predicted_noise, t):
        beta = tf.sqrt(self.extract(self.beta, t))
        alpha = tf.sqrt(self.extract(self.alpha, t))
        sqrt_alpha = tf.sqrt(self.extract(self.alpha, t))
        sqrt_one_minus_alpha_cum_prod = tf.sqrt(1 - self.extract(self.alpha_cum_prod, t))

        if any(t - 1) > 1:
            noise = tf.random.normal(shape=tf.shape(samples))

        else:
            noise = tf.zeros_like(samples)

        return (1 / sqrt_alpha) * (samples - ((1 - alpha) / sqrt_one_minus_alpha_cum_prod) * predicted_noise) + tf.sqrt(beta) * noise


class DiffusionModel(tf.keras.Model):
    def __init__(self, beta_start, beta_end, timesteps, input_image_shape, use_ema=False, decay=0.999):
        super().__init__()

        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.input_image_shape = input_image_shape
        self.use_ema = use_ema
        self.decay = decay

        self.diffusion_utility = DiffusionUtility(beta_start, beta_end, timesteps)
        self.network = get_unet(input_image_shape)

        if use_ema:
            self.ema_network = get_unet(input_image_shape)
            self.ema_network.set_weights(self.network.get_weights())

    def train_step(self, images):
        batch_size = tf.shape(images)[0]

        t = tf.random.uniform(shape=(batch_size,), minval=1, maxval=self.timesteps, dtype=tf.int64)

        with tf.GradientTape() as tape:
            noised_images, noise = self.diffusion_utility.forward_diffuse(images, t)

            predicted_noise = self.network([noised_images, t], training=True)

            loss = self.loss(noise, predicted_noise)

        gradients = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        if self.use_ema:
            for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
                ema_weight.assign(self.decay * ema_weight + (1 - self.decay) * weight)

        return {'loss': loss}

    def generate_images(self, num_images):
        samples = tf.random.normal(shape=(num_images, *self.input_image_shape))

        network = self.network if not self.use_ema else self.ema_network

        for i in tqdm(reversed(range(1, self.timesteps)), position=0):
            timesteps = tf.ones((num_images,), dtype=tf.int64) * i

            predicted_noise = network.predict([samples, timesteps], verbose=0, batch_size=num_images)

            samples = self.diffusion_utility.backward_sample(samples, predicted_noise, timesteps)

        samples = postprocess_images(samples)

        return samples
