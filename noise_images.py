import numpy as np
import tensorflow as tf

from argparse import ArgumentParser

from ddpm import DiffusionUtility
from utils import preprocess_dataset, postprocess_images, plot_images


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--dataset', type=str, required=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=args.dataset,
        image_size=(args.image_size, args.image_size)
    ).map(preprocess_dataset)

    test_image = next(iter(dataset.take(1)))[0]

    diffusion_utility = DiffusionUtility()

    timesteps = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    timesteps_tensor = tf.convert_to_tensor(timesteps, dtype=tf.int64)

    noised_images, _ = diffusion_utility.forward_diffuse(test_image, timesteps_tensor)
    noised_images = postprocess_images(noised_images)

    noised_images = np.concatenate([postprocess_images(test_image)[None, :], noised_images], axis=0)

    plot_images(noised_images, 'noised_images.png')
