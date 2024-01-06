import tensorflow as tf

from argparse import ArgumentParser

from ddpm import DiffusionModel
from utils import preprocess_dataset, GenerateAndPlotCallback


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--decay', type=float, default=0.999)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--beta_start', type=float, default=1e-4)
    parser.add_argument('--beta_end', type=float, default=2e-2)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)

    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--dataset', type=str, required=True)

    parser.add_argument('--num_images', type=int, default=16)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=args.dataset,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size)
    ).map(preprocess_dataset)

    model = DiffusionModel(
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        timesteps=args.timesteps,
        input_image_shape=(args.image_size, args.image_size, 3),
        decay=args.decay
    )

    model.compile(
        loss=tf.keras.losses.MeanAbsoluteError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
    )

    model.fit(
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[GenerateAndPlotCallback(args.num_images)]
    )
