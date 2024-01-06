import tensorflow as tf
import matplotlib.pyplot as plt


class GenerateAndPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, num_images):
        super().__init__()

        self.num_images = num_images

    def on_epoch_end(self, epoch, logs=None):
        images = self.model.generate_images(self.num_images)

        plot_images(images, f'generated_epoch_{epoch}.png')


def preprocess_dataset(x, y):
    x = x / 127.5 - 1
    x = tf.clip_by_value(x, -1.0, 1.0)

    return x


def postprocess_images(x):
    x = tf.clip_by_value(x, -1.0, 1.0)

    x = (x + 1) * 127.5
    x = tf.clip_by_value(x, -255.0, 255.0)

    x = x.numpy().astype('uint8')

    return x


def plot_images(image_tensor, path, figsize=(15, 3)):
    figure, axes = plt.subplots(nrows=1, ncols=image_tensor.shape[0], figsize=figsize)

    for i in range(image_tensor.shape[0]):
        axes[i].imshow(image_tensor[i])
        axes[i].axis('off')

    plt.tight_layout()

    figure.savefig(path, bbox_inches='tight')
    figure.clf()
    
    plt.close()
