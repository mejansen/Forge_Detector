# necessary imports
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from CNN import CNN
from Siamese_NN import Siamese_Network
import numpy as np
import config

# load the configuration variables
data_dir = config.data_path
weight_saving_dir = config.weight_saving
batch_size = config.batch_size
shuffle_buffer_size = config.shuffle_buffer_size
prefetch_size = config.prefetch_size



def data_prep(directory_path, shuffle_buffer_size, batch_size, prefetch_size):
    """This function shall at some point take the directory path where the data is located, as well as some guiding variables. It shall prepare the data and convert it to a tf.Dataset object.

    @param directory_path: (string) directory
    @param shuffle_buffer_size: (int) how many items are shuffled before batching
    @param batch_size: (int) determines the size of the training examples
    @param prefetch_size: (int) determines how many training examples are always kept ready
    ----------------------------------
    @out: tf.Dataset object
    """

    img_height = 150
    img_width = 200

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split = 0.2,
        subset = "training",
        seed = 123,
        image_size = (img_height, img_width),
        batch_size = batch_size)

    validation_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split = 0.2,
        subset = "validation",
        seed = 123,
        image_size = (img_height, img_width),
        batch_size = batch_size)

    # rescale the values to a range of -1 and 1
    def preprocessing_func(img, label):
        img = tf.cast(img, tf.float32)
        img = (img/128) - 1
        return img, label

    train_ds = train_ds.map(lambda img, target: preprocessing_func(img, target))
    validation_ds = validation_ds.map(lambda img, target: preprocessing_func(img, target))

    # shuffle, batch, and prefetch
    train_ds = train_ds.shuffle(shuffle_buffer_size).prefetch(prefetch_size)
    validation_ds = validation_ds.shuffle(shuffle_buffer_size).prefetch(prefetch_size)

    return train_ds, validation_ds

def main():
    train_ds, validation_ds = data_prep(data_dir, shuffle_buffer_size, batch_size, prefetch_size)
    print("Done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")

