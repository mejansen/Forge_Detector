# necessary imports
import tensorflow as tf
from CNN import CNN
from Siamese_NN import Siamese_Network
import numpy as np
import config

# load the configuration variables
# data_dir = ?
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
    pass

def main():
    pass



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")

