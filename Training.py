# necessary imports
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from CNN import CNN
from Siamese_NN import Siamese_Network
import numpy as np
import config
import tqdm
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# load the configuration variables
data_dir = config.data_path
num_epochs = config.num_epochs
weight_saving_dir = config.weight_saving
batch_size = config.batch_size
shuffle_buffer_size = config.shuffle_buffer_size
prefetch_size = config.prefetch_size

train_res_los = []
train_res_acc = []



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

    #
    # create the training and validation data set from the files in the directory
    #
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


    # rescale the values to a range of -1 and 1 for both data sets
    def preprocessing_func(img, label):
        img = tf.cast(img, tf.float32)
        img = (img/128) - 1
        return img, label

    train_ds = train_ds.map(lambda img, target: preprocessing_func(img, target))
    validation_ds = validation_ds.map(lambda img, target: preprocessing_func(img, target))

    # shuffle, batch, and prefetch
    train_ds = train_ds.shuffle(shuffle_buffer_size).prefetch(prefetch_size)
    validation_ds = validation_ds.shuffle(shuffle_buffer_size).prefetch(prefetch_size)

    return train_ds, validation_ds # finished

def log_training(network, epoch):

    if epoch != 0:
        epoch_loss = network.metric_loss.result()
        epoch_acc = network.metric_accuracy.result()
        tf.summary.scalar(f"loss", epoch_loss, step=epoch)
        tf.summary.scalar(f"accuracy", epoch_acc, step=epoch)
        print(f"Accurracy: {epoch_acc}, Loss: {epoch_loss}")
        train_res_los.append(epoch_loss.numpy())
        train_res_acc.append(epoch_acc.numpy())

        network.metric_loss.reset_states()
        network.metric_accuracy.reset_states()

def main():

    #
    # create the data
    #
    train_ds, validation_ds = data_prep(data_dir, shuffle_buffer_size, batch_size, prefetch_size)

    #
    # create the model
    #
    model = Siamese_Network()

    for epoch in range(num_epochs):
        # for every step, we always need two samples from which the targets will be constructed
        for x, target_x in tqdm.tqdm(train_ds):
            for y, target_y in train_ds:
                target = tf.equal(target_x, target_y)
                target = tf.cast(target, tf.int32) # <- this is our finished target!

                model.train_step(x, y, target)
        log_training(model, epoch)
        print("Epoch " + str(epoch) + " finished")
    
    plotting = {}

    plotting["train"] = [train_res_los, train_res_acc]

    fig, axs = plt.subplots(2, 2, figsize=(15, 20))
    for ax, key in zip(axs.flat, plotting.keys()):
    
        train_l, train_a = plotting[key]
    
        line1, = ax.plot(train_l)
        line2, = ax.plot(train_a)
        ax.legend((line1,line2),("training loss", "train accuracy"))
        ax.set_title(key)
        ax.set(xlabel="epochs", ylabel="Loss/Accuracy")
        #ax.label_outer()
    
    plt.show()  
    




if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")

