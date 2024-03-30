# necessary imports
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from CNN import CNN
from Siamese_NN import Siamese_Network
import create_dataset
import numpy as np
import config
import tqdm
import os
import tempfile

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
test_res_los = []
test_res_acc = []



def data_prep(directory_path, shuffle_buffer_size, batch_size, prefetch_size):
    """This function shall at some point take the directory path where the data is located, as well as some guiding variables. It shall prepare the data and convert it to a tf.Dataset object.

    @param directory_path: (string) directory
    @param shuffle_buffer_size: (int) how many items are shuffled before batching
    @param batch_size: (int) determines the size of the training examples
    @param prefetch_size: (int) determines how many training examples are always kept ready
    ----------------------------------
    @out: tf.Dataset object
    """

    all_data = create_dataset.create_dataset_from_Data(False).shuffle(shuffle_buffer_size)
    count = len(list(all_data))
    print(f"Amount of Data: {count}")
    validation_ds = all_data.take(300)
    train_ds = all_data.skip(300)

    #for image, label in validation_ds:

    # rescale the values to a range of -1 and 1 for both data sets
    def preprocessing_func(img_a, img_b, label):
        img_a = tf.cast(img_a, tf.float32)
        img_a = (img_a/128) - 1
        # adding some random noise to perform data augmentation
        noise = tf.random.normal(shape = tf.shape(img_a), mean = 0.0, stddev = 1, dtype = tf.float32)
        img_a = tf.add(img_a, noise)

        img_b = tf.cast(img_b, tf.float32)
        img_b = (img_b/128) - 1
        # adding some random noise to perform data augmentation
        noise = tf.random.normal(shape = tf.shape(img_b), mean = 0.0, stddev = 1, dtype = tf.float32)
        img_b = tf.add(img_b, noise)
        return img_a, img_b, label

    train_ds = train_ds.map(lambda img_a, img_b, target: preprocessing_func(img_a, img_b, target))

    validation_ds = validation_ds.map(lambda img_a, img_b, target: preprocessing_func(img_a, img_b, target))

    # shuffle, batch, and prefetch
    train_ds = train_ds.shuffle(shuffle_buffer_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    validation_ds = validation_ds.shuffle(shuffle_buffer_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return train_ds, validation_ds # finished

def log_training(network, epoch):
    epoch_train_loss = network.metric_train_loss.result()
    epoch_train_acc = network.metric_train_accuracy.result()
    epoch_test_loss = network.metric_test_loss.result()
    epoch_test_acc = network.metric_test_accuracy.result()

    print(f"in Epoch " + str(epoch) + f" Train accurracy: {epoch_train_acc}, loss: {epoch_train_loss} - Test accuracy:{epoch_test_acc} , loss {epoch_test_loss}")

    train_res_los.append(epoch_train_loss.numpy())
    train_res_acc.append(epoch_train_acc.numpy())
    test_res_los.append(epoch_test_loss.numpy())
    test_res_acc.append(epoch_test_acc.numpy())
    
    network.metric_train_loss.reset_states()
    network.metric_train_accuracy.reset_states()
    network.metric_test_loss.reset_states()
    network.metric_test_accuracy.reset_states()

def show_graph(plotting):
    fig, axs = plt.subplots(2, 1, figsize=(15, 20))
    for ax, key in zip(axs.flat, plotting.keys()):
    
        train_l, train_a, test_l, test_a = plotting[key]
    
        line1, = ax.plot(train_l)
        line2, = ax.plot(train_a)
        line3, = ax.plot(test_l)
        line4, = ax.plot(test_a)
        ax.legend((line1,line2, line3, line4),("training loss", "train accuracy", "test loss", "test accuracy"))
        ax.set_title(key)
        ax.set(xlabel="epochs", ylabel="Loss/Accuracy")
        #ax.label_outer()
    
    plt.show()  

def main():

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

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
        for x, y, target in tqdm.tqdm(train_ds):
            model.train_step(x, y, target)

                
        
        for x, y, target in validation_ds:
            model.test_step(x, y, target)

        log_training(model, epoch)

    plotting = {}

    plotting["train"] = [train_res_los, train_res_acc, test_res_los, test_res_acc]

    show_graph(plotting)

    for x, y, target in validation_ds:
                prediction = model.call(x,y)
                print(f"Prediction: {prediction} Target: {target}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")

