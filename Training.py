# necessary imports
import tensorflow as tf
from Siamese_NN import Siamese_Network
import create_dataset
import config
import tqdm
import os
import Log_modul_data

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



def data_prep(shuffle_buffer_size, batch_size):
    """This function shall at some point take the directory path where the data is located, as well as some guiding variables. It shall prepare the data and convert it to a tf.Dataset object.
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

def main():

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    #
    # create the data
    #
    train_ds, validation_ds = data_prep( shuffle_buffer_size, batch_size)

    #
    # create the model
    #
    model = Siamese_Network()

    model.siam.build(input_shape=(1,150,200,1))
    #model.build()

    model.siam.summary()

    for epoch in range(num_epochs):
        for x, y, target in tqdm.tqdm(train_ds):
            model.train_step(x, y, target)
        Log_modul_data.logging_after_train(model)      
        
        for x, y, target in validation_ds:
            model.test_step(x, y, target)
            
        Log_modul_data.logging_after_test(model, epoch)

        if epoch % config.save_model_epoch == 0:
            # Save model (its parameters)
            model.save_weights(f"./saved_models/trained_weights_{epoch}", save_format="tf")

    Log_modul_data.show_graph()

    for x, y, target in validation_ds:
        prediction = model.call(x,y)
        print(f"Prediction: {prediction} Target: {target}")
        break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")

