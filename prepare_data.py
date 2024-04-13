import tensorflow as tf
import create_dataset
import config

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

def preprocessing_func_no_noise(img_a, img_b, label):
    img_a = tf.cast(img_a, tf.float32)
    img_a = (img_a/128) - 1
    # adding some random noise to perform data augmentation

    img_b = tf.cast(img_b, tf.float32)
    img_b = (img_b/128) - 1
    # adding some random noise to perform data augmentation
    return img_a, img_b, label

def data_prep_siames():
    """This function shall at some point take the directory path where the data is located, as well as some guiding variables. It shall prepare the data and convert it to a tf.Dataset object.
    ----------------------------------
    @out: tf.Dataset object
    """

    all_data = create_dataset.create_dataset_from_Data(False).shuffle(config.shuffle_buffer_size)
    count = len(list(all_data))
    print(f"Amount of Data: {count}")
    validation_ds = all_data.take((int)(count * config.data_split))
    train_ds = all_data.skip((int)(count * config.data_split))
    print(f"Amount of Data: {len(list(train_ds))} + {len(list(validation_ds))}")

    #for image, label in validation_ds:

    # rescale the values to a range of -1 and 1 for both data sets
    
    train_ds = train_ds.map(lambda img_a, img_b, target: preprocessing_func(img_a, img_b, target))

    validation_ds = validation_ds.map(lambda img_a, img_b, target: preprocessing_func(img_a, img_b, target))

    # shuffle, batch, and prefetch
    train_ds = train_ds.shuffle(config.shuffle_buffer_size).batch(config.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    validation_ds = validation_ds.shuffle(config.shuffle_buffer_size).batch(config.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return train_ds, validation_ds # finished

def data_prep_gan():
    all_data = create_dataset.create_dataset_for_GAN().shuffle(config.shuffle_buffer_size)
    count = len(list(all_data))
    print(f"Amount of Data  for GAN: {count}")
    validation_ds = all_data.take((int)(count * config.data_split))
    train_ds = all_data.skip((int)(count * config.data_split))
    print(f"Amount of Data: {len(list(train_ds))} + {len(list(validation_ds))}")


    train_ds = train_ds.apply(prepare_dataset_for_gan)
    validation_ds = validation_ds.apply(prepare_dataset_for_gan)

    return train_ds, validation_ds



def prepare_dataset_for_gan(dataset):

    # Only '0' digits
    dataset = dataset.filter(lambda img, label: label == 0) 

    # Remove label
    dataset = dataset.map(lambda img, label: img)

    # Resize into (32, 32)
    #dataset = dataset.map(lambda img: tf.image.resize(img, [144,192]) )

    # Convert data from uint8 to float32
    dataset = dataset.map(lambda img: tf.cast(img, tf.float32) )

    # Normalization: [0, 255] to [-1, 1]
    dataset = dataset.map(lambda img: (img/128.)-1. )

    # Cache
    dataset = dataset.cache()
    
    #
    # Shuffle, batch, prefetch
    #
    dataset = dataset.shuffle(config.shuffle_buffer_size)
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
