import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from CNN import CNN
from Siamese_NN import Siamese_Network
import numpy as np
import config
import tqdm
import multiprocessing
import os
import tempfile

data_dir = config.data_path
data_dir_forced = config.data_path_forged


def create_dataset_siames():
    path = os.path.join(tempfile.gettempdir(), "saved_data")
    #
    # create the training and validation data set from the files in the directory
    #
    img_height = 150
    img_width = 200

    # we get a sorted and an unsortet dataset to have fewer iterations over the whole data (but it cost memory :() 
    # It reduced the calculation time by a minimum of 3 minutes
    all_data_ds_sortet_in_batches = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        shuffle = False,
        color_mode = "grayscale",
        image_size = (img_height, img_width),
        batch_size = 5
    )

    all_data_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        color_mode = "grayscale",
        image_size = (img_height, img_width),
        batch_size = None)
        
    combined_new_data = tf.data.Dataset
    first = True

    for images, labels in tqdm.tqdm(all_data_ds_sortet_in_batches):
        for image in images:
            for image_to_pair in images:
                if first:
                    combined_new_data = tf.data.Dataset.from_tensor_slices(([image], [image_to_pair],[1.0]))
                    first = False
                else:
                    combined_new_data = combined_new_data.concatenate(tf.data.Dataset.from_tensor_slices(([image], [image_to_pair],[1.0])))
                
            count_wrong = 0
            while count_wrong < 5:
                for image_y, label_y in all_data_ds.take(1):
                    target = tf.equal(labels[0], label_y)
                    if target == False:
                        combined_new_data = combined_new_data.concatenate(tf.data.Dataset.from_tensor_slices(([image], [image_y],[0.0])))
                        count_wrong += 1
    tf.data.experimental.save(combined_new_data, path)



def create_dataset_from_Data(create_new = False):
    path = os.path.join(tempfile.gettempdir(), "saved_data")
    data_is_not_on_disk = False
    
    try:
        combined_new_data = tf.data.experimental.load(path)
    except Exception:
        data_is_not_on_disk = True


    if(create_new or data_is_not_on_disk):
        #using a thread to free the memory after the dataset is createt (there was a few errors that happend sometimes)
        p = multiprocessing.Process(target=create_dataset_siames) 
        p.start() 
        p.join()
        combined_new_data = tf.data.experimental.load(path)

    return combined_new_data

def create_dataset_for_GAN():
    img_height = 150
    img_width = 200

    # we get a sorted and an unsortet dataset to have fewer iterations over the whole data (but it cost memory :() 
    # It reduced the calculation time by a minimum of 3 minutes
    correct_data = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        color_mode = "grayscale",
        image_size = (img_height, img_width),
        batch_size = None
    )

    forced_data = tf.keras.utils.image_dataset_from_directory(
        data_dir_forced,
        color_mode = "grayscale",
        image_size = (img_height, img_width),
        batch_size = None
    )

    combined_data = correct_data.concatenate(forced_data)
    return combined_data
