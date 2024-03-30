import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from CNN import CNN
from Siamese_NN import Siamese_Network
import numpy as np
import config
import tqdm
import os
import tempfile

data_dir = config.data_path

def create_dataset_from_Data(create_new = False):
    path = os.path.join(tempfile.gettempdir(), "saved_data")
    

    if(create_new):
        #
        # create the training and validation data set from the files in the directory
        #
        img_height = 150
        img_width = 200
        all_data_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            #validation_split = 0.0,
            #subset = "both",
            #seed = 123,
            image_size = (img_height, img_width),
            batch_size = None)
    
        combined_new_data = tf.data.Dataset
        first = True
        for image, label in tqdm.tqdm(all_data_ds):
            count_wrong = 0
            count_right = 0
            for image_y, label_y in all_data_ds:
                target = tf.equal(label, label_y)
                if target:
                    if first:
                        combined_new_data = tf.data.Dataset.from_tensor_slices(([image], [image_y],[1.0]))
                        first = False
                    else:
                        combined_new_data = combined_new_data.concatenate(tf.data.Dataset.from_tensor_slices(([image], [image_y],[1.0])))
                    count_right += 1

            while count_wrong < count_right:
                for image_y, label_y in all_data_ds.take(1):
                    target = tf.equal(label, label_y)
                    if target == False:
                        combined_new_data = combined_new_data.concatenate(tf.data.Dataset.from_tensor_slices(([image], [image_y],[0.0])))
                        count_wrong += 1

        tf.data.experimental.save(combined_new_data, path)
    else:
        combined_new_data = tf.data.experimental.load(path)
    return combined_new_data