# necessary imports
import tensorflow as tf
from Siamese_NN import Siamese_Network
import matplotlib.pyplot as plt
from GAN import *
import datetime
import prepare_data
import config
import tqdm
import os
import Log_modul_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


epoch_to_load = 80

def main():
    img_height = 150
    img_width = 200

    # we get a sorted and an unsortet dataset to have fewer iterations over the whole data (but it cost memory :() 
    # It reduced the calculation time by a minimum of 3 minutes
    correct_data = tf.keras.utils.image_dataset_from_directory(
        "test_image",
        color_mode = "grayscale",
        image_size = (img_height, img_width),
        batch_size = None
    )

    combined_new_data = tf.data.Dataset
    first = True

    for image, labels in correct_data:
        for image_y, label_y in correct_data:
            if tf.equal(labels, label_y) == False:
                if first:
                    combined_new_data = tf.data.Dataset.from_tensor_slices(([image], [image_y],[1.0]))
                    first = False
                else:
                    combined_new_data = combined_new_data.concatenate(tf.data.Dataset.from_tensor_slices(([image], [image_y],[1.0])))


    combined_new_data = combined_new_data.map(lambda img_a, img_b, target: prepare_data.preprocessing_func_no_noise(img_a, img_b, target))

    # shuffle, batch, and prefetch
    combined_new_data = combined_new_data.batch(2).prefetch(tf.data.experimental.AUTOTUNE)


    model = Siamese_Network()

    model.siam.build(input_shape=(1,150,200,1))

    #it's the worst bugfix if ever done. Pls don't hate me :/
    #the model won't load and the error said the model must be trained
    #so we doing on train step to final initalise the model to load the weigts 
    for x, y, target in combined_new_data:
        model.train_step(x,y,target)
        break
    
    model.load_weights(f"saved_models/trained_weights_{epoch_to_load}.h5")
    for x, y, target in combined_new_data:
        prediction = model.call(x,y)
        fig = plt.figure()
        ax1 = fig.add_subplot(2,2,1)
        ax1.imshow(x[0], cmap='gray')
        ax1.set_title(str(prediction[0]))
        ax2 = fig.add_subplot(2,2,2)
        ax2.imshow(y[0], cmap='gray')
        ax3 = fig.add_subplot(2,2,3)
        ax3.imshow(x[1], cmap='gray')
        ax3.set_title(str(prediction[1]))
        ax4 = fig.add_subplot(2,2,4)
        ax4.imshow(y[1], cmap='gray')
        plt.show()
    
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")