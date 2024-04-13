# necessary imports
import tensorflow as tf
from Siamese_NN import Siamese_Network
from GAN import *
import datetime
import prepare_data
import config
import tqdm
import os
import Log_modul_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.random.set_seed(161)

train_res_los = []
train_res_acc = []
test_res_los = []
test_res_acc = []

def train_siames():
    #
    # create the data
    #
    train_ds, validation_ds = prepare_data.data_prep_siames()

    #
    # create the model
    #
    model = Siamese_Network()

    model.siam.build(input_shape=(1,150,200,1))
    #model.build()

    model.siam.summary()

    for epoch in range(1,config.num_epochs+1):
        for x, y, target in tqdm.tqdm(train_ds):
            model.train_step(x, y, target)
        Log_modul_data.logging_after_train(model)      
        
        for x, y, target in validation_ds:
            model.test_step(x, y, target)
            
        Log_modul_data.logging_after_test(model, epoch)

        if epoch % config.save_model_epoch == 0:
            # Save model (its parameters)
            model.save_weights(f"./saved_models/trained_weights_{epoch}", save_format="tf")
            model.save_weights(f"./saved_models/trained_weights_{epoch}.h5")

    Log_modul_data.show_graph()

def train_gan():
    train_ds, validation_ds = prepare_data.data_prep_gan()

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = f"logs/{current_time}"
    train_summary_writer = tf.summary.create_file_writer(file_path)

    gan = GAN()

    # Build 
    gan.discriminator.build(input_shape=(1, 144, 192, 1))
    gan.generator.build(input_shape=(1, gan.generator.noise_dim))
    
    # Get overview of number of parameters
    gan.discriminator.summary()
    gan.generator.summary()

    # 
    # Evaluation: Take always the same noise for generating images
    # -> better to compare these generated images
    #
    noise = tf.random.uniform(minval=-1, maxval=1, shape=(config.batch_size, gan.generator.noise_dim))

    print("Epoch: 0")

    for epoch in range(1, config.num_epochs + 1):
            
        print(f"Epoch {epoch}")

        for img_real in tqdm.tqdm(train_ds, position=0, leave=True): 
            gan.train_step(img_real)

        Log_modul_data.log_gan(train_summary_writer,gan, noise, epoch)

    return

def main():

    if config.train_gan:
        train_gan()
    else:
        train_siames()
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")

