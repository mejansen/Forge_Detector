import tensorflow as tf

from Generator import *
from Discriminator import *

class GAN(tf.keras.Model):

    def __init__(self):
        super(GAN, self).__init__()

        self.generator = Generator()
        self.discriminator = Discriminator()

        self.bce_loss = tf.keras.losses.BinaryCrossentropy()


    @tf.function
    def train_step(self, img_real):
        pass
    
        #
        # Train
        #

        batch_size_info = tf.shape(img_real)
        batch_size_info[0] # batch size
        noise = tf.random.uniform((batch_size_info[0],100), minval = -1, maxval = 1)

        with tf.GradientTape(persistent=True) as tape:
            fake_img = self.generator.call(noise)
            rating_fake = self.discriminator.call(fake_img, True)
            generator_loss = self.bce_loss(tf.ones_like(rating_fake), rating_fake)
            
            rating_real = self.discriminator.call(img_real, True)  

            discriminator_real_loss = self.bce_loss(tf.ones_like(rating_real), rating_real)
            discriminator_fake_loss = self.bce_loss(tf.zeros_like(rating_fake),rating_fake)

            discriminator_loss = discriminator_fake_loss + discriminator_real_loss

        #
        # Update metrices
        #
        gradients = tape.gradient(generator_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        gradients = tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
        self.discriminator.optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        # Generator
        self.generator.metric_loss.update_state(generator_loss)

        # Discriminator

        # Loss
        self.discriminator.metric_loss.update_state(discriminator_loss)
        self.discriminator.metric_fake_loss.update_state(discriminator_fake_loss)
        self.discriminator.metric_real_loss.update_state(discriminator_real_loss)

        # Accuracy
        classified_fake = tf.math.round(rating_fake)
        classified_real = tf.math.round(rating_real)

        self.discriminator.metric_fake_accuracy.update_state(tf.zeros_like(classified_fake), classified_fake)
        self.discriminator.metric_real_accuracy.update_state(tf.ones_like(classified_real), classified_real)