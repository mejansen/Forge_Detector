import tensorflow as tf
from CNN import CNN

class Siamese_Network(tf.keras.Model):
    def __init__(self, name : str):
        super().__init__()
        self.name = name
        self.siam_one = CNN(output_size = 10, name = "Siamese_Network_One")
        self.siam_two = CNN(output_size = 10, name = "Siamese_Network_Two")

        self.loss_object = tf.keras.losses.contrastive_loss()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)



    def call(self, sig_1, sig_2):
        latent_rep_1 = self.siam_one.call(sig_1)
        latent_rep_2 = self.siam_two.call(sig_2)

        return latent_rep_1, latent_rep_2

    def train_step(self, img_1, img_2):
        # let's keep the gradient tape persistent
        with tf.GradientTape(persistent = True) as tape:
            rep_1, rep_2 = self.call(img_1, img_2)
            loss = self.loss_object(rep_1, rep_2)

        # read out the gradients
        gradients_model_1 = tape.gradient(loss, self.siam_one.trainable_variables)
        gradients_model_2 = tape.gradient(loss, self.siam_two.trainable_variables)

        # apply the gradients
        self.optimizer.apply_gradients(zip(gradients_model_1, self.siam_one.trainable_variables))
        self.optimizer.apply_gradients(zip(gradients_model_2, self.siam_two.trainable_variables))

        # discard the tape? I don't yet know how this works :'(

