import tensorflow as tf
from CNN import CNN

class Siamese_Network(tf.keras.Model):
    def __init__(self, name : str):
        super().__init__()
        self.name = name
        self.siam_one = CNN(output_size = 10, name = "Siamese_Network_One")
        self.siam_two = CNN(output_size = 10, name = "Siamese_Network_Two")

        # add some dense layers on top that end in a sigmoid activation with only one unit
        self.dense_layers = [tf.keras.layers.Dense(units = 128, activation = 'relu'),
                             tf.keras.layers.Dense(units = 256, activation = 'relu'),
                             tf.keras.layers.Dense(units = 128, activation = 'relu'),
                             tf.keras.layers.Dense(units = 1, activation = "sigmoid")]

        self.loss_object = tf.keras.losses.BinaryCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)



    def call(self, sig_1, sig_2):
        latent_rep_1 = self.siam_one.call(sig_1)
        latent_rep_2 = self.siam_two.call(sig_2)

        x = tf.concat([latent_rep_1, latent_rep_2], axis = 1)
        for layer in self.dense_layers:
            x = layer(x)


        return x

    def train_step(self, img_1, img_2, target):
        # let's keep the gradient tape persistent
        with tf.GradientTape(persistent = True) as tape:
            pred = self.call(img_1, img_2)
            loss = self.loss_object(target, pred)

        # read out the gradients
        gradients = tape.gradient(loss, self.trainable_variables)

        # apply the gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # discard the tape? I don't yet know how this works :'(

