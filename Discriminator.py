import tensorflow as tf
import config

class Discriminator(tf.keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.layer_list = [
            tf.keras.layers.Conv2D(filters = 16, kernel_size = (3, 3), padding = "same", activation = "relu"),#shape [batch_size,150,200,32]
            tf.keras.layers.Conv2D(filters = 16, kernel_size = (3, 3), padding = "same", activation = "relu"),#shape [batch_size,150,200,32]
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), padding = "same", activation = "relu"),#shape [batch_size,75,100,64]
            tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), padding = "same", activation = "relu"),#shape [batch_size,75,100,64]
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu"),#shape [batch_size,37,50,128]
            tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu"),#shape [batch_size,37,50,128]
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), padding = "same", activation = "relu"),#shape [batch_size,18,25,256]
            tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), padding = "same", activation = "relu"),#shape [batch_size,18,25,256]
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(units = 1024, activation = "sigmoid"),
            tf.keras.layers.Dense(units = 1, activation = "sigmoid")

        ]

        self.metric_fake_loss = tf.keras.metrics.Mean(name="fake_loss")
        self.metric_real_loss = tf.keras.metrics.Mean(name="real_loss")
        self.metric_loss = tf.keras.metrics.Mean(name="loss")

        self.metric_real_accuracy = tf.keras.metrics.Accuracy(name="real_accuracy")
        self.metric_fake_accuracy = tf.keras.metrics.Accuracy(name="fake_accuracy")

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

    @tf.function
    def call(self, x, training=False):
        for layer in self.layer_list:
            #layer.trainable = training
            x = layer(x)
        return x
