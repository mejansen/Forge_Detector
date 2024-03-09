import tensorflow as tf

class CNN(tf.keras.Model):
    def __init__(self, output_size=10, name = "network"): # we want to have a ten_dimensional representation of the signature for comparison
        super().__init__()
        self.name = name

        self.cnn_layers = [
            tf.keras.layers.Conv2D(filters = 16, kernel_size = (3, 3), strides = (2, 2), padding = "same", activation = "relu"),
            tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = (2, 2), padding = "same", activation = "relu"),
            tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), strides = (2, 2), padding = "same", activation = "relu"),
            tf.keras.layers.Conv2D(filters = 256, kernel_size = (3, 3), strides = (2, 2), padding = "same", activation = "relu"),
            tf.keras.layers.GobalAveragePool(),
            tf.keras.layers.Dense(units = 10, activation = "relu")
        ]


    def call(self, x):
        for layer in self.cnn_layers:
            x = layer(x)
        return x