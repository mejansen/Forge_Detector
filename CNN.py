import tensorflow as tf

class CNN(tf.keras.Model):
    def __init__(self, output_size=128): # we want to have a ten_dimensional representation of the signature for comparison
        super().__init__()

        self.cnn_layers = [
            tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), padding = "same", activation = "relu"),
            tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), padding = "same", activation = "relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu"),
            tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), padding = "same", activation = "relu"),
            tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), padding = "same", activation = "relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(units = output_size, activation = "sigmoid")
        ]


    def call(self, x):
        for layer in self.cnn_layers:
            x = layer(x)
        return x