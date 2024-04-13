import tensorflow as tf
import config

class Generator(tf.keras.Model):

    def __init__(self):
        super(Generator, self).__init__()

        self.layer_list = [
            tf.keras.layers.Dense(3*4*256, input_shape = (100,)),
            tf.keras.layers.Reshape((3,4,256)), #3x4 128
            tf.keras.layers.Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', use_bias = False), # 6x8 64
            tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', use_bias = False), # 12x16 32
            tf.keras.layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', use_bias = False), # 24x32 32
            tf.keras.layers.Conv2DTranspose(32, (4,4), strides=(2,2), padding='same', use_bias = False), # 48x64 32
            tf.keras.layers.Conv2DTranspose(16, (4,4), strides=(3,3), padding='same', use_bias = False), # 144x192 
            tf.keras.layers.Conv2DTranspose(1, (4,4), padding='same', use_bias = False, activation='tanh') # 144x192 
        ]

        self.metric_loss = tf.keras.metrics.Mean(name="loss")
     
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

        self.noise_dim = 100

    @tf.function
    def call(self, x, training=False):
        for layer in self.layer_list:
            x = layer(x)
        return x
