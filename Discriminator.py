import tensorflow as tf

class Discriminator(tf.keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.layer_list = [
            tf.keras.layers.Conv2D(16, kernel_size=(3,3), strides=(2,2), padding ="same", activation ='relu'),
            tf.keras.layers.Conv2D(32, kernel_size=(3,3), strides=(2,2), padding ="same", activation ='relu'),
            tf.keras.layers.Conv2D(64, kernel_size=(3,3), strides=(2,2), padding ="same", activation ='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(units = 32, activation= 'relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ]

        self.metric_fake_loss = tf.keras.metrics.Mean(name="fake_loss")
        self.metric_real_loss = tf.keras.metrics.Mean(name="real_loss")
        self.metric_loss = tf.keras.metrics.Mean(name="loss")

        self.metric_real_accuracy = tf.keras.metrics.Accuracy(name="real_accuracy")
        self.metric_fake_accuracy = tf.keras.metrics.Accuracy(name="fake_accuracy")

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    @tf.function
    def call(self, x, training=False):
        for layer in self.layer_list:
            #layer.trainable = training
            x = layer(x)
        return x
