import numpy
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x, test_x = tf.cast(train_x/255.0, tf.float32), tf.cast(test_x/255.0, tf.float32)


EPOCHS = 10

class MyMode(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(MyMode, self).__init__()
        inputs = tf.keras.Input(shape=(28, 28), name="input")
        self.x0 = tf.keras.layers.Flatten()
        self.x1 = tf.keras.layers.Dense(512, activation=tf.nn.relu, name="d1")
        self.x2 = tf.keras.layers.Dropout(0.2)
        self.predictions = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax, name="prediction")

    def __call__(self, inputs):

        x = self.x0(inputs)
        x = self.x1(x)
        x = self.x2(x)
        return self.predictions(x)

BATCH_SIZE = 512
model4 = MyMode()
step_per_epoch = len(train_x) // BATCH_SIZE

model4.compile(optimiser=tf.keras.optimizers.Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model4.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS)
model4.evaluate(test_x, test_y)