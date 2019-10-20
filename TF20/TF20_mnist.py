import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

print(tf.__version__)
BATCH_SIZE = 32
EPOCHS = 10
(xs, ys), _ = tf.keras.datasets.mnist.load_data()
print("dataset", xs.shape, ys.shape, xs.min(), xs.max())

xs = tf.convert_to_tensor(xs, dtype=tf.float32) / 255.
db = tf.data.Dataset.from_tensor_slices((xs,ys))
db = db.batch(32).repeat(10)

network = Sequential([layers.Dense(256, activation='relu'),
                     layers.Dense(256, activation='relu'),
                     layers.Dense(256, activation='relu'),
                     layers.Dense(10)])


