import tensorflow as tf

mnist = tf.keras.datasets.mnist

(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x, test_x = tf.cast(train_x/255.0, tf.float32), tf.cast(test_x/255.0, tf.float32)
train_y, test_y = tf.cast(train_y, tf.int64), tf.cast(test_y, tf.int64)

BATCH_SIZE = 512
EPOCH = 10
model2 = tf.keras.models.Sequential()
model2.add(tf.keras.layers.Flatten())
model2.add(tf.keras.layers.Dense(512, activation="relu"))
model2.add(tf.keras.layers.Dropout(0.2))
model2.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model2.compile(optimiser=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metric=['accuracy'])

model2.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCH)
model2.evaluate(test_x, test_y)
