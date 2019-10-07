import tensorflow as tf

mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()

EPOCH = 10
BATCH_SIZE = 512

train_x, test_x = tf.cast(train_x/255.0, tf.float32), tf.cast(test_x/255.0, tf.float32)
train_y, test_y = tf.cast(train_y, tf.int64), tf.cast(test_y, tf.int64)

model1 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

optimiser = tf.keras.optimizers.Adam()
model1.compile(optimiser=optimiser, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model1.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCH)
model1.evaluate(test_x, test_y)
print(model1.summary())