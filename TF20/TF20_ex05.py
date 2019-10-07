import tensorflow as tf

mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x, test_x = tf.cast(train_x/255.0, tf.float32), tf.cast(test_x/255.0, tf.float32)

EPOCHS = 10

inputs = tf.keras.Input(shape=(28, 28))
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(512, activation="relu", name='d1')(x)
x = tf.keras.layers.Dropout(0.2)(x)
prediction = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name='d2')(x)

model3 = tf.keras.Model(inputs=inputs, outputs=prediction)
print(model3.summary())

optimiser = tf.keras.optimizers.Adam()
model3.compile(optimiser=optimiser, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model3.fit(train_x, train_y, epochs=EPOCHS, batch_size=512)
model3.evaluate(test_x, test_y)
