import tensorflow as tf
import os


# Load MNIST data directly from TensorFlow
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets("MNIST_data", one_hot=True)

train_mnist_data = mnist_data.train
test_mnist_data = mnist_data.test
valid_mnist_dat = mnist_data.validation

height = 28
width = 28

input_size = height * width

no_classes = train_mnist_data.labels.shape[1]
batch_size = 100
total_batches = 5000

x_input = tf.placeholder(tf.float32, shape=[None, input_size], name='x_input')
y_input = tf.placeholder(tf.float32, shape=[None, no_classes], name='y_input')

weights = tf.Variable(tf.random_normal([input_size, no_classes]), name='weights')
bias = tf.Variable(tf.random_normal([no_classes]), name='bias')

logits = tf.add(tf.matmul(x_input, weights), bias, name="logits")

softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=logits, name='cross_entropy')

loss_operation = tf.reduce_mean(softmax_cross_entropy, name='loss')

optimiser = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss_operation)

session = tf.Session()

session.run(tf.global_variables_initializer())


for batch_no in range(total_batches):
    mnist_batch = train_mnist_data.next_batch(batch_size)
    _, loss_value = session.run([optimiser, loss_operation], feed_dict={x_input: mnist_batch[0],
                                                                        y_input: mnist_batch[1]})

    if batch_no % 100 == 0:
        print("step: {}, trainning accuracy: {}".format(batch_no, loss_value))

predictions = tf.argmax(logits, 1)

correct_predictions = tf.equal(predictions, tf.argmax(y_input, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
test_images, test_lables = test_mnist_data.images, test_mnist_data.labels

accuracy_value = session.run(accuracy_operation, feed_dict={x_input: test_images,
                                                            y_input: test_lables})

print("Accuracy: {}".format(accuracy_value))
session.close()


