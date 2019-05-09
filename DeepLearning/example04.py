import tensorflow as tf
import os

# Load mnist dataset
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets("MNIST_data", one_hot=True)

train_mnist_data = mnist_data.train
test_mnist_data = mnist_data.test
valid_mnist_data = mnist_data.validation

LOG_DIR = "./tmp"
if os.path.isdir(LOG_DIR) == False:
    os.mkdir(LOG_DIR)


def add_variables_summary(tf_variable, summary_name):
    with tf.name_scope(summary_name + "_summary"):
        mean = tf.reduce_mean(tf_variable)
        tf.summary.scalar("Mean", mean)
        with tf.name_scope("standard_deviation"):
            standard_deviation = tf.sqrt(tf.reduce_mean(tf.square(tf_variable - mean)))

    tf.summary.scalar("standarDeviation", standard_deviation)
    tf.summary.scalar("Maximum", tf.reduce_max(tf_variable))
    tf.summary.scalar("Minimum", tf.reduce_min(tf_variable))
    tf.summary.histogram("Histogram", tf_variable)

height = 28
width = 28
input_size = height * width
no_classes = train_mnist_data.labels.shape[1]

x_input = tf.placeholder(tf.float32, shape=[None, input_size], name='x_input')
y_input = tf.placeholder(tf.float32, shape=[None, no_classes], name='y_input')

x_input_reshape = tf.reshape(x_input, shape=[-1, height, width, 1], name='x_input_reshape')

convolution_layer_1 = tf.layers.conv2d(inputs=x_input_reshape,
                                       filters=64,
                                       kernel_size=[3, 3],
                                       activation=tf.nn.relu, name='convolution_layer_1')
add_variables_summary(convolution_layer_1, 'convolution')

pooling_layer_1 = tf.layers.max_pooling2d()
