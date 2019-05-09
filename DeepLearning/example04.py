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

