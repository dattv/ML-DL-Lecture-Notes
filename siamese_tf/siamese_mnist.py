import os

import numpy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

root_path = os.path.dirname(os.path.dirname(__file__))

siamese_path = os.path.join(root_path, "siamese_tf")

mnist_path = os.path.join(siamese_path, "MNIST_data")
# if os.path.isdir(mnist_path) == False:
mnist_data = input_data.read_data_sets("MNIST_data", one_hot=True)

input_size = 784
no_classes = 10
batch_size = 1
total_batch = 300000

def add_variable_summary(tf_variable, summary_name):
    with tf.name_scope(summary_name + '_summary'):
        mean = tf.reduce_mean(tf_variable)
        tf.summary.scalar('Mean', mean)
        with tf.name_scope('standard_deviation'):
            standard_deviation = tf.sqrt(tf.reduce_mean(
                tf.square(tf_variable - mean)))
        tf.summary.scalar('StandardDeviation', standard_deviation)
        tf.summary.scalar('Maximum', tf.reduce_max(tf_variable))
        tf.summary.scalar('Minimum', tf.reduce_min(tf_variable))
        tf.summary.histogram('Histogram', tf_variable)

init_bias_value = 0.1
nChanel = 1
stddev = 0.1



def main():
    print("djkfljd")

if __name__ == '__main__':
    main()