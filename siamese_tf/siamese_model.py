import tensorflow as tf
import numpy as np


class siamese():
    def __init__(self, input_shape):
        print("init")

    def sub_model(self, input_tensor):
        n_chanel = input_tensor.shape[3]
        stddev_ = 0.1

        with tf.name_scope("conv_layer_1") as scope:
            with tf.name_scope("weights") as scope:
                w1_1 = tf.Variable(tf.truncated_normal([10, 10, n_chanel, 64], stddev=stddev_), name="w1_1")
            with tf.name_scope("biases") as scope:
                b1_1 = tf.Variable(tf.constant(0.1, shape=[64]), name="b1_1")

            conv1_1 = tf.nn.conv2d(input=input_tensor,
                                   filter=w1_1,
                                   strides=[1, 1, 1, 1],
                                   padding="SAME")
            conv1_1 += b1_1
            conv1_1 = tf.nn.relu(conv1_1, name="CONV1_1")

        with tf.name_scope("pooling_layer_1") as scope:
            pool1_1 = tf.nn.max_pool(conv1_1,
                                     ksize=[1, 2, 2, 1],
                                     strides=[1, 2, 2, 1],
                                     padding="SAME",
                                     name="POOL1_1")

        with tf.name_scope("conv_layer_2") as scope:
            with tf.name_scope("weights") as scope:
                w2_1 = tf.Variable(tf.truncated_normal([7, 7, 64, 128], stddev=stddev_), name="w2_1")
            with tf.name_scope("biases") as scope:
                b2_1 = tf.Variable(tf.constant(0.1, shape=[128]), name="b2_1")

            conv2_1 = tf.nn.conv2d(input=pool1_1,
                                   filter=w2_1,
                                   strides=[1, 1, 1, 1],
                                   padding="SAME")
            conv2_1 += b2_1
            conv2_1 = tf.nn.relu(conv2_1, name="CONV2_1")

        with tf.name_scope("pooling_layer_2") as scope:
            pool2_1 = tf.nn.max_pool(conv2_1,
                                     ksize=[1, 2, 2, 1],
                                     strides=[1, 2, 2, 1],
                                     padding="SAME",
                                     name="POOL2_1")

        with tf.name_scope("conv_layer_3") as scope:
            with tf.name_scope("weights") as scope:
