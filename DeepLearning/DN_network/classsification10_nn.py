import tensorflow as tf
import numpy as np


class classification10_nn:
    def __init__(self, input_tensor, keep_prop_tensor, C=[30, 50, 80, 500, 10]):
        self.C = C
        self.keep_prop_tensor = keep_prop_tensor
        self.input_tensor = input_tensor
        self.shape_input = input_tensor.shape
        self.output = self.forward()

    def forward(self):
        with tf.variable_scope("layer_1") as scope:
            with tf.variable_scope("convolution_1_1") as scope:
                with tf.variable_scope("weights") as scope:
                    w1_1 = tf.get_variable("w1_1", shape=[3, 3, self.shape_input[3], self.C[0]],
                                           initializer=tf.contrib.layers.xavier_initializer())

                with tf.variable_scope("biases") as scope:
                    b1_1 = tf.get_variable("b1_1", shape=[self.C[0]],
                                           initializer=tf.constant_initializer(0.))

                conv1_1 = tf.nn.conv2d(input=self.input_tensor,
                                       filter=w1_1,
                                       strides=[1, 1, 1, 1],
                                       padding="SAME",
                                       name="conv1_1")

                conv1_1 = tf.nn.bias_add(conv1_1, b1_1, name="bias_add")
                conv1_1 = tf.nn.relu(conv1_1, name="relu")

            with tf.variable_scope("convolution_1_2") as scope:
                with tf.variable_scope("weights") as scope:
                    w1_2 = tf.get_variable("w1_2", shape=[3, 3, self.C[0], self.C[0]],
                                           initializer=tf.contrib.layers.xavier_initializer())

                with tf.variable_scope("biases") as scope:
                    b1_2 = tf.get_variable("b1_2", shape=[self.C[0]],
                                           initializer=tf.constant_initializer(0.))

                conv1_2 = tf.nn.conv2d(input=conv1_1,
                                       filter=w1_2,
                                       strides=[1, 1, 1, 1],
                                       padding="SAME",
                                       name="conv1_2")
                conv1_2 = tf.nn.bias_add(conv1_2, b1_2, name="bias_add")
                conv1_2 = tf.nn.relu(conv1_2, name="relu")

            with tf.variable_scope("convolution_1_3") as scope:
                with tf.variable_scope("weights") as scope:
                    w1_3 = tf.get_variable('w1_3', shape=[3, 3, self.C[0], self.C[0]],
                                           initializer=tf.contrib.layers.xavier_initializer())
                with tf.variable_scope("biases") as scope:
                    b1_3 = tf.get_variable('b1_3', shape=[self.C[0]],
                                           initializer=tf.constant_initializer(0.))

                conv1_3 = tf.nn.conv2d(input=conv1_2,
                                       filter=w1_3,
                                       strides=[1, 1, 1, 1],
                                       padding='SAME',
                                       name="conv1_3")

                conv1_3 = tf.nn.bias_add(conv1_3, b1_3, name="bias_add")
                conv1_3 = tf.nn.relu(conv1_3, name='relu')

            with tf.variable_scope("maxpooling_1_1") as scope:
                conv1_pool = tf.nn.max_pool(conv1_3,
                                            ksize=[1, 2, 2, 1],
                                            strides=[1, 2, 2, 1],
                                            padding='SAME',
                                            name="conv1_pool")

            with tf.variable_scope("dropout_1_1") as scope:
                conv1_drop = tf.nn.dropout(conv1_pool, keep_prob=self.keep_prop_tensor)

        with tf.variable_scope("layer_2") as scope:
            with tf.variable_scope("convolution_2_1") as scope:
                with tf.variable_scope("weights") as scope:
                    w2_1 = tf.get_variable("w2_1", shape=[3, 3, self.C[0], self.C[1]],
                                           initializer=tf.contrib.layers.xavier_initializer())

                with tf.variable_scope("biases") as scope:
                    b2_1 = tf.get_variable("b2_1", shape=[self.C[1]],
                                           initializer=tf.constant_initializer(0.))

                conv2_1 = tf.nn.conv2d(input=conv1_drop,
                                       filter=w2_1,
                                       strides=[1, 1, 1, 1],
                                       padding='SAME',
                                       name="conv2_1")

                conv2_1 = tf.nn.bias_add(conv2_1, b2_1, name="bias_add")

                conv2_1 = tf.nn.relu(conv2_1, name='relu')

            with tf.variable_scope("convolution_2_2") as scope:
                with tf.variable_scope("weights") as scope:
                    w2_2 = tf.get_variable("w2_2", shape=[3, 3, self.C[1], self.C[1]],
                                           initializer=tf.contrib.layers.xavier_initializer())

                with tf.variable_scope("biases") as scope:
                    b2_2 = tf.get_variable("b2_2", shape=[self.C[1]],
                                           initializer=tf.constant_initializer(0.))

                conv2_2 = tf.nn.conv2d(input=conv2_1,
                                       filter=w2_2,
                                       strides=[1, 1, 1, 1],
                                       padding='SAME',
                                       name="conv2_2")

                conv2_2 = tf.nn.bias_add(conv2_2, b2_2, name="bias_add")

                conv2_2 = tf.nn.relu(conv2_2, name='relu')

            with tf.variable_scope("convolution_2_3") as scope:
                with tf.variable_scope("weights") as scope:
                    w2_3 = tf.get_variable("w2_3", shape=[3, 3, self.C[1], self.C[1]],
                                           initializer=tf.contrib.layers.xavier_initializer())

                with tf.variable_scope("biases") as scope:
                    b2_3 = tf.get_variable("b2_3", shape=[self.C[1]],
                                           initializer=tf.constant_initializer(0.))

                conv2_3 = tf.nn.conv2d(input=conv2_2,
                                       filter=w2_3,
                                       strides=[1, 1, 1, 1],
                                       padding='SAME',
                                       name="conv2_3")

                conv2_3 = tf.nn.bias_add(conv2_3, b2_3, name="bias_add")

                conv2_3 = tf.nn.relu(conv2_3, name='relu')
            with tf.variable_scope("maxpooling_2_1") as scope:
                conv2_pool = tf.nn.max_pool(conv2_3,
                                            ksize=[1, 2, 2, 1],
                                            strides=[1, 2, 2, 1],
                                            padding='SAME',
                                            name='conv2_pool')
            with tf.variable_scope("dropout_2_1") as scope:
                conv2_drop = tf.nn.dropout(conv2_pool, keep_prob=self.keep_prop_tensor)

        with tf.variable_scope("layer_3") as scope:
            with tf.variable_scope("convolution_3_1") as scope:
                with tf.variable_scope("weights") as scope:
                    w3_1 = tf.get_variable("w3_1", shape=[3, 3, self.C[1], self.C[2]],
                                           initializer=tf.contrib.layers.xavier_initializer())
                with tf.variable_scope("biases") as scope:
                    b3_1 = tf.get_variable("b3_1", shape=[self.C[2]],
                                           initializer=tf.constant_initializer(0.))


                conv3_1 = tf.nn.conv2d(input=conv2_drop,
                                       filter=w3_1,
                                       strides=[1, 1, 1, 1],
                                       padding='SAME',
                                       name="conv3_1")

                conv3_1 = tf.nn.bias_add(conv3_1, b3_1, name="bias_add")

                conv3_1 = tf.nn.relu(conv3_1, name='conv3_1')

            with tf.variable_scope("convolution_3_2") as scope:
                with tf.variable_scope("weights") as scope:
                    w3_2 = tf.get_variable("w3_2", shape=[3, 3, self.C[2], self.C[2]],
                                           initializer=tf.contrib.layers.xavier_initializer())
                with tf.variable_scope("biases") as scope:
                    b3_2 = tf.get_variable("b3_2", shape=[self.C[2]],
                                           initializer=tf.constant_initializer(0.))

                conv3_2 = tf.nn.conv2d(input=conv3_1,
                                       filter=w3_2,
                                       strides=[1, 1, 1, 1],
                                       padding='SAME',
                                       name="conv3_2")

                conv3_2 = tf.nn.bias_add(conv3_2, b3_2, name="bias_add")

                conv3_2 = tf.nn.relu(conv3_2, name='conv3_2')

            with tf.variable_scope("convolution_3_3") as scope:
                with tf.variable_scope("weights") as scope:
                    w3_3 = tf.get_variable("w3_3", shape=[3, 3, self.C[2], self.C[2]],
                                           initializer=tf.contrib.layers.xavier_initializer())

                with tf.variable_scope("biases") as scope:
                    b3_3 = tf.get_variable("b3_3", shape=[self.C[2]],
                                           initializer=tf.constant_initializer(0.))

                conv3_3 = tf.nn.conv2d(input=conv3_2,
                                       filter=w3_3,
                                       strides=[1, 1, 1, 1],
                                       padding='SAME',
                                       name="conv3_3")
                conv3_3 = tf.nn.bias_add(conv3_3, b3_3, name="bias_add")
                conv3_3 = tf.nn.relu(conv3_3, name='conv3_3')

            with tf.variable_scope("maxpooling_3_1") as scope:
                conv3_pool = tf.nn.max_pool(conv3_3,
                                            ksize=[1, 8, 8, 1],
                                            strides=[1, 8, 8, 1],
                                            padding='SAME',
                                            name='conv3_pool')

            with tf.variable_scope("reshape_3_1") as scope:
                conv3_flat = tf.reshape(conv3_pool, shape=[-1, self.C[2]], name='conv3_flat')

            with tf.variable_scope("dropout_3_1") as scope:
                conv3_drop = tf.nn.dropout(conv3_flat, keep_prob=self.keep_prop_tensor, name='conv3_drop')


        with tf.variable_scope("fully_layer") as scope:
            with tf.variable_scope("fully_1_1") as scope:
                with tf.variable_scope("weights") as scope:
                    w4 = tf.get_variable("w4", shape=[self.C[2], self.C[3]],
                                         initializer=tf.contrib.layers.xavier_initializer())
                with tf.variable_scope("biases") as scope:
                    b4 = tf.get_variable("b4", shape=[self.C[3]],
                                         initializer=tf.constant_initializer(0.))

                full1 = tf.add(tf.matmul(conv3_drop, w4), b4)
                full1 = tf.nn.relu(full1, name='relu')

            with tf.variable_scope("dropout_1") as scope:
                full1_drop = tf.nn.dropout(full1, keep_prob=self.keep_prop_tensor)


        with tf.variable_scope("output_layer") as scope:
            with tf.variable_scope("fully_1_1") as scope:
                with tf.variable_scope("weights") as scope:
                    w5 = tf.get_variable("w5", shape=[self.C[3], self.C[4]],
                                         initializer=tf.contrib.layers.xavier_initializer())

                with tf.variable_scope("biases") as scope:
                    b5 = tf.get_variable("b5", shape=[self.C[4]],
                                         initializer=tf.constant_initializer(0.))

            logits = tf.add(tf.matmul(full1_drop, w5), b5, name='logits')

        return logits
if __name__ == '__main__':

    X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name="input_tensor")
    keep_prop_tensor = tf.placeholder(tf.float32, name="keep_prop")

    net = classification10_nn(X, keep_prop_tensor, [30, 50, 80, 500, 10])

    with tf.Session() as session:
        tensorboard_writer = tf.summary.FileWriter("./log", session.graph)

