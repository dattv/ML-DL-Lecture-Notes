import tensorflow as tf
import os
import sys
import logging
import numpy as np

sys.setrecursionlimit(2 ** 20)
np.random.seed(2 ** 10)


class WideResNet:
    def __init__(self, image_size, depth=16, k=8):
        self._depth = depth
        self._k = k
        self._dropout_probability = 0
        self._weight_decay = 0.0005
        self._use_bias = False
        self._weight_init = tf.contrib.layers.xavier_initializer(uniform=False)  # tf.initializers.he_normal()

        logging.debug("image_dim_ordering = 'tf'")
        self._channel_axis = -1
        self._input_shape = (None, image_size, image_size, 3)

    # Wide residual network http://arxiv.org/abs/1605.07146
    def _wide_basic(self, n_input_plane, n_output_plane, stride):
        def f(net):

            # format of conv_params:
            #               [ [kernel_size=("kernel width", "kernel height"),
            #               strides="(stride_vertical,stride_horizontal)",
            #               padding="same" or "valid"] ]
            # B(3,3): orignal <<basic>> block
            conv_params = [[3, 3, stride, "same"],
                           [3, 3, (1, 1), "same"]]

            n_bottleneck_plane = n_output_plane

            # Residual block
            for i, v in enumerate(conv_params):
                if i == 0:
                    if n_input_plane != n_output_plane:
                        net = tf.layers.batch_normalization(net, axis=self._channel_axis, training=True)
                        net = tf.nn.relu(net)
                        convs = net
                    else:
                        convs = tf.layers.batch_normalization(net, axis=self._channel_axis, training=True)
                        convs = tf.nn.relu(convs)

                    convs = tf.layers.conv2d(convs, n_bottleneck_plane,
                                             kernel_size=(v[0], v[1]),
                                             strides=v[2],
                                             padding=v[3],
                                             kernel_initializer=self._weight_init,
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(self._weight_decay),
                                             use_bias=self._use_bias)
                else:
                    convs = tf.layers.batch_normalization(convs, axis=self._channel_axis, training=True)
                    convs = tf.nn.relu(convs)
                    if self._dropout_probability > 0:
                        convs = tf.layers.dropout(convs, rate=self._dropout_probability)

                    convs = tf.layers.conv2d(convs, n_bottleneck_plane, kernel_size=(v[0], v[1]),
                                             strides=v[2],
                                             padding=v[3],
                                             kernel_initializer=self._weight_init,
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(self._weight_decay),
                                             use_bias=self._use_bias)

            # Shortcut Connection: identity function or 1x1 convolutional
            #  (depends on difference between input & output shape - this
            #   corresponds to whether we are using the first block in each
            #   group; see _layer() ).
            if n_input_plane != n_output_plane:
                shotcut = tf.layers.conv2d(net, n_output_plane, kernel_size=(1, 1),
                                           strides=stride,
                                           padding="same",
                                           kernel_initializer=self._weight_init,
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(self._weight_decay),
                                           use_bias=self._use_bias)

            else:
                shotcut = net

            return tf.keras.layers.add([convs, shotcut])

        return f

    # "Stacking Residual Units on the same stage"
    def _layer(self, block, n_input_plane, n_output_plane, count, stride):
        def f(net):
            net = block(n_input_plane, n_output_plane, stride)(net)
            for i in range(2, int(count + 1)):
                net = block(n_output_plane, n_output_plane, stride=(1, 1))(net)
            return net

        return f

    #    def create_model(self):
    def __call__(self):
        logging.debug("Creating model...")

        assert ((self._depth - 4) % 6 == 0)
        n = (self._depth - 4) / 6

        inputs = tf.placeholder(tf.float32, shape=self._input_shape)

        n_stages = [16, 16 * self._k, 32 * self._k, 64 * self._k]
        with tf.variable_scope("wide_resnet") as scope:
            conv1 = tf.layers.conv2d(inputs, filters=n_stages[0], kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding="same",
                                     kernel_initializer=self._weight_init,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self._weight_decay),
                                     use_bias=self._use_bias)  # "One conv at the beginning (spatial size: 32x32)"

            # Add wide residual blocks
            with tf.variable_scope("wider_residual_block") as scope:
                block_fn = self._wide_basic
                with tf.variable_scope("layer_1") as scope:
                    conv2 = self._layer(block_fn, n_input_plane=n_stages[0], n_output_plane=n_stages[1], count=n, stride=1)(
                        conv1)

                with tf.variable_scope("layer_2") as scope:
                    conv3 = self._layer(block_fn, n_input_plane=n_stages[1], n_output_plane=n_stages[2], count=n, stride=2)(
                        conv2)

                with tf.variable_scope("layer_3") as scope:
                    conv4 = self._layer(block_fn, n_input_plane=n_stages[2], n_output_plane=n_stages[3], count=n, stride=2)(
                        conv3)

                batch_norm = tf.layers.batch_normalization(conv4, axis=self._channel_axis, training=True)
                relu = tf.nn.relu(batch_norm)

            # Classifier block
            with tf.variable_scope("classifier_block") as scope:
                pool = tf.layers.average_pooling2d(relu, pool_size=(8, 8), strides=1, padding="same")
                flatten = tf.layers.flatten(pool)
                predictions_g = tf.layers.dense(flatten, 2, kernel_initializer=self._weight_init, use_bias=self._use_bias,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(self._weight_decay),
                                                activation=tf.nn.softmax, name="pred_gender")

                predictions_a = tf.layers.dense(flatten, 101, kernel_initializer=self._weight_init, use_bias=self._use_bias,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(self._weight_decay),
                                                activation=tf.nn.softmax, name="pred_age")


        return predictions_g, predictions_a


def main():
    print("main")
    model = WideResNet(64)()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter("./log", session.graph)


if __name__ == '__main__':

    main()
