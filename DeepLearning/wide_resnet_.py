import tensorflow as tf
import os
import sys
import logging
import numpy as np

sys.setrecursionlimit(2 ** 20)
np.random.seed(2 ** 10)


class WideResNet:
    def __init__(self, input_tensor, image_size, depth=16, k=8, is_train=False, is_train_able=False):
        self._depth = depth
        self._k = k
        self._dropout_probability = 0
        self._weight_decay = 0.0005
        self._use_bias = False
        self._weight_init = tf.contrib.layers.xavier_initializer(uniform=False)  # tf.initializers.he_normal()

        logging.debug("image_dim_ordering = 'tf'")
        self._channel_axis = -1
        self._input_shape = (None, image_size, image_size, 3)
        self._input_tensor = input_tensor
        self._is_train = is_train
        self._is_train_able = is_train_able

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
                        net = tf.layers.batch_normalization(net, axis=self._channel_axis, training=self._is_train,
                                                            trainable=self._is_train_able)
                        net = tf.nn.relu(net)
                        convs = net
                    else:
                        convs = tf.layers.batch_normalization(net, axis=self._channel_axis, training=self._is_train,
                                                              trainable=self._is_train_able)
                        convs = tf.nn.relu(convs)

                    convs = tf.layers.conv2d(convs, n_bottleneck_plane,
                                             kernel_size=(v[0], v[1]),
                                             strides=v[2],
                                             padding=v[3],
                                             kernel_initializer=self._weight_init,
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(self._weight_decay),
                                             use_bias=self._use_bias)
                else:
                    convs = tf.layers.batch_normalization(convs, axis=self._channel_axis, training=self._is_train,
                                                          trainable=self._is_train_able)
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

        inputs = self._input_tensor

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
                    conv2 = self._layer(block_fn, n_input_plane=n_stages[0], n_output_plane=n_stages[1], count=n,
                                        stride=1)(
                        conv1)

                with tf.variable_scope("layer_2") as scope:
                    conv3 = self._layer(block_fn, n_input_plane=n_stages[1], n_output_plane=n_stages[2], count=n,
                                        stride=2)(
                        conv2)

                with tf.variable_scope("layer_3") as scope:
                    conv4 = self._layer(block_fn, n_input_plane=n_stages[2], n_output_plane=n_stages[3], count=n,
                                        stride=2)(
                        conv3)

                batch_norm = tf.layers.batch_normalization(conv4, axis=self._channel_axis, training=self._is_train,
                                                           trainable=self._is_train_able)
                relu = tf.nn.relu(batch_norm)

            # Classifier block
            with tf.variable_scope("classifier_block") as scope:
                pool = tf.layers.average_pooling2d(relu, pool_size=(8, 8), strides=1, padding="same")
                flatten = tf.layers.flatten(pool)
                predictions_g = tf.layers.dense(flatten, 2, kernel_initializer=self._weight_init,
                                                use_bias=self._use_bias,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(self._weight_decay),
                                                activation=tf.nn.softmax, name="pred_gender")

                predictions_a = tf.layers.dense(flatten, 101, kernel_initializer=self._weight_init,
                                                use_bias=self._use_bias,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(self._weight_decay),
                                                activation=tf.nn.softmax, name="pred_age")

        return predictions_g, predictions_a


def WideResNet_v2(input_tensor, is_train=False, is_train_able=False, depth=16, k=8, nb_class=10):

    dropout_probability = 0
    weight_decay = 0.0005
    use_bias = True
    weight_init = tf.contrib.layers.xavier_initializer(uniform=False)
    channel_axis = -1

    n_stages = [16, 16 * k, 32 * k, 64 * k]

    with tf.variable_scope("WideResNet") as scope:
        conv1 = tf.layers.conv2d(input_tensor, filters=16, kernel_size=(3, 3),
                                 strides=(1, 1),
                                 padding="same",
                                 kernel_initializer=weight_init,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 use_bias=use_bias)

        # LAYER_1
        with tf.variable_scope("layer_1") as scope:
            batch_norm1 = tf.layers.batch_normalization(conv1, axis=channel_axis, training=is_train,
                                                        trainable=is_train_able)

            activation1 = tf.nn.relu(batch_norm1)

            conv2 = tf.layers.conv2d(activation1, filters=128, kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding="same",
                                     kernel_initializer=weight_init,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                     use_bias=use_bias)

            batch_norm2 = tf.layers.batch_normalization(conv2, axis=channel_axis, training=is_train,
                                                        trainable=is_train_able)

            activation2 = tf.nn.relu(batch_norm2)

            if dropout_probability > 0:
                activation2 = tf.layers.dropout(activation2, rate=dropout_probability)

            conv3 = tf.layers.conv2d(activation2, filters=128, kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding="same",
                                     kernel_initializer=weight_init,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                     use_bias=use_bias)

            conv4 = tf.layers.conv2d(activation1, filters=128, kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding="same",
                                     kernel_initializer=weight_init,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                     use_bias=use_bias)

            add = tf.add_n([conv3, conv4])

            batch_norm3 = tf.layers.batch_normalization(add, axis=channel_axis, training=is_train,
                                                        trainable=is_train_able)

            activation3 = tf.nn.relu(batch_norm3)

            conv5 = tf.layers.conv2d(activation3, filters=128, kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding="same",
                                     kernel_initializer=weight_init,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                     use_bias=use_bias)

            batch_norm4 = tf.layers.batch_normalization(conv5, axis=channel_axis, training=is_train,
                                                        trainable=is_train_able)

            activation4 = tf.nn.relu(batch_norm4)

            if dropout_probability > 0:
                activation4 = tf.layers.dropout(activation4, rate=dropout_probability)

            conv6 = tf.layers.conv2d(activation4, filters=128, kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding="same",
                                     kernel_initializer=weight_init,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                     use_bias=use_bias)

            add1 = tf.add_n([conv6, add])

        # LAYER2
        with tf.variable_scope("layer_2") as scope:
            batch_norm5 = tf.layers.batch_normalization(add1, axis=channel_axis, training=is_train,
                                                        trainable=is_train_able)

            activation5 = tf.nn.relu(batch_norm5)

            conv7 = tf.layers.conv2d(activation5, filters=256, kernel_size=(3, 3),
                                     strides=(2, 2),
                                     padding="same",
                                     kernel_initializer=weight_init,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                     use_bias=use_bias)

            batch_norm6 = tf.layers.batch_normalization(conv7, axis=channel_axis, training=is_train,
                                                        trainable=is_train_able)

            activation6 = tf.nn.relu(batch_norm6)

            if dropout_probability > 0:
                activation6 = tf.layers.dropout(activation6, axis=channel_axis, training=is_train,
                                                trainable=is_train_able)

            conv8 = tf.layers.conv2d(activation6, filters=256, kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding="same",
                                     kernel_initializer=weight_init,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                     use_bias=use_bias)

            conv9 = tf.layers.conv2d(activation5, filters=256, kernel_size=(1, 1),
                                     strides=(2, 2),
                                     padding="same",
                                     kernel_initializer=weight_init,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                     use_bias=use_bias)

            add2 = tf.add_n([conv8, conv9])

            batch_norm7 = tf.layers.batch_normalization(add2, axis=channel_axis, training=is_train,
                                                        trainable=is_train_able)

            activation7 = tf.nn.relu(batch_norm7)

            conv10 = tf.layers.conv2d(activation7, filters=256, kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding="same",
                                      kernel_initializer=weight_init,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                      use_bias=use_bias)

            batch_norm8 = tf.layers.batch_normalization(conv10, axis=channel_axis, training=is_train,
                                                        trainable=is_train_able)

            activation8 = tf.nn.relu(batch_norm8)

            if dropout_probability > 0:
                activation8 = tf.layers.batch_normalization(activation8)

            conv11 = tf.layers.conv2d(activation8, filters=256, kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding="same",
                                      kernel_initializer=weight_init,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                      use_bias=use_bias)

            add3 = tf.add_n([conv11, add2])

        # LAYER3
        with tf.variable_scope("layer_3") as scope:
            batch_norm9 = tf.layers.batch_normalization(add3, axis=channel_axis, training=is_train,
                                                        trainable=is_train_able)

            activation9 = tf.nn.relu(batch_norm9)

            conv12 = tf.layers.conv2d(activation9, filters=512, kernel_size=(3, 3),
                                      strides=(2, 2),
                                      padding="same",
                                      kernel_initializer=weight_init,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                      use_bias=use_bias)

            batch_norm10 = tf.layers.batch_normalization(conv12, axis=channel_axis, training=is_train,
                                                        trainable=is_train_able)

            activation10 = tf.nn.relu(batch_norm10)

            if dropout_probability > 0:
                activation10 = tf.layers.dropout(activation10, rate=dropout_probability)

            conv13 = tf.layers.conv2d(activation10, filters=512, kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding="same",
                                      kernel_initializer=weight_init,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                      use_bias=use_bias)

            conv14 = tf.layers.conv2d(activation9, filters=512, kernel_size=(1, 1),
                                      strides=(2, 2),
                                      padding="same",
                                      kernel_initializer=weight_init,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                      use_bias=use_bias)

            add4 = tf.add_n([conv13, conv14])

            batch_norm11 = tf.layers.batch_normalization(add4, axis=channel_axis, training=is_train,
                                                        trainable=is_train_able)

            activation11 = tf.nn.relu(batch_norm11)

            conv15 = tf.layers.conv2d(activation11, filters=512, kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding="same",
                                      kernel_initializer=weight_init,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                      use_bias=use_bias)

            batch_norm12 = tf.layers.batch_normalization(conv15, axis=channel_axis, training=is_train,
                                                        trainable=is_train_able)

            activation11 = tf.nn.relu(batch_norm12)

            if dropout_probability > 0:
                activation11 = tf.layers.dropout(activation11, rate=dropout_probability)

            conv15 = tf.layers.conv2d(activation11, filters=512, kernel_size=(3, 3),
                                      padding="same",
                                      kernel_initializer=weight_init,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                      use_bias=use_bias)

            add5 = tf.add_n([conv15, add4])


        with tf.variable_scope("prediction") as scope:

            batch_norm13 = tf.layers.batch_normalization(add1, axis=channel_axis, training=is_train,
                                                        trainable=is_train_able)

            activation12 = tf.nn.relu(batch_norm13)


            avg_pool = tf.layers.average_pooling2d(activation12, pool_size=(8, 8), strides=1, padding="same")

            flatten = tf.layers.flatten(avg_pool)

            prediction = tf.layers.dense(flatten, nb_class, activation=None)

    return prediction


def main():
    print("main")
    input_tensor = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
    # model = WideResNet(64)()
    prediction = WideResNet_v2(input_tensor)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter("./log", session.graph)


if __name__ == '__main__':
    main()
