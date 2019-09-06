import tensorflow as tf
import os
import sys
import pickle as cPickle
import numpy as np
import matplotlib.pyplot as plt


class WideResNet:
    def __init__(self, input_tensor, image_size, depth=16, k=8):
        self._depth = depth
        self._k = k
        self._dropout_probability = 0
        self._weight_decay = 0.0005
        self._use_bias = False
        self._weight_init = tf.contrib.layers.xavier_initializer(uniform=False)  # tf.initializers.he_normal()

        self._channel_axis = -1
        self._input_shape = (None, image_size, image_size, 3)

        self._inputs = input_tensor

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

        assert ((self._depth - 4) % 6 == 0)
        n = (self._depth - 4) / 6

        inputs = self._inputs

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

                batch_norm = tf.layers.batch_normalization(conv4, axis=self._channel_axis, training=True)
                relu = tf.nn.relu(batch_norm)

            # Classifier block
            with tf.variable_scope("classifier_block") as scope:
                pool = tf.layers.average_pooling2d(relu, pool_size=(8, 8), strides=1, padding="same")
                flatten = tf.layers.flatten(pool)
                predictions = tf.layers.dense(flatten, 10, kernel_initializer=self._weight_init,
                                              use_bias=self._use_bias,
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(self._weight_decay),
                                              activation=tf.nn.softmax, name="pred_gender")

        return predictions

#
# # Process data
# DATA_PATH = "./cifar-10-batches-py"
#
#
# def unpickle(file):
#     with open(os.path.join(DATA_PATH, file), 'rb') as fo:
#         dict = cPickle.load(fo, encoding='bytes')
#
#     return dict
#
#
# def one_hot(vec, vals=10):
#     n = len(vec)
#     out = np.zeros((n, vals))
#
#     out[range(n), vec] = 1
#     return out
#
#
# class CifarLoader(object):
#
#     def __init__(self, source_files):
#         self._source = source_files
#         self._i = 0
#         self.images = None
#         self.labels = None
#
#     def load(self):
#         data = [unpickle(f) for f in self._source]
#         images = np.vstack([d[b"data"] for d in data])
#         n = len(images)
#
#         self.images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1).astype(float) / 255.
#         self.labels = one_hot(np.hstack([d[b"labels"] for d in data]), 10)
#
#         return self
#
#     def nex_batch(self, batch_size):
#         x, y = self.images[self._i:self._i + batch_size], self.labels[self._i:self._i + batch_size]
#         self._i = (self._i + batch_size) % len(self.images)
#
#         return x, y
#
#
# class CifarDataManager(object):
#
#     def __init__(self):
#         self.train = CifarLoader(["data_batch_{}".format(i) for i in range(1, 6)]).load()
#         self.test = CifarLoader(["test_batch"]).load()
#
#
# def display_cifar(images, size):
#     n = len(images)
#     plt.figure()
#     plt.gca().set_axis_off()
#
#     im = np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size)]) for i in range(size)])
#
#     plt.imshow(im)
#     plt.savefig("./CIFA/cifar-10-" + str(n))
#     plt.show()
#
#
# data = CifarDataManager()
# train_images = data.train.images
# train_labels = data.train.labels
#
# test_images = data.test.images
# test_labels = data.test.labels
#
# # display_cifar(images, 10)
#
# WIDTH, HEIGHT, CHANEL = train_images[0].shape
# NUM_CLASSES = train_labels.shape[1]
#
# BATCH_SIZE = 16
# TOTAL_BATCH = train_labels.shape[0] // BATCH_SIZE
#
# STDDEV_ = 0.1

X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name="input")
TARGET = tf.placeholder(tf.float32, shape=[None, 10], name="target")

output = WideResNet(X, 32, 16, 8)()

with tf.variable_scope("loss") as scope:
    softmax_cross_entropy = tf.compat.v1.keras.backend.categorical_crossentropy(TARGET,
                                                                                output)

    loss_operation = tf.reduce_mean(softmax_cross_entropy, name="loss")

    tf.summary.scalar("loss", loss_operation)

with tf.variable_scope("optimization") as scope:
    optimiser = tf.train.AdamOptimizer(1.e-5).minimize(loss_operation)

with tf.variable_scope("accuracy") as scope:
    with tf.name_scope("correct_preduction") as scope:
        predictions = tf.argmax(output, 1)
        correct_prediction = tf.equal(predictions, tf.argmax(TARGET, 1))

    with tf.name_scope("acc") as scope:
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar("accuracy", accuracy_operation)

merged_summary_operation = tf.summary.merge_all()
saver = tf.train.Saver()

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    train_writer = tf.summary.FileWriter("./log_cifar10/train", session.graph)
    # test_writer = tf.summary.FileWriter("./log_cifar10/test")
    # builder = tf.saved_model.Builder("./pb_cifar10_model")
    # builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.TRAINING])

    val_acc = None
    summary_var_acc = tf.Summary()
    summary_var_acc.value.add(tag="var_acc", simple_value=val_acc)

    for epoch in range(10000):

    #     saver.save(session, "./ckpt_cifar10_model/ckpt_cifar10_model.ckpt")
    #     batch = data.train.nex_batch(BATCH_SIZE)
    #     _, merge_sum = session.run([optimiser, merged_summary_operation],
    #                                feed_dict={X: batch[0],
    #                                           TARGET: batch[1]})
    #
    #     train_writer.add_summary(merge_sum, epoch)
    #
    #     if epoch % 1000 == 0:
    #         test_imgs = data.test.images.reshape(100, 100, 32, 32, 3)
    #         test_lbs = data.test.labels.reshape(100, 100, 10)
    #
    #         acc = np.mean([session.run([accuracy_operation],
    #                                    feed_dict={X: test_imgs[i],
    #                                               TARGET: test_lbs[i]}) for i in range(100)])
    #
    #         print("EPOCH: {}, ACC: {}".format(epoch, acc * 100))
    #
    #         val_acc = acc
    #         summary_var_acc.value[0].simple_value = val_acc
    #         test_writer.add_summary(summary_var_acc, epoch)
    #
    #         test_writer.add_summary(merge_sum, epoch)
    #
    # saver.save(session, "./ckpt_cifar10_model/ckpt_cifar10_model.ckpt")
    # builder.save()
