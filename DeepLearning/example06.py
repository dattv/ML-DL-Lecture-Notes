from builtins import int
import logging
import tensorflow as tf
import numpy as np

import os
import urllib.request

import tarfile
import zipfile
import numpy as np

import pickle as cPickle

import time
from tqdm import tqdm
import matplotlib.pyplot as plt

CIFAR_DIR = "./CIFA"


def my_hook(t):
    """
    Wraps tqdm instance
    :param t:
    :return:
    """
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """

        :param b:       int option
        :param bsize:   int
        :param tsize:
        :return:
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


if os.path.isdir(CIFAR_DIR) == False:
    os.mkdir(CIFAR_DIR)

cifar_usr = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
name = cifar_usr.split("/")
name = name[len(name) - 1]

full_file_path = CIFAR_DIR + "/" + name
if os.path.isfile(full_file_path) == False:
    print("downloading from: {}".format(cifar_usr))
    with tqdm(unit="B", unit_scale=True, leave=True, miniters=1, desc=cifar_usr.split("/")[-1]) as t:
        urllib.request.urlretrieve(cifar_usr, filename=full_file_path, reporthook=my_hook(t), data=None)
    print("finish download")

# extract compressed file
tar = tarfile.open(full_file_path)
tar.extractall()
tar.close()

# Process data
DATA_PATH = "./cifar-10-batches-py"


def unpickle(file):
    with open(os.path.join(DATA_PATH, file), 'rb') as fo:
        dict = cPickle.load(fo, encoding='bytes')

    return dict


def one_hot(vec, vals=10):
    n = len(vec)
    out = np.zeros((n, vals))

    out[range(n), vec] = 1
    return out


class CifarLoader(object):

    def __init__(self, source_files):
        self._source = source_files
        self._i = 0
        self.images = None
        self.labels = None

    def load(self):
        data = [unpickle(f) for f in self._source]
        images = np.vstack([d[b"data"] for d in data])
        n = len(images)

        self.images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1).astype(float) / 255.
        self.labels = one_hot(np.hstack([d[b"labels"] for d in data]), 10)

        return self

    def nex_batch(self, batch_size):
        x, y = self.images[self._i:self._i + batch_size], self.labels[self._i:self._i + batch_size]
        self._i = (self._i + batch_size) % len(self.images)

        return x, y


class CifarDataManager(object):

    def __init__(self):
        self.train = CifarLoader(["data_batch_{}".format(i) for i in range(1, 6)]).load()
        self.test = CifarLoader(["test_batch"]).load()


def display_cifar(images, size):
    n = len(images)
    plt.figure()
    plt.gca().set_axis_off()

    im = np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size)]) for i in range(size)])

    plt.imshow(im)
    plt.savefig("./CIFA/cifar-10-" + str(n))
    plt.show()


d = CifarDataManager()
train_images = d.train.images
train_labels = d.train.labels

test_images = d.test.images
test_labels = d.test.labels

# display_cifar(images, 10)

WIDTH, HEIGHT, CHANEL = train_images[0].shape
NUM_CLASSES = train_labels.shape[1]

BATCH_SIZE = 64
TOTAL_BATCH = train_labels.shape[0] // BATCH_SIZE

STDDEV_ = 0.1


def net(x_input, keep_prob, is_training):
    C1, C2, C3 = 30, 50, 80
    F1 = 500
    # is_training = tf.cast(is_training, tf.bool)

    # layer 1 ===============================================================================
    with tf.variable_scope("layer_1") as scope:
        with tf.variable_scope("conv111") as sc:
            w1_1 = tf.Variable(tf.truncated_normal([3, 3, CHANEL, C1], stddev=STDDEV_), name="w1_1")
            b1_1 = tf.Variable(tf.constant(0.1, shape=[C1]), name="b1_1")
            conv1_1 = tf.nn.conv2d(input=x_input,
                                   filter=w1_1,
                                   strides=[1, 1, 1, 1],
                                   padding='SAME')
            conv1_1 = conv1_1 + b1_1

            # conv1_1 = tf.layers.batch_normalization(conv1_1, training=is_training)

            conv1_1 = tf.nn.relu(conv1_1, name='relu1_1')

            # conv1_1 = tf.layers.batch_normalization(conv1_1, training=is_training)

        conv1_1 = tf.layers.batch_normalization(conv1_1, training=is_training)

        w1_2 = tf.Variable(tf.truncated_normal([3, 3, C1, C1], stddev=STDDEV_), name='w1_2')
        b1_2 = tf.Variable(tf.constant(0.1, shape=[C1]), name='b1_2')
        conv1_2 = tf.nn.conv2d(input=conv1_1,
                               filter=w1_2,
                               strides=[1, 1, 1, 1],
                               padding='SAME')
        conv1_2 = conv1_2 + b1_2

        # conv1_2 = tf.layers.batch_normalization(conv1_2, training=is_training, fused=True)

        conv1_2 = tf.nn.relu(conv1_2, name='relu1_2')

        w1_3 = tf.Variable(tf.truncated_normal([3, 3, C1, C1], stddev=STDDEV_), name='w1_3')
        b1_3 = tf.Variable(tf.constant(0.1, shape=[C1]), name='b1_3')
        conv1_3 = tf.nn.conv2d(input=conv1_2,
                               filter=w1_3,
                               strides=[1, 1, 1, 1],
                               padding='SAME')
        conv1_3 = conv1_3 + b1_3

        # conv1_3 = tf.layers.batch_normalization(conv1_3, training=is_training, fused=True)

        conv1_3 = tf.nn.relu(conv1_3, name='conv1_3')

        conv1_pool = tf.nn.max_pool(conv1_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name="conv1_pool")

        conv1_drop = conv1_pool  # tf.nn.dropout(conv1_pool, keep_prob=keep_prob)

    # Layer 2 ===============================================================================
    with tf.variable_scope("layer_2") as scope:
        # w2_1 = tf.Variable(tf.truncated_normal([3, 3, C1, C2], stddev=STDDEV_), name='w2_1')
        # b2_1 = tf.Variable(tf.constant(0.1, shape=[C2]), name='b2_1')
        # conv2_1 = tf.nn.conv2d(input=conv1_drop,
        #                        filter=w2_1,
        #                        strides=[1, 1, 1, 1],
        #                        padding='SAME')
        # conv2_1 = conv2_1 + b2_1
        # conv2_1 = tf.nn.relu(conv2_1, name='conv2_1')
        conv2_1 = tf.layers.conv2d(conv1_drop, filters=C2, kernel_size=3, strides=1, padding="same")
        conv2_1 = tf.layers.batch_normalization(conv2_1, training=is_training, fused=True)

        w2_2 = tf.Variable(tf.truncated_normal([3, 3, C2, C2], stddev=STDDEV_), name='w2_2')
        b2_2 = tf.Variable(tf.constant(0.1, shape=[C2]), name='b2_2')
        conv2_2 = tf.nn.conv2d(input=conv2_1,
                               filter=w2_2,
                               strides=[1, 1, 1, 1],
                               padding='SAME')
        conv2_2 = conv2_2 + b2_2

        conv2_2 = tf.layers.batch_normalization(conv2_2, training=is_training, fused=True)

        conv2_2 = tf.nn.relu(conv2_2, name='conv2_2')

        w2_3 = tf.Variable(tf.truncated_normal([3, 3, C2, C2], stddev=STDDEV_), name='w2_3')
        b2_3 = tf.Variable(tf.constant(0.1, shape=[C2]), name='b2_3')
        conv2_3 = tf.nn.conv2d(input=conv2_2,
                               filter=w2_3,
                               strides=[1, 1, 1, 1],
                               padding='SAME')
        conv2_3 = conv2_3 + b2_3

        conv2_3 = tf.layers.batch_normalization(conv2_3, training=is_training, fused=True)

        conv2_3 = tf.nn.relu(conv2_3, name='conv2_3')
        conv2_pool = tf.nn.max_pool(conv2_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='conv2_pool')
        conv2_drop = tf.nn.dropout(conv2_pool, keep_prob=keep_prob)

    # layer 3 ===============================================================================
    with tf.variable_scope("layer_3") as scope:
        w3_1 = tf.Variable(tf.truncated_normal([3, 3, C2, C3], stddev=STDDEV_), name='w3_1')
        b3_1 = tf.Variable(tf.constant(0.1, shape=[C3]), name='b3_1')
        conv3_1 = tf.nn.conv2d(input=conv2_drop,
                               filter=w3_1,
                               strides=[1, 1, 1, 1],
                               padding='SAME')
        conv3_1 = conv3_1 + b3_1

        conv3_1 = tf.layers.batch_normalization(conv3_1, training=is_training)

        conv3_1 = tf.nn.relu(conv3_1, name='conv3_1')

        w3_2 = tf.Variable(tf.truncated_normal([3, 3, C3, C3], stddev=STDDEV_), name='w3_2')
        b3_2 = tf.Variable(tf.constant(0.1, shape=[C3]), name='b3_2')
        conv3_2 = tf.nn.conv2d(input=conv3_1,
                               filter=w3_2,
                               strides=[1, 1, 1, 1],
                               padding='SAME')
        conv3_2 = conv3_2 + b3_2

        conv3_2 = tf.layers.batch_normalization(conv3_2, training=is_training, fused=True)

        conv3_2 = tf.nn.relu(conv3_2, name='conv3_2')

        w3_3 = tf.Variable(tf.truncated_normal([3, 3, C3, C3], stddev=STDDEV_), name='w3_3')
        b3_3 = tf.Variable(tf.constant(0.1, shape=[C3]), name='b3_3')
        conv3_3 = tf.nn.conv2d(input=conv3_2,
                               filter=w3_3,
                               strides=[1, 1, 1, 1],
                               padding='SAME')
        conv3_3 = conv3_3 + b3_3

        conv3_3 = tf.layers.batch_normalization(conv3_3, training=is_training, fused=True)

        conv3_3 = tf.nn.relu(conv3_3, name='conv3_3')

        conv3_pool = tf.nn.max_pool(conv3_3,
                                    ksize=[1, 8, 8, 1],
                                    strides=[1, 8, 8, 1],
                                    padding='SAME',
                                    name='conv3_pool')
        conv3_flat = tf.reshape(conv3_pool, shape=[-1, C3], name='conv3_flat')
        conv3_drop = conv3_flat  # tf.nn.dropout(conv3_flat, keep_prob=keep_prob, name='conv3_drop')

    # fully layer
    with tf.variable_scope("layer_4") as scope:
        w4 = tf.Variable(tf.truncated_normal(shape=[C3, F1], stddev=STDDEV_), name='w4')
        b4 = tf.Variable(tf.constant(0.1, shape=[F1]), name='b4')
        full1 = tf.add(tf.matmul(conv3_drop, w4), b4)

        full1 = tf.nn.relu(full1, name='full1')

        full1_drop = full1  # tf.nn.dropout(full1, keep_prob=keep_prob)

    # output layer
    with tf.variable_scope("output_layer") as scope:
        w5 = tf.Variable(tf.truncated_normal(shape=[F1, NUM_CLASSES], stddev=STDDEV_), name='w5')
        b5 = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]), name='b5')
        logits = tf.add(tf.matmul(full1_drop, w5), b5, name='logits')

    return logits


def classification_block_WideResNet(input_tensor, weight_decay):
    with tf.variable_scope("classifier_block") as sc:
        tf.layers.separable_conv2d
        pool = tf.nn.avg_pool(input_tensor, ksize=[1, 8, 8, 1],
                              strides=[1, 1, 1, 1], padding="SAME", name="avg_pool")

        pool_size = int(pool.shape[1] * pool.shape[2] * pool.shape[3])
        flatten = tf.reshape(pool, shape=[-1, pool_size], name="flatten")

        predict_g = tf.layers.dense(flatten, 10, activation=tf.nn.softmax, use_bias=False,
                                    kernel_initializer=tf.initializers.he_normal(),
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))

    return predict_g


def WideResNet_BasicBlock(inputs, n_input_plane, n_output_plane,
                          stride, weight_decay, sc_name,
                          is_training=True, keep_prob=1.):
    conv_params = [[3, 3, stride, "same"],
                   [3, 3, 1, "same"]]

    n_bottleneck_plane = n_output_plane

    # Residual block
    # for i, v in enumerate(conv_params):
    #     if i == 0:
    net = inputs
    with tf.variable_scope(sc_name + "_BasicBlock") as sc:
        if n_input_plane != n_output_plane:
            net = tf.layers.batch_normalization(net, training=is_training, fused=True)
            net = tf.nn.relu(net)
            convs = net

        else:
            convs = tf.layers.batch_normalization(net, training=is_training)
            convs = tf.nn.relu(convs)

        convs = tf.layers.conv2d(convs, n_bottleneck_plane, kernel_size=3, strides=stride, use_bias=False,
                                 padding="same",
                                 kernel_initializer=tf.initializers.he_normal(),
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))

        convs = tf.layers.batch_normalization(convs, training=is_training)
        convs = tf.nn.relu(convs)
        if keep_prob != 1.:
            convs = tf.nn.dropout(convs, keep_prob=keep_prob)

        convs = tf.layers.conv2d(convs, filters=n_bottleneck_plane, kernel_size=3, strides=1, use_bias=False,
                                 padding="same",
                                 kernel_initializer=tf.initializers.he_normal(),
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
                                 )

        if n_input_plane != n_output_plane:
            shortcut = tf.layers.conv2d(inputs=net, filters=n_output_plane, kernel_size=1, strides=stride,
                                        use_bias=False,
                                        padding="same",
                                        kernel_initializer=tf.initializers.he_normal(),
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay))



        else:
            shortcut = net

        output = tf.math.add(convs, shortcut)

    return output


def WideResNet(input_tensor=None, is_training=True, keep_prob=1., depth=16, k=8):
    weight_decay = 0.0005
    n = (depth - 4) / 6
    n_stages = [16, 16 * k, 32 * k, 64 * k]

    input_chanel = input_tensor.shape[3]
    with tf.variable_scope("WideResNet") as sc:
        conv1 = tf.layers.conv2d(inputs=input_tensor, filters=n_stages[0], kernel_size=3, strides=1, use_bias=False,
                                    padding="same",
                                    kernel_initializer=tf.initializers.he_normal(),
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))

        batch_normalization = tf.layers.batch_normalization(conv1, training=is_training)
        activation = tf.nn.relu(batch_normalization)



        count = n
        with tf.variable_scope("layer1") as scope:

            conv2 = WideResNet_BasicBlock(conv1, n_stages[0], n_stages[1], 1,
                                          weight_decay,
                                          "block1_" + str(0),
                                          is_training, keep_prob)
            for i in range(2, int(count + 1)):
                conv3 = WideResNet_BasicBlock(conv2, n_stages[1], n_stages[1], 1,
                                              weight_decay,
                                              "block1_" + str(i),
                                              is_training, keep_prob)
        with tf.variable_scope("layer2") as scope:
            conv4 = WideResNet_BasicBlock(conv3, n_stages[1], n_stages[2], 2,
                                          weight_decay,
                                          "block2_" + str(0),
                                          is_training, keep_prob)
            for i in range(2, int(count + 1)):
                conv5 = WideResNet_BasicBlock(conv4, n_stages[2], n_stages[2], 1,
                                              weight_decay,
                                              "block2_" + str(i),
                                              is_training, keep_prob)

        with tf.variable_scope("layer3") as scope:
            conv6 = WideResNet_BasicBlock(conv5, n_stages[2], n_stages[3], 2,
                                          weight_decay,
                                          "block3_" + str(0),
                                          is_training, keep_prob)

            for i in range(2, int(count + 1)):
                conv7 = WideResNet_BasicBlock(conv6, n_stages[3], n_stages[3], 1,
                                              weight_decay,
                                              "block3_" + str(i),
                                              is_training, keep_prob)

        # conv7 = conv2d_2
        conv8 = tf.layers.batch_normalization(conv7, training=is_training)
        conv9 = tf.keras.layers.Activation("relu")(conv8)

        # Classification Block
        # print("dkjfldkjkfld", conv1)
        return classification_block_WideResNet(conv9, weight_decay)



class WideResNet2:
    def __init__(self, input_tensor, image_size, depth=16, k=8):
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

        inputs = self._input_tensor #tf.placeholder(tf.float32, shape=self._input_shape)

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
                predictions_g = tf.layers.dense(flatten, 10, kernel_initializer=self._weight_init, use_bias=self._use_bias,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(self._weight_decay),
                                                activation=tf.nn.softmax, name="pred_gender")

        #         predictions_a = tf.layers.dense(flatten, 101, kernel_initializer=self._weight_init, use_bias=self._use_bias,
        #                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(self._weight_decay),
        #                                         activation=tf.nn.softmax, name="pred_age")
        #
        #
        # return predictions_g, predictions_a
        return predictions_g

def main():
    x_input = tf.placeholder(tf.float32, shape=[None, WIDTH, HEIGHT, CHANEL], name='x_input')
    y_input = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name='y_input')
    # keep_prob = tf.placeholder(tf.float32, shape=[1], name="keep_prob")
    is_training = True  # tf.placeholder(tf.float32, name="is_training")

    # logits = net(x_input, 0.5, is_training)
    # print(logits)

    logits = WideResNet(x_input, is_training=is_training, keep_prob=0.5, depth=16, k=8)
    # logits = WideResNet(x_input, 32)()
    # print("kdjfldkj", logits)

    with tf.name_scope("loss") as scope:
        softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                        labels=y_input)

        loss_operation = tf.reduce_mean(softmax_cross_entropy, name="loss")

        tf.summary.scalar("loss", loss_operation)

    g = tf.get_default_graph()
    tf.contrib.quantize.create_training_graph(input_graph=g)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        with tf.name_scope("optimization") as scope:
            optimiser = tf.train.AdamOptimizer(1.e-3).minimize(loss_operation)

    with tf.name_scope("accuracy") as scope:
        with tf.name_scope("correct_preduction") as scope:
            predictions = tf.argmax(logits, 1)
            correct_prediction = tf.equal(predictions, tf.argmax(y_input, 1))

        with tf.name_scope("acc") as scope:
            accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy_operation)

    file_name = os.path.basename(__file__)
    file_name = file_name.split(".")
    name = file_name[0]

    LOG_DIR = "./tmp"

    merged_summary_operation = tf.summary.merge_all()
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        train_summary_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, name) + "/train", session.graph)
        test_summary_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, name) + "/test")
        saver.save(session, "./cifar_model/cifar.ckpt")
        # summary = tf.Summary()

        test_images = d.test.images
        test_labels = d.test.labels
        batch_num = int(len(d.train.images) / BATCH_SIZE)

        step = 0
        for epoch in range(200):
            # id = np.random.randint(1, TOTAL_BATCH)
            # train_img = d.train.nex_batch(BATCH_SIZE)[0]
            # train_lab = d.train.nex_batch(BATCH_SIZE)[1]
            for batch_id in range(batch_num):
                batch = d.train.nex_batch(BATCH_SIZE)

                _, merged_sum = session.run([optimiser, merged_summary_operation],
                                            feed_dict={x_input: batch[0],
                                                       y_input: batch[1]})

                train_summary_writer.add_summary(merged_sum, step)
                step += 1

            if epoch % 1 == 0:
                X = d.test.images.reshape(100, 100, 32, 32, 3)
                Y = d.test.labels.reshape(100, 100, 10)

                acc = np.mean([session.run([accuracy_operation],
                                           feed_dict={x_input: X[i],
                                                      y_input: Y[i]}) for i in range(100)])

                summary = tf.Summary(value=[tf.Summary.Value(tag='acc', simple_value=acc)])
                # summary.value.add(tag="acc", simple_value=acc)
                test_summary_writer.add_summary(summary, epoch)
                # acc, loss = session.run([accuracy_operation, loss_operation],
                #                                     feed_dict={x_input: batch[0],
                #                                                y_input: batch[1],
                #                                                keep_prob: 1.0})

                print("EPOCH: {}, ACC: {:.4}%".format(epoch, acc * 100))
                saver.save(session, "./cifar_model/cifar.ckpt")

        saver.save(session, "./cifar_model/cifar.ckpt")


if __name__ == '__main__':
    main()
