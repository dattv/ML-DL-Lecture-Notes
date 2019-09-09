from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy
import tensorflow as tf
import pickle
from tqdm import tqdm
from urllib.request import urlretrieve
import tarfile


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


class cifar100:

    def __init__(self, cifar100_dir=None):

        self.n_channels = 3
        self.width = 32
        self.heigh = 32
        self.DIR = "./cifar100_dataset"
        if os.path.isdir(self.DIR) == False:
            os.mkdir(self.DIR)

        if cifar100_dir == None:
            cifar100_url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"

            name = cifar100_url.split("/")
            name = name[len(name) - 1]

            full_file_path = os.path.join(self.DIR, name)

            if os.path.isfile(full_file_path) == False:
                print("downloading from: {}".format(cifar100_url))
                with tqdm(unit="B", unit_scale=True, leave=True, miniters=1, desc=cifar100_url.split("/")[-1]) as t:
                    urlretrieve(cifar100_url, filename=full_file_path, reporthook=my_hook(t), data=None)
                print("finish download")

            # extract compressed file
            tar = tarfile.open(full_file_path)
            sub_folders = tar.getnames()
            train_subfolder = [f for f in sub_folders if os.path.split(f)[1] == "train"]
            self._train_subfolder = train_subfolder[0]
            test_subfolder = [f for f in sub_folders if os.path.split(f)[1] == "test"]
            self._test_subfolder = test_subfolder[0]
            tar.extractall()
            tar.close()

    def get_data(self):

        with open(self._train_subfolder, 'rb') as fo:
            try:
                self.samples_train = pickle.load(fo)
            except UnicodeDecodeError:  # python 3.x
                fo.seek(0)
                self.samples_train = pickle.load(fo, encoding='latin1')

        with open(self._test_subfolder, 'rb') as fo:
            try:
                self.samples_test = pickle.load(fo)
            except UnicodeDecodeError:
                fo.seek(0)
                self.samples_test = pickle.load(fo, encoding='latin1')

        train_data_img = self.samples_train["data"]
        train_data_labels = self.samples_train["fine_labels"]
        test_data_img = self.samples_test["data"]
        test_data_lables = self.samples_test["fine_labels"]

        return (train_data_img, train_data_labels), (test_data_img, test_data_lables)


def Conv3x3(inputs, filters, strides, kernel=3, is_training=True):
    """

    :param inputs:
    :param strides:
    :param filters:
    :param kernel:
    :return:
    """
    with tf.variable_scope("Conv3x3") as scope:
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        inputs = tf.pad(inputs, paddings, "CONSTANT")

        output = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel, use_bias=False, strides=strides)

        output = tf.layers.batch_normalization(output, training=is_training)

        output = tf.nn.relu6(output)

    return output


def DwideConv(inputs, kernel_size=3, strides=1, depth_multiplier=1, is_training=True, name=""):
    input_shape = inputs.shape
    weights = tf.get_variable(name + "weight_depth_wide_conv",
                              shape=[kernel_size, kernel_size, input_shape[3], int(input_shape[3] * depth_multiplier)],
                              trainable=is_training, initializer=tf.constant_initializer(0.1))

    output = tf.nn.depthwise_conv2d(inputs, weights, strides=[1, strides, strides, 1], padding="SAME")

    return output


def SepConv3x3(inputs, filters, alpha, pointwise_conv_filters, depth_multiplier=1, strides=1, is_training=True,
               name=""):
    """

    :param inputs:
    :param filters:
    :param alpha:
    :param pointwise_conv_filters:
    :param depth_multiplier:
    :param strides:
    :param is_training:
    :return:
    """
    input_shape = inputs.shape
    with tf.variable_scope("SepConv3x3") as scope:
        pointwise_conv_filters = int(pointwise_conv_filters * alpha)

        output = DwideConv(inputs, kernel_size=3, strides=strides, depth_multiplier=depth_multiplier,
                           is_training=is_training, name=name)

        output = tf.layers.conv2d(output, pointwise_conv_filters, kernel_size=1, strides=strides, padding="valid",
                                  use_bias=False)

        output = tf.layers.batch_normalization(output, training=is_training)

        output = tf.nn.relu6(output)

    return output


def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def MBConv(inputs, kernel=3, expansion=3, stride=1, alpha=1.0, filters=3, block_id=1, is_training=True):
    with tf.variable_scope("MBConv_{}_".format(block_id)) as scope:
        shape = inputs.shape
        pointwise_conv_filters = int(filters * alpha)
        pointwise_filters = make_divisible(pointwise_conv_filters, 8)

        prefix = 'block_{}_'.format(block_id)
        output = inputs
        if block_id:
            output = tf.layers.conv2d(output, expansion * shape[3], kernel_size=1, padding="same",
                                      use_bias=False, activation=None)

            output = tf.layers.batch_normalization(output, training=is_training)

            output = tf.nn.relu6(output)

        else:
            prefix = 'expanded_conv_'

        output = DwideConv(output, kernel_size=kernel, strides=stride, is_training=is_training,
                           name=prefix + 'depthwise')

        output = tf.layers.batch_normalization(output, epsilon=1.e-3, momentum=0.999, name=prefix + "depthwise_bn",
                                               training=is_training)

        output = tf.nn.relu6(output)

        output = tf.layers.conv2d(output, filters=pointwise_filters,
                                  kernel_size=1, padding="same", use_bias=False, activation=None,
                                  name=prefix + "project")

        output = tf.layers.batch_normalization(output, epsilon=1.e-3, momentum=0.999, name=prefix + "project_bn")

        if shape[3] == pointwise_filters and stride == 1:
            return inputs + output
        return output


def MnasNet(input_tensor=None, alpha=1.0, depth_multiplier=1, pooling=None, nb_classes=10, is_training=True):
    input_shape = input_tensor.shape

    first_block_filters = make_divisible(32 * alpha, 8)

    output = Conv3x3(input_tensor, filters=first_block_filters, strides=2, is_training=is_training)

    output = SepConv3x3(output, filters=16, alpha=alpha, pointwise_conv_filters=16, depth_multiplier=depth_multiplier,
                        is_training=is_training)

    output = MBConv(output, kernel=3, expansion=3, stride=2, alpha=alpha, filters=24, block_id=1,
                    is_training=is_training)
    output = MBConv(output, kernel=3, expansion=3, stride=1, alpha=alpha, filters=24, block_id=2,
                    is_training=is_training)
    output = MBConv(output, kernel=3, expansion=3, stride=1, alpha=alpha, filters=24, block_id=3,
                    is_training=is_training)

    output = MBConv(output, kernel=5, expansion=3, stride=2, alpha=alpha, filters=40, block_id=4,
                    is_training=is_training)
    # output = MBConv(output, kernel=5, expansion=3, stride=1, alpha=alpha, filters=40, block_id=5, is_training=is_training)
    # output = MBConv(output, kernel=5, expansion=3, stride=1, alpha=alpha, filters=40, block_id=6, is_training=is_training)

    output = MBConv(output, kernel=5, expansion=6, stride=2, alpha=alpha, filters=80, block_id=7,
                    is_training=is_training)
    # output = MBConv(output, kernel=5, expansion=6, stride=1, alpha=alpha, filters=80, block_id=8, is_training=is_training)
    # output = MBConv(output, kernel=5, expansion=6, stride=1, alpha=alpha, filters=80, block_id=9, is_training=is_training)

    output = MBConv(output, kernel=3, expansion=6, stride=1, alpha=alpha, filters=96, block_id=10,
                    is_training=is_training)
    # output = MBConv(output, kernel=3, expansion=6, stride=1, alpha=alpha, filters=96, block_id=11, is_training=is_training)

    output = MBConv(output, kernel=5, expansion=6, stride=2, alpha=alpha, filters=192, block_id=12,
                    is_training=is_training)
    # output = MBConv(output, kernel=5, expansion=6, stride=1, alpha=alpha, filters=192, block_id=13, is_training=is_training)
    # output = MBConv(output, kernel=5, expansion=6, stride=1, alpha=alpha, filters=192, block_id=14, is_training=is_training)
    # output = MBConv(output, kernel=5, expansion=6, stride=1, alpha=alpha, filters=192, block_id=15, is_training=is_training)

    output = MBConv(output, kernel=3, expansion=6, stride=1, alpha=alpha, filters=320, block_id=16,
                    is_training=is_training)

    if pooling == "avg":
        output = tf.layers.average_pooling2d(output, pool_size=2, strides=2, padding="valid")
    else:
        output = tf.layers.max_pooling2d(output, pool_size=2, strides=2, padding="valid")

    output_shape = output.shape
    output = tf.reshape(output, shape=[-1, int(output_shape[1] * output_shape[2] * output_shape[3])])
    output = tf.layers.dense(output, units=nb_classes, activation=tf.nn.softmax, use_bias=True)

    return output


def main():
    root = os.path.dirname(__file__)
    log_name = "logs_" + os.path.basename(__file__).split(".")[0]

    cifar100_dir = "/home/dat/Downloads/cifar-100-python/train"
    dataset = cifar100()
    (train_data_img, train_data_labels), (test_data_img, test_data_lables) = dataset.get_data()

    input_tensor = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="input")
    target_tensor = tf.placeholder(tf.float32, shape=[None, 10], name="target")
    output = MnasNet(input_tensor)

    with tf.variable_scope("loss") as scope:
        loss_categorical = tf.keras.losses.categorical_crossentropy(target_tensor, output)
        loss = tf.reduce_mean(loss_categorical)
    tf.summary.scalar("loss", loss)

    optimiser = tf.train.AdamOptimizer().minimize(loss)

    with tf.variable_scope("accuracy") as scope:
        correction = tf.argmax(output, 1)
        acc = tf.equal(correction, tf.argmax(target_tensor, 1))
        acc = tf.reduce_mean(tf.cast(acc, tf.float32))

    tf.summary.scalar("accuracy", acc)

    summary_merged = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(os.path.join(root, log_name) + "/train", sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(root, log_name) + "/test", sess.graph)

        batch_size = 2
        total_batch = int(len(train_data_img) // batch_size)
        step = 0
        for epoch_id in range(100):
            for batch_id in range(total_batch):
                first = batch_id * batch_size
                last = first + batch_size

                _, accuracy, error, summ = sess.run([optimiser, acc, loss, summary_merged],
                                                    feed_dict={input_tensor: train_data_img[first:last],
                                                               target_tensor: train_data_labels[first, last]})

                train_writer.add_summary(summ, step)
                step += 1

            if epoch_id % 2 == 0:

                total_batch = int(len(test_data_img) // batch_size)
                accuracy_array = []
                error_array = []
                for batch_id in range(total_batch):
                    first = batch_id * batch_size
                    last = first + batch_size
                    accuracy, error = sess.run([acc, loss], feed_dict={input_tensor: test_data_img[first:last],
                                                                       target_tensor: test_data_lables[first:last]})

                    accuracy_array.append(accuracy)
                    error_array.append(error)

                accuracy_total = numpy.mean(accuracy_array)
                error_total = numpy.mean(error_array)
                print("TEST: EPOCH: {}, ACCURACY: {}, LOSS: {}".format(epoch_id, accuracy_total, error_total))


if __name__ == '__main__':
    main()
