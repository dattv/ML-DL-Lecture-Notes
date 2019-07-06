import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
import numpy
import matplotlib.pyplot as plt

root_path = os.path.dirname(os.path.dirname(__file__))
siamese_path = os.path.join(root_path, "siamese_tf")

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

train_images = mnist.train.images
train_labels = mnist.train.labels

input_size = 784
n_chanel = 1
no_classes = 10
total_batch = 50000

file_name = __file__
file_name = file_name.split("/")
file_name = file_name[len(file_name) - 1]

file_name = file_name.split(".")[0]

log_path = os.path.join(siamese_path, file_name)

if os.path.isdir(log_path) == False:
    os.mkdir(log_path)


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
stddev = 0.1


class data_set():
    def __init__(self, images, labels):

        self._images = images
        self._labels = labels

        self._dict = []
        for i in range(10):
            self._dict.append([])

        for i in range(len(labels)):
            number = numpy.argmax(labels[i])
            self._dict[number].append(i)

    def _get_triplet(self):
        anchor_index1 = numpy.random.randint(0, 10)
        size_anchor1 = len(self._dict[anchor_index1])
        anchor_index2 = numpy.random.randint(0, size_anchor1)

        # find similar image
        similar_index2 = numpy.random.randint(0, size_anchor1)

        # find dissimilar image
        bool_continue = True
        while (bool_continue):
            dissimilar_index1 = numpy.random.randint(0, 10)
            if dissimilar_index1 != anchor_index1:
                bool_continue = False

        size_dissimilar = len(self._dict[dissimilar_index1])
        dissimilar_index2 = numpy.random.randint(0, size_dissimilar)

        return self._dict[anchor_index1][anchor_index2], self._dict[anchor_index1][similar_index2], \
               self._dict[dissimilar_index1][dissimilar_index2]

    def _get_triplet_batch(self, batch_size):
        result_anchor, result_positive, result_negative = [], [], []
        for _ in range(batch_size):
            anchor_index, positive_index, negative_index = self._get_triplet()

            result_anchor.append(self._images[anchor_index, :])
            result_positive.append(self._images[positive_index, :])
            result_negative.append(self._images[negative_index, :])
        #
        # for i in range(batch_size):
        #     plt.subplot(3, batch_size, i + 1)
        #     plt.imshow(result_anchor[i].reshape((28, 28)))
        #
        #     plt.subplot(3, batch_size, batch_size + i + 1)
        #     plt.imshow(result_positive[i].reshape((28, 28)))
        #
        #     plt.subplot(3, batch_size, 2 * batch_size + i + 1)
        #     plt.imshow(result_negative[i].reshape((28, 28)))
        #
        # plt.show()

        return result_anchor, result_positive, result_negative


def mnist_model(input):
    nChanel = input.shape[3]

    with tf.name_scope("convolution1") as scope:
        with tf.name_scope("weights") as scope:
            w_1_1 = tf.get_variable("w_1_1",
                                    shape=[7, 7, nChanel, 32],
                                    initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                    dtype=tf.float32)

        with tf.name_scope("biases") as scope:
            b_1_1 = tf.get_variable("b_1_1",
                                    shape=[32],
                                    initializer=tf.constant_initializer(stddev),
                                    dtype=tf.float32)

        conv1 = tf.nn.conv2d(input=input,
                             filter=w_1_1,
                             strides=[1, 1, 1, 1],
                             padding="SAME")
        conv1 += b_1_1
        conv1 = tf.nn.relu(conv1, name="conv1")

    with tf.name_scope("pooling1") as scope:
        pool1 = tf.nn.max_pool(conv1,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding="SAME",
                               name="pool1")

    with tf.name_scope("convolution2") as scope:
        with tf.name_scope("weights") as scope:
            w_2_1 = tf.get_variable("w_2_1",
                                    shape=[5, 5, 32, 64],
                                    initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                    dtype=tf.float32)

        with tf.name_scope("biases") as scope:
            b_2_1 = tf.get_variable("b_2_1",
                                    shape=[64],
                                    initializer=tf.constant_initializer(stddev),
                                    dtype=tf.float32)

        conv2 = tf.nn.conv2d(input=pool1,
                             filter=w_2_1,
                             strides=[1, 1, 1, 1],
                             padding="SAME")
        conv2 += b_2_1
        conv2 = tf.nn.relu(conv2, name="conv2")

    with tf.name_scope("pooling2") as scope:
        pool2 = tf.nn.max_pool(conv2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding="SAME",
                               name="pool2")

    with tf.name_scope("convolution3") as scope:
        with tf.name_scope("weights") as scope:
            w_3_1 = tf.get_variable("w_3_1",
                                    shape=[3, 3, 64, 128],
                                    initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                    dtype=tf.float32)
        with tf.name_scope("biases") as scope:
            b_3_1 = tf.get_variable("b_3_1",
                                    shape=[128],
                                    initializer=tf.constant_initializer(init_bias_value),
                                    dtype=tf.float32)
        conv3 = tf.nn.conv2d(input=pool2,
                             filter=w_3_1,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        conv3 += b_3_1
        conv3 = tf.nn.relu(conv3, name="conv3")

    with tf.name_scope("pooling3") as scope:
        pool3 = tf.nn.max_pool(conv3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding="SAME",
                               name="pool3")

    with tf.name_scope("convolution4") as scope:
        with tf.name_scope("weights") as scope:
            w_4_1 = tf.get_variable("w_4_1",
                                    shape=[1, 1, 128, 256],
                                    initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                    dtype=tf.float32)
        with tf.name_scope("biases") as scope:
            b_4_1 = tf.get_variable("b_4_1",
                                    shape=[256],
                                    initializer=tf.constant_initializer(init_bias_value),
                                    dtype=tf.float32)

        conv4 = tf.nn.conv2d(input=pool3,
                             filter=w_4_1,
                             strides=[1, 1, 1, 1],
                             padding="SAME")
        conv4 += b_4_1
        conv4 = tf.nn.relu(conv4, name="conv4")

    with tf.name_scope("pooling4") as scope:
        pool4 = tf.nn.max_pool(conv4,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding="SAME",
                               name="pool4")

    with tf.name_scope("convolution5") as scope:
        with tf.name_scope("weights") as scope:
            w_5_1 = tf.get_variable("w_5_1",
                                    shape=[1, 1, 256, 2],
                                    initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                    dtype=tf.float32)
        with tf.name_scope("biases") as scope:
            b_5_1 = tf.get_variable("b_5_1",
                                    shape=[2],
                                    initializer=tf.constant_initializer(init_bias_value),
                                    dtype=tf.float32)

        conv5 = tf.nn.conv2d(input=pool4,
                             filter=w_5_1,
                             strides=[1, 1, 1, 1],
                             padding="SAME")
        conv5 += b_5_1
        conv5 = tf.nn.relu(conv5, name="conv5")

    with tf.name_scope("pooling5") as scope:
        pool5 = tf.nn.max_pool(conv5,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding="SAME")

    size = pool5.shape
    size = size[1] * size[2] * size[3]
    with tf.name_scope("flatten") as scope:
        flat = tf.reshape(pool5, shape=[-1, int(size)], name="flat")

    return flat


def triple_loss_func(anchor, positive, negative):
    margin = 5.0
    d_pos = tf.reduce_sum(tf.square(anchor - positive), 1)
    d_neg = tf.reduce_sum(tf.square(anchor - negative), 1)

    loss = tf.maximum(0., margin + d_pos - d_neg)
    loss = tf.reduce_mean(loss, name="triplet_loss")

    return loss


def main():
    anchor = tf.placeholder(tf.float32, shape=[None, 784], name="anchor")
    positive = tf.placeholder(tf.float32, shape=[None, 784], name="positive")
    negative = tf.placeholder(tf.float32, shape=[None, 784], name="negative")
    y = tf.placeholder(tf.float32, shape=[None], name="y")

    anchor_2d = tf.reshape(anchor, shape=[-1, 28, 28, n_chanel], name="anchor_2d")
    positive_2d = tf.reshape(positive, shape=[-1, 28, 28, n_chanel], name="positive_2d")
    negative_2d = tf.reshape(negative, shape=[-1, 28, 28, n_chanel], name="negative_2d")

    with tf.variable_scope("siamese") as scope:
        anchor_model = mnist_model(anchor_2d)

        scope.reuse_variables()

        positive_model = mnist_model(positive_2d)

        scope.reuse_variables()

        negative_model = mnist_model(negative_2d)

    loss = triple_loss_func(anchor_model, positive_model, negative_model)

    tf.summary.scalar("loss", loss)

    optimiser = tf.train.AdamOptimizer(1.e-5).minimize(loss)

    merged = tf.summary.merge_all()

    mnist_data_set = data_set(train_images, train_labels)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(log_path + "/train.log", session.graph)

        for batch in range(total_batch):
            anchor_inputs, positive_inputs, negative_inputs = mnist_data_set._get_triplet_batch(10)

            _, l, summation = session.run([optimiser, loss, merged], feed_dict={anchor: anchor_inputs,
                                                             positive: positive_inputs,
                                                             negative: negative_inputs})

            writer.add_summary(summation, batch)
            if batch % 1000 == 0:
                print("epoch: {}, error: {}".format(batch, l))


if __name__ == '__main__':
    main()
