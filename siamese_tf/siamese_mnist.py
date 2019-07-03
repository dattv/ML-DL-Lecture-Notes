import os

import numpy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

root_path = os.path.dirname(os.path.dirname(__file__))

siamese_path = os.path.join(root_path, "siamese_tf")

mnist_path = os.path.join(siamese_path, "MNIST_data")
mnist_data = input_data.read_data_sets("MNIST_data", one_hot=True)

mnist_data_train = mnist_data.train
mnist_data_train_images = mnist_data_train._images
mnist_data_train_labels = mnist_data_train._labels

mnist_data_test = mnist_data.test
mnist_data_test_images = mnist_data_test._images
mnist_data_test_labels = mnist_data_test._labels

mnist_data_val = mnist_data.validation


class Dataset(object):
    def __init__(self, images, labels):

        self._images = images
        self._labels = labels

        self._dict = []
        for i in range(10):
            self._dict.append([])

        for i in range(len(labels)):
            number = numpy.argmax(labels[i])
            self._dict[number].append(i)

    def _get_siamese_similar_pair(self):
        result = []
        index1 = numpy.random.randint(0, 10)
        size = len(self._dict[index1])
        for pair in range(2):
            index2 = numpy.random.randint(0, size)

            result.append([index1, index2])
        return result, 0

    def _get_siamese_dissimilar_pair(self):
        result = []
        index1 = numpy.random.randint(0, 10)
        size = len(self._dict[index1])
        index2 = numpy.random.randint(0, size)
        result.append([index1, index2])

        bool_coninue = True
        while bool_coninue:
            temp_index1 = numpy.random.randint(0, 10)
            if temp_index1 != index1:
                bool_coninue = False

        size = len(self._dict[temp_index1])
        temp_index2 = numpy.random.randint(0, size)
        result.append([temp_index1, temp_index2])

        return result, 1

    def _get_siamese_pair(self):
        if numpy.random.random() < 0.5:
            return self._get_siamese_similar_pair()
        else:
            return self._get_siamese_dissimilar_pair()

    def _get_siamese_batch(self, n_batch):
        idx_l, idx_r, labels = [], [], []
        for _ in range(n_batch):
            lr, y = self._get_siamese_pair()
            idx_l.append(lr[0][1])
            idx_r.append(lr[1][1])
            labels.append(y)
        return self._images[idx_l, :], self._images[idx_r, :], numpy.expand_dims(labels, axis=1)


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
stddev = 0.1


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


def contrastive_loss(model1, model2, y, margin):
    with tf.name_scope("constrastive_loss") as scope:
        distance = tf.sqrt(tf.reduce_sum(tf.pow(model1 - model2, 2), 1, keepdims=True))
        similarity = y * tf.square(distance)  # keep the similar label (1) close to each other
        dissimilarity = (1 - y) * tf.square(tf.maximum((margin - distance),
                                                       0))  # give penalty to dissimilar label if the distance is bigger than margin
    return tf.reduce_mean(dissimilarity + similarity) / 2


def main():
    # d = Dataset(mnist_data_train_images, mnist_data_train_labels)
    #
    # img_l, img_r, l = d._get_siamese_batch(10)
    left = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="left")
    right = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="right")

    l_img = tf.reshape(left, shape=[-1, 28, 28, 1], name="l_img")
    r_img = tf.reshape(right, shape=[-1, 28, 28, 1], name="r_img")

    with tf.name_scope("similarity"):
        label = tf.placeholder(tf.int64, [None, 1], name='label')  # 1 if same, 0 if different

    label_float = tf.to_float(label)
    margin = 0.5
    with tf.variable_scope("siamese") as scope:
        left_output = mnist_model(l_img)

        scope.reuse_variables()

        right_output = mnist_model(r_img)

    loss = contrastive_loss(left_output, right_output, label_float, margin)

    # Setup Optimizer
    global_step = tf.Variable(0, trainable=False)

    train_step = tf.train.AdamOptimizer(1.e-5).minimize(loss)
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # setup tensorboard
        tf.summary.scalar('step', global_step)
        tf.summary.scalar('loss', loss)

        writer = tf.summary.FileWriter('train.log', sess.graph)

        train_data_set = Dataset(mnist_data_train_images, mnist_data_train_labels)
        test_data_set = Dataset(mnist_data_test_images, mnist_data_test_labels)

        for i in range(50000):
            l_imgs, r_imgs, lbs = train_data_set._get_siamese_batch(100)
            _ = sess.run([train_step], feed_dict={left: l_imgs,
                                                  right: r_imgs,
                                                  label: lbs})
            if i % 1000 == 0:
                l_imgs, r_imgs, lbs = test_data_set._get_siamese_batch(100)
                l = sess.run([loss], feed_dict={left: l_imgs,
                                                right: r_imgs,
                                                label: lbs})
                print(i, l)


if __name__ == '__main__':
    main()
