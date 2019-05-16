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

BATCH_SIZE = 500
TOTAL_BATCH = train_labels.shape[0] // BATCH_SIZE

STDDEV_ = 0.1

x_input = tf.placeholder(tf.float32, shape=[None, WIDTH, HEIGHT, CHANEL], name='x_input')
y_input = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name='y_input')

C1, C2, C3 = 30, 50, 80
F1 = 500

# layer 1 ===============================================================================
w1_1 = tf.Variable(tf.truncated_normal([3, 3, CHANEL, C1], stddev=STDDEV_), name="w1_1")
b1_1 = tf.Variable(tf.constant(0.1, shape=[C1]), name="b1_1")
conv1_1 = tf.nn.conv2d(input=x_input,
                       filter=w1_1,
                       strides=[1, 1, 1, 1],
                       padding='SAME')
conv1_1 = conv1_1 + b1_1
conv1_1 = tf.nn.relu(conv1_1, name='conv1_1')

w1_2 = tf.Variable(tf.truncated_normal([3, 3, C1, C1], stddev=STDDEV_), name='w1_2')
b1_2 = tf.Variable(tf.constant(0.1, shape=[C1]), name='b1_2')
conv1_2 = tf.nn.conv2d(input=conv1_1,
                       filter=w1_2,
                       strides=[1, 1, 1, 1],
                       padding='SAME')
conv1_2 = conv1_2 + b1_2
conv1_2 = tf.nn.relu(conv1_2, name='conv1_2')

w1_3 = tf.Variable(tf.truncated_normal([3, 3, C1, C1], stddev=STDDEV_), name='w1_3')
b1_3 = tf.Variable(tf.constant(0.1, shape=[C1]), name='b1_3')
conv1_3 = tf.nn.conv2d(input=conv1_2,
                       filter=w1_3,
                       strides=[1, 1, 1, 1],
                       padding='SAME')
conv1_3 = conv1_3 + b1_3
conv1_3 = tf.nn.relu(conv1_3, name='conv1_3')

conv1_pool = tf.nn.max_pool(conv1_3,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            name="conv1_pool")

keep_prob = tf.placeholder(tf.float32)
conv1_drop = tf.nn.dropout(conv1_pool, keep_prob=keep_prob)

# Layer 2 ===============================================================================
w2_1 = tf.Variable(tf.truncated_normal([3, 3, C1, C2], stddev=STDDEV_), name='w2_1')
b2_1 = tf.Variable(tf.constant(0.1, shape=[C2]), name='b2_1')
conv2_1 = tf.nn.conv2d(input=conv1_drop,
                       filter=w2_1,
                       strides=[1, 1, 1, 1],
                       padding='SAME')
conv2_1 = conv2_1 + b2_1
conv2_1 = tf.nn.relu(conv2_1, name='conv2_1')

w2_2 = tf.Variable(tf.truncated_normal([3, 3, C2, C2], stddev=STDDEV_), name='w2_2')
b2_2 = tf.Variable(tf.constant(0.1, shape=[C2]), name='b2_2')
conv2_2 = tf.nn.conv2d(input=conv2_1,
                       filter=w2_2,
                       strides=[1, 1, 1, 1],
                       padding='SAME')
conv2_2 = conv2_2 + b2_2
conv2_2 = tf.nn.relu(conv2_2, name='conv2_2')

w2_3 = tf.Variable(tf.truncated_normal([3, 3, C2, C2], stddev=STDDEV_), name='w2_3')
b2_3 = tf.Variable(tf.constant(0.1, shape=[C2]), name='b2_3')
conv2_3 = tf.nn.conv2d(input=conv2_2,
                       filter=w2_3,
                       strides=[1, 1, 1, 1],
                       padding='SAME')
conv2_3 = conv2_3 + b2_3
conv2_3 = tf.nn.relu(conv2_3, name='conv2_3')
conv2_pool = tf.nn.max_pool(conv2_3,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            name='conv2_pool')
conv2_drop = tf.nn.dropout(conv2_pool, keep_prob=keep_prob)

# layer 3 ===============================================================================
w3_1 = tf.Variable(tf.truncated_normal([3, 3, C2, C3], stddev=STDDEV_), name='w3_1')
b3_1 = tf.Variable(tf.constant(0.1, shape=[C3]), name='b3_1')
conv3_1 = tf.nn.conv2d(input=conv2_drop,
                       filter=w3_1,
                       strides=[1, 1, 1, 1],
                       padding='SAME')
conv3_1 = conv3_1 + b3_1
conv3_1 = tf.nn.relu(conv3_1, name='conv3_1')

w3_2 = tf.Variable(tf.truncated_normal([3, 3, C3, C3], stddev=STDDEV_), name='w3_2')
b3_2 = tf.Variable(tf.constant(0.1, shape=[C3]), name='b3_2')
conv3_2 = tf.nn.conv2d(input=conv3_1,
                       filter=w3_2,
                       strides=[1, 1, 1, 1],
                       padding='SAME')
conv3_2 = conv3_2 + b3_2
conv3_2 = tf.nn.relu(conv3_2, name='conv3_2')

w3_3 = tf.Variable(tf.truncated_normal([3, 3, C3, C3], stddev=STDDEV_), name='w3_3')
b3_3 = tf.Variable(tf.constant(0.1, shape=[C3]), name='b3_3')
conv3_3 = tf.nn.conv2d(input=conv3_2,
                       filter=w3_3,
                       strides=[1, 1, 1, 1],
                       padding='SAME')
conv3_3 = conv3_3 + b3_3
conv3_3 = tf.nn.relu(conv3_3, name='conv3_3')

conv3_pool = tf.nn.max_pool(conv3_3,
                            ksize=[1, 8, 8, 1],
                            strides=[1, 8, 8, 1],
                            padding='SAME',
                            name='conv3_pool')
conv3_flat = tf.reshape(conv3_pool, shape=[-1, C3], name='conv3_flat')
conv3_drop = tf.nn.dropout(conv3_flat, keep_prob=keep_prob, name='conv3_drop')

# fully layer
w4 = tf.Variable(tf.truncated_normal(shape=[C3, F1], stddev=STDDEV_), name='w4')
b4 = tf.Variable(tf.constant(0.1, shape=[F1]), name='b4')
full1 = tf.add(tf.matmul(conv3_drop, w4), b4)
full1 = tf.nn.relu(full1, name='full1')

full1_drop = tf.nn.dropout(full1, keep_prob=keep_prob)

# output layer
w5 = tf.Variable(tf.truncated_normal(shape=[F1, NUM_CLASSES], stddev=STDDEV_), name='w5')
b5 = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]), name='b5')
logits = tf.add(tf.matmul(full1_drop, w5), b5, name='logits')

with tf.name_scope("loss") as scope:
    softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                    labels=y_input)

    loss_operation = tf.reduce_mean(softmax_cross_entropy, name="loss")

    tf.summary.scalar("loss", loss_operation)

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

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    train_summary_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, name) + "/train", session.graph)
    test_summary_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, name) + "/test")

    test_images = d.test.images
    test_labels = d.test.labels

    for epoch in range(10000):
        # id = np.random.randint(1, TOTAL_BATCH)
        # train_img = d.train.nex_batch(BATCH_SIZE)[0]
        # train_lab = d.train.nex_batch(BATCH_SIZE)[1]

        batch = d.train.nex_batch(BATCH_SIZE)

        _, merged_sum = session.run([optimiser, merged_summary_operation],
                                    feed_dict={x_input: batch[0],
                                               y_input: batch[1],
                                               keep_prob: 0.5})

        train_summary_writer.add_summary(merged_sum, epoch)

        if epoch % 100 == 0:
            X = d.test.images.reshape(10, 1000, 32, 32, 3)
            Y = d.test.labels.reshape(10, 1000, 10)

            acc, loss, merged_sum = session.run([accuracy_operation, loss_operation, merged_summary_operation],
                                                feed_dict={x_input: X,
                                                           y_input: Y,
                                                           keep_prob: 1.0})

            test_summary_writer.add_summary(merged_sum, epoch)
            # acc, loss = session.run([accuracy_operation, loss_operation],
            #                                     feed_dict={x_input: batch[0],
            #                                                y_input: batch[1],
            #                                                keep_prob: 1.0})

            print("EPOCH: {}, ACC: {:.4}%, LOSS: {}".format(epoch, acc * 100, loss))
