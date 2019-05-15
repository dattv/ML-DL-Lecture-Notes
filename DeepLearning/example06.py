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

BATCH_SIZE = 100
TOTAL_BATCH = train_labels.shape[0] // BATCH_SIZE

STDDEV_ = 0.1

x_input = tf.placeholder(tf.float32, shape=[None, WIDTH, HEIGHT, CHANEL], name='x_input')
y_input = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name='y_input')

w1 = tf.Variable(tf.truncated_normal([3, 3, CHANEL, 64], stddev=STDDEV_), name='w1')
b1 = tf.Variable(tf.constant(0., shape=[64]), name='b1')

conv1 = tf.nn.conv2d(input=x_input,
                     filter=w1,
                     strides=[1, 1, 1, 1],
                     padding='VALID',
                     name='conv_1')

conv1 = conv1 + b1
convolution_layer_1 = tf.nn.relu(conv1)

pooling_layer_1 = tf.nn.max_pool(convolution_layer_1,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='VALID',
                                 name='pooling_layer_1')

w2 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=STDDEV_), name='w2')
b2 = tf.Variable(tf.constant(0., shape=[128]), name='b3')

conv2 = tf.nn.conv2d(input=pooling_layer_1,
                     filter=w2,
                     strides=[1, 1, 1, 1],
                     padding='VALID',
                     name='conv_2')

conv2 = conv2 + b2
convolution_layer_2 = tf.nn.relu(conv2)

pooling_layer_2 = tf.nn.max_pool(convolution_layer_2,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='VALID',
                                 name='pooling_layer_2')
new_shape = pooling_layer_2.shape[1] * pooling_layer_2.shape[2] * pooling_layer_2.shape[3]
flatten = tf.reshape(pooling_layer_2, shape=[-1, int(new_shape)], name='flatten')

w3 = tf.Variable(tf.truncated_normal(shape=[int(new_shape), 1024], stddev=STDDEV_), name='w3')
b3 = tf.Variable(tf.constant(0.1, shape=[1024]), name='b3')

dense_layer_bottleneck = tf.add(tf.matmul(flatten, w3), b3)
dense_layer_bottleneck = tf.nn.relu(dense_layer_bottleneck, name='dense_layer_bottleneck')

dropout_bool = tf.placeholder(tf.bool)
dropout_layer = tf.layers.dropout(
    inputs=dense_layer_bottleneck,
    rate=0.4,
    training=dropout_bool
)

w4 = tf.Variable(tf.truncated_normal(shape=[1024, NUM_CLASSES], stddev=STDDEV_), name='w4')
b4 = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]), name='b4')
logits = tf.add(tf.matmul(dropout_layer, w4), b4)
logits = tf.nn.relu(logits, name='logits')

with tf.name_scope("loss") as scope:
    softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                    labels=y_input)

    loss_operation = tf.reduce_mean(softmax_cross_entropy, name="loss")

    tf.summary.scalar("loss", loss_operation)

with tf.name_scope("optimization") as scope:
    optimiser = tf.train.AdamOptimizer().minimize(loss_operation)

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

    for epoch in range(100):
        id = np.random.randint(1, TOTAL_BATCH)
        train_img = d.train.nex_batch(BATCH_SIZE)[0]
        # train_img = train_images[(id - 1)*BATCH_SIZE: id*BATCH_SIZE]
        # train_lab = train_labels[(id - 1)*BATCH_SIZE: id*BATCH_SIZE]
        train_lab = d.train.nex_batch(BATCH_SIZE)[1]
        print(train_img.shape)
        print(train_lab.shape)

        _, merged_sum = session.run([optimiser, merged_summary_operation], feed_dict={x_input: train_images,
                                                                                      y_input: train_lab,
                                                                                      dropout_bool: True})
        train_summary_writer.add_summary(merged_sum, epoch)

        # if epoch % 10 == 0:
        #     acc, loss, merged_sum = session.run([accuracy_operation, loss_operation, merged_summary_operation],
        #                                         feed_dict={x_input: test_images,
        #                                                    y_input: test_labels,
        #                                                    dropout_bool: False})
        #
        #     test_summary_writer.add_summary(merged_sum, epoch)
        #
        #     print("EPOCH: {}, ACC: {}, LOSS: {}". format(epoch, acc, loss))
