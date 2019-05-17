import os
import pickle as cPickle
import tarfile
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

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

# downloading tensorflow model ====================================================================================
model_url = "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"
file_name = model_url.split("/")[-1]

work_dir = os.getcwd()
work_dir = os.path.join(work_dir, file_name.split(".")[0])
if os.path.isdir(work_dir) == False:
    os.mkdir(work_dir)

file_path = os.path.join(work_dir, file_name)

if not os.path.exists(file_path):
    with tqdm(unit="B", unit_scale=True, leave=True, miniters=1, desc=model_url.split("/")[-1]) as t:
        file_path, _ = urllib.request.urlretrieve(model_url, filename=file_path, reporthook=my_hook(t), data=None)

tarfile.open(file_path, "r:gz").extractall(work_dir)

# extract VGG16 model =============================================================================================

# x = tf.get_variable("", shape=[224, 224, 3])
if os.path.isfile(work_dir + "/vgg_16.ckpt") == True:
    vgg_16_dir = os.path.join(work_dir, "vgg_16.ckpt")

from tensorflow.python import pywrap_tensorflow

reader = pywrap_tensorflow.NewCheckpointReader(vgg_16_dir)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor name: {}".format(key))

TRAINABLE = False

x_input = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='x_input')
y_input = tf.placeholder(tf.float32, shape=[None, 1000], name='y_input')

with tf.name_scope("vgg_16") as scope:
    with tf.name_scope("conv1") as scope:
        with tf.name_scope("conv1_1") as scope:
            weights_1_1 = reader.get_tensor("vgg_16/conv1/conv1_1/weights")
            t_weights_1_1 = tf.Variable(initial_value=weights_1_1, name='weights', trainable=TRAINABLE)

            bias_1_ = reader.get_tensor("vgg_16/conv1/conv1_1/biases")
            t_bias_1_1 = tf.Variable(initial_value=bias_1_, name='biases', trainable=TRAINABLE)

            conv1_1 = tf.nn.conv2d(input=x_input,
                                   filter=t_weights_1_1,
                                   strides=[1, 1, 1, 1],
                                   padding='SAME')
            conv1_1 = conv1_1 + t_bias_1_1
            conv1_1 = tf.nn.relu(conv1_1, name='activation')

        with tf.name_scope("conv1_2") as scope:
            weights_1_2 = reader.get_tensor("vgg_16/conv1/conv1_2/weights")
            t_weights_1_2 = tf.Variable(initial_value=weights_1_2, name='weights', trainable=TRAINABLE)

            bias_1_2 = reader.get_tensor("vgg_16/conv1/conv1_2/biases")
            t_biases_1_2 = tf.Variable(initial_value=bias_1_2, name='biases', trainable=TRAINABLE)

            conv1_2 = tf.nn.conv2d(input=conv1_1,
                                   filter=t_weights_1_2,
                                   strides=[1, 1, 1, 1],
                                   padding='SAME')
            conv1_2 = conv1_2 + t_biases_1_2
            conv1_2 = tf.nn.relu(conv1_2, name='activation')

    with tf.name_scope("pool1") as scope:
        pooling1 = tf.nn.max_pool(conv1_2,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='VALID',
                                  name='pooling1')

    with tf.name_scope("conv2") as scope:
        with tf.name_scope("conv2_1") as scope:
            weights_2_1 = reader.get_tensor("vgg_16/conv2/conv2_1/weights")
            t_weights_2_1 = tf.Variable(initial_value=weights_2_1, name='weights', trainable=TRAINABLE)

            biases_2_1 = reader.get_tensor("vgg_16/conv2/conv2_1/biases")
            t_biases_2_1 = tf.Variable(initial_value=biases_2_1, name='biases', trainable=TRAINABLE)

            conv2_1 = tf.nn.conv2d(input=pooling1,
                                   filter=t_weights_2_1,
                                   strides=[1, 1, 1, 1],
                                   padding='SAME')
            conv2_1 = conv2_1 + t_biases_2_1
            conv2_1 = tf.nn.relu(conv2_1)


        with tf.name_scope("conv2_2") as scope:
            weights_2_2 = reader.get_tensor("vgg_16/conv2/conv2_2/weights")
            t_weights_2_2 = tf.Variable(initial_value=weights_2_2, name='weights', trainable=TRAINABLE)

            biases_2_2 = reader.get_tensor("vgg_16/conv2/conv2_2/biases")
            t_biases_2_2 = tf.Variable(initial_value=biases_2_2, name='biases', trainable=TRAINABLE)

            conv2_2 = tf.nn.conv2d(input=conv2_1,
                                   filter=t_weights_2_2,
                                   strides=[1, 1, 1, 1],
                                   padding='SAME')
            conv2_2 = conv2_2 + t_biases_2_2
            conv2_2 = tf.nn.relu(conv2_2, name='conv2_2')

    with tf.name_scope("pool2") as scope:
        pooling2 = tf.nn.max_pool(conv2_2,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='VALID',
                                  name='pooling2')

    with tf.name_scope("conv3") as scope:
        with tf.name_scope("conv3_1") as scope:
            weights_3_1 = reader.get_tensor("vgg_16/conv3/conv3_1/weights")
            t_weights_3_1 = tf.Variable(initial_value=weights_3_1, name='weights', trainable=TRAINABLE)

            biases_3_1 = reader.get_tensor("vgg_16/conv3/conv3_1/biases")
            t_biases_3_1 = tf.Variable(initial_value=biases_3_1, name='biases', trainable=TRAINABLE)

            conv3_1 = tf.nn.conv2d(input=pooling2,
                                   filter=t_weights_3_1,
                                   strides=[1, 1, 1, 1],
                                   padding='SAME')
            conv3_1 = conv3_1 + t_biases_3_1
            conv3_1 = tf.nn.relu(conv3_1, name='conv3_1')

        with tf.name_scope("conv3_2") as scope:
            weights_3_2 = reader.get_tensor("vgg_16/conv3/conv3_2/weights")
            t_weights_3_2 = tf.Variable(initial_value=weights_3_2, name='weights', trainable=TRAINABLE)

            biases_3_2 = reader.get_tensor("vgg_16/conv3/conv3_2/biases")
            t_biases_3_2 = tf.Variable(initial_value=biases_3_2, name='biases', trainable=TRAINABLE)

            conv3_2 = tf.nn.conv2d(input=conv3_1,
                                   filter=t_weights_3_2,
                                   strides=[1, 1, 1, 1],
                                   padding='SAME')
            conv3_2 = conv3_2 + t_biases_3_2
            conv3_2 = tf.nn.relu(conv3_2, name='conv3_2')

        with tf.name_scope("conv3_3") as scope:
            weights_3_3 = reader.get_tensor("vgg_16/conv3/conv3_3/weights")
            t_weights_3_3 = tf.Variable(initial_value=weights_3_3, name='weights', trainable=TRAINABLE)

            biases_3_3 = reader.get_tensor("vgg_16/conv3/conv3_3/biases")
            t_biases_3_3 = tf.Variable(initial_value=biases_3_3, name='biases', trainable=TRAINABLE)

            conv3_3 = tf.nn.conv2d(input=conv3_2,
                                   filter=t_weights_3_3,
                                   strides=[1, 1, 1, 1],
                                   padding='SAME')
            conv3_3 = conv3_3 + t_biases_3_3
            conv3_3 = tf.nn.relu(conv3_3, name='conv3_3')

    with tf.name_scope("pool3") as scope:
        pooling3 = tf.nn.max_pool(conv3_3,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='VALID',
                                  name='pooling3')

    with tf.name_scope("conv4") as scope:
        with tf.name_scope("conv4_1") as scope:
            weights_4_1 = reader.get_tensor("vgg_16/conv4/conv4_1/weights")
            t_weights_4_1 = tf.Variable(initial_value=weights_4_1, name='weights', trainable=TRAINABLE)

            biases_4_1 = reader.get_tensor("vgg_16/conv4/conv4_1/biases")
            t_biases_4_1 = tf.Variable(initial_value=biases_4_1, name='biases', trainable=TRAINABLE)

            conv4_1 = tf.nn.conv2d(input=pooling3,
                                   filter=t_weights_4_1,
                                   strides=[1, 1, 1, 1],
                                   padding='SAME')
            conv4_1 = conv4_1 + t_biases_4_1
            conv4_1 = tf.nn.relu(conv4_1, name='conv4_1')

        with tf.name_scope("conv4_2") as scope:
            weights_4_2 = reader.get_tensor("vgg_16/conv4/conv4_2/weights")
            t_weights_4_2 = tf.Variable(initial_value=weights_4_2, name='weights', trainable=TRAINABLE)

            biases_4_2 = reader.get_tensor("vgg_16/conv4/conv4_2/biases")
            t_biases_4_2 = tf.Variable(initial_value=biases_4_2, name='biases', trainable=TRAINABLE)

            conv4_2 = tf.nn.conv2d(input=conv4_1,
                                   filter=t_weights_4_2,
                                   strides=[1, 1, 1, 1],
                                   padding='SAME')
            conv4_2 = conv4_2 + t_biases_4_2
            conv4_2 = tf.nn.relu(conv4_2, name='conv4_2')

        with tf.name_scope("conv4_3") as scope:
            weights_4_3 = reader.get_tensor("vgg_16/conv4/conv4_3/weights")
            t_weights_4_3 = tf.Variable(initial_value=weights_4_3, name='weights', trainable=TRAINABLE)

            biases_4_3 = reader.get_tensor("vgg_16/conv4/conv4_3/biases")
            t_biases_4_3 = tf.Variable(initial_value=biases_4_3, name='biases', trainable=TRAINABLE)

            conv4_3 = tf.nn.conv2d(input=conv4_2,
                                   filter=t_weights_4_3,
                                   strides=[1, 1, 1, 1],
                                   padding='SAME')
            conv4_3 = conv4_3 + t_biases_4_3
            conv4_3 = tf.nn.relu(conv4_3, name='conv4_3')

    with tf.name_scope("pool4") as scope:
        pooling4 = tf.nn.max_pool(conv4_3,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='VALID',
                                  name='pooling4')

    with tf.name_scope("conv5") as scope:
        with tf.name_scope("conv5_1") as scope:
            weights_5_1 = reader.get_tensor("vgg_16/conv5/conv5_1/weights")
            t_weights_5_1 = tf.Variable(initial_value=weights_5_1, name='weights', trainable=TRAINABLE)

            biases_5_1 = reader.get_tensor("vgg_16/conv5/conv5_1/biases")
            t_biases_5_1 = tf.Variable(initial_value=biases_5_1, name='biases', trainable=TRAINABLE)

            conv5_1 = tf.nn.conv2d(input=pooling4,
                                   filter=t_weights_5_1,
                                   strides=[1, 1, 1, 1],
                                   padding='SAME')
            conv5_1 = conv5_1 + t_biases_5_1
            conv5_1 = tf.nn.relu(conv5_1, name='conv5_1')

        with tf.name_scope("conv5_2") as scope:
            weights_5_2 = reader.get_tensor("vgg_16/conv5/conv5_2/weights")
            t_weights_5_2 = tf.Variable(initial_value=weights_5_2, name='weights', trainable=TRAINABLE)

            biases_5_2 = reader.get_tensor("vgg_16/conv5/conv5_2/biases")
            t_biases_5_2 = tf.Variable(initial_value=biases_5_2, name='biases', trainable=TRAINABLE)

            conv5_2 = tf.nn.conv2d(input=conv5_1,
                                   filter=t_weights_5_2,
                                   strides=[1, 1, 1, 1],
                                   padding='SAME')
            conv5_2 = conv5_2 + t_biases_5_2
            conv5_2 = tf.nn.relu(conv5_2, name='conv5_2')

        with tf.name_scope("conv5_3") as scope:
            weights_5_3 = reader.get_tensor("vgg_16/conv5/conv5_3/weights")
            t_weights_5_3 = tf.Variable(initial_value=weights_5_3, name='weights', trainable=TRAINABLE)

            biases_5_3 = reader.get_tensor("vgg_16/conv5/conv5_3/biases")
            t_biases_5_3 = tf.Variable(initial_value=biases_5_3, name='biases', trainable=TRAINABLE)

            conv5_3 = tf.nn.conv2d(input=conv5_2,
                                   filter=t_weights_5_3,
                                   strides=[1, 1, 1, 1],
                                   padding='SAME')
            conv5_3 = conv5_3 + t_biases_5_3
            conv5_3 = tf.nn.relu(conv5_3, name='conv5_3')

    with tf.name_scope("pool5") as scope:
        pooling5 = tf.nn.max_pool(conv5_3,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='VALID',
                                  name='pooling5')

    with tf.name_scope("dense6") as scope:
        weights_6 = reader.get_tensor("vgg_16/fc6/weights")
        t_weights_6 = tf.Variable(initial_value=weights_6, name='weights', trainable=TRAINABLE)

        biases_6 = reader.get_tensor("vgg_16/fc6/biases")
        t_biases_6 = tf.Variable(initial_value=biases_6, name='biases', trainable=TRAINABLE)

        fc6 = tf.nn.conv2d(input=pooling5,
                           filter=t_weights_6,
                           strides=[1, 1, 1, 1],
                           padding='VALID')
        fc6 = fc6 + t_biases_6
        fc6 = tf.nn.relu(fc6, name='fc6')

    with tf.name_scope("dense7") as scope:
        weights_7 = reader.get_tensor("vgg_16/fc7/weights")
        t_weights_7 = tf.Variable(initial_value=weights_7, name='weights', trainable=TRAINABLE)

        biases_7 = reader.get_tensor("vgg_16/fc7/weights")
        t_biases_7 = tf.Variable(initial_value=biases_7, name='biases', trainable=TRAINABLE)

        fc7 = tf.nn.conv2d(input=fc6,
                           filter=t_weights_7,
                           strides=[1, 1, 1, 1],
                           padding='VALID')
        fc7 = fc7 + t_biases_7
        fc7 = tf.nn.relu(fc7, name='fc7')

print(fc7)

print("jkdflkdsjf")
