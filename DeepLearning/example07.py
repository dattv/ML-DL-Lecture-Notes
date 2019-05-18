import os
import pickle as cPickle
import tarfile
import urllib.request

import cv2 as cv
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

if os.path.isfile(work_dir + "/vgg_16.ckpt") == True:
    vgg_16_dir = os.path.join(work_dir, "vgg_16.ckpt")

from tensorflow.python import pywrap_tensorflow

reader = pywrap_tensorflow.NewCheckpointReader(vgg_16_dir)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor name: {}".format(key))

# Recover VGG16 ===================================================================================================
TRAINABLE = False

x_input = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='x_input')
y_input = tf.placeholder(tf.float32, shape=[None, 1000], name='y_input')


class VGG16:
    def __init__(self, VGG_URL="http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz",
                 trainable=False, dropout=0.5):
        self._VGG_URL = VGG_URL
        self._trainable = trainable
        self._dropout = dropout

        work_dir = os.getcwd()
        work_dir = os.path.join(work_dir, file_name.split(".")[0])
        if os.path.isdir(work_dir) == False:
            os.mkdir(work_dir)

        file_path = os.path.join(work_dir, file_name)

        if not os.path.exists(file_path):
            with tqdm(unit="B", unit_scale=True, leave=True, miniters=1, desc=model_url.split("/")[-1]) as t:
                file_path, _ = urllib.request.urlretrieve(model_url, filename=file_path, reporthook=my_hook(t),
                                                          data=None)

        tarfile.open(file_path, "r:gz").extractall(work_dir)

        if os.path.isfile(work_dir + "/vgg_16.ckpt") == True:
            vgg_16_dir = os.path.join(work_dir, "vgg_16.ckpt")

        from tensorflow.python import pywrap_tensorflow

        reader = pywrap_tensorflow.NewCheckpointReader(vgg_16_dir)
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in var_to_shape_map:
            print("tensor name: {}".format(key))

        # Layer 1
        self._weights_1_1 = reader.get_tensor("vgg_16/conv1/conv1_1/weights")
        self._t_weights_1_1 = tf.Variable(initial_value=self._weights_1_1, name='weights', trainable=TRAINABLE)

        self._bias_1 = reader.get_tensor("vgg_16/conv1/conv1_1/biases")
        self._t_bias_1_1 = tf.Variable(initial_value=self._bias_1, name='biases', trainable=TRAINABLE)

        self._weights_1_2 = reader.get_tensor("vgg_16/conv1/conv1_2/weights")
        self._t_weights_1_2 = tf.Variable(initial_value=self._weights_1_2, name='weights', trainable=TRAINABLE)

        self._bias_1_2 = reader.get_tensor("vgg_16/conv1/conv1_2/biases")
        self._t_biases_1_2 = tf.Variable(initial_value=self._bias_1_2, name='biases', trainable=TRAINABLE)

        # Layer 2
        self._weights_2_1 = reader.get_tensor("vgg_16/conv2/conv2_1/weights")
        self._t_weights_2_1 = tf.Variable(initial_value=self._weights_2_1, name='weights', trainable=TRAINABLE)

        self._biases_2_1 = reader.get_tensor("vgg_16/conv2/conv2_1/biases")
        self._t_biases_2_1 = tf.Variable(initial_value=self._biases_2_1, name='biases', trainable=TRAINABLE)

        self._weights_2_2 = reader.get_tensor("vgg_16/conv2/conv2_2/weights")
        self._t_weights_2_2 = tf.Variable(initial_value=self._weights_2_2, name='weights', trainable=TRAINABLE)

        self._biases_2_2 = reader.get_tensor("vgg_16/conv2/conv2_2/biases")
        self._t_biases_2_2 = tf.Variable(initial_value=self._biases_2_2, name='biases', trainable=TRAINABLE)

        # Layer 3
        self._weights_3_1 = reader.get_tensor("vgg_16/conv3/conv3_1/weights")
        self._t_weights_3_1 = tf.Variable(initial_value=self._weights_3_1, name='weights', trainable=TRAINABLE)

        self._biases_3_1 = reader.get_tensor("vgg_16/conv3/conv3_1/biases")
        self._t_biases_3_1 = tf.Variable(initial_value=self._biases_3_1, name='biases', trainable=TRAINABLE)

        self._weights_3_2 = reader.get_tensor("vgg_16/conv3/conv3_2/weights")
        self._t_weights_3_2 = tf.Variable(initial_value=self._weights_3_2, name='weights', trainable=TRAINABLE)

        self._biases_3_2 = reader.get_tensor("vgg_16/conv3/conv3_2/biases")
        self._t_biases_3_2 = tf.Variable(initial_value=self._biases_3_2, name='biases', trainable=TRAINABLE)

        self._weights_3_3 = reader.get_tensor("vgg_16/conv3/conv3_3/weights")
        self._t_weights_3_3 = tf.Variable(initial_value=self._weights_3_3, name='weights', trainable=TRAINABLE)

        self._biases_3_3 = reader.get_tensor("vgg_16/conv3/conv3_3/biases")
        self._t_biases_3_3 = tf.Variable(initial_value=self._biases_3_3, name='biases', trainable=TRAINABLE)

        # Layer 4
        self._weights_4_1 = reader.get_tensor("vgg_16/conv4/conv4_1/weights")
        self._t_weights_4_1 = tf.Variable(initial_value=self._weights_4_1, name='weights', trainable=TRAINABLE)

        self._biases_4_1 = reader.get_tensor("vgg_16/conv4/conv4_1/biases")
        self._t_biases_4_1 = tf.Variable(initial_value=self._biases_4_1, name='biases', trainable=TRAINABLE)

        self._weights_4_2 = reader.get_tensor("vgg_16/conv4/conv4_2/weights")
        self._t_weights_4_2 = tf.Variable(initial_value=self._weights_4_2, name='weights', trainable=TRAINABLE)

        self._biases_4_2 = reader.get_tensor("vgg_16/conv4/conv4_2/biases")
        self._t_biases_4_2 = tf.Variable(initial_value=self._biases_4_2, name='biases', trainable=TRAINABLE)

        self._weights_4_3 = reader.get_tensor("vgg_16/conv4/conv4_3/weights")
        self._t_weights_4_3 = tf.Variable(initial_value=self._weights_4_3, name='weights', trainable=TRAINABLE)

        self._biases_4_3 = reader.get_tensor("vgg_16/conv4/conv4_3/biases")
        self._t_biases_4_3 = tf.Variable(initial_value=self._biases_4_3, name='biases', trainable=TRAINABLE)

        # Layer 5
        self._weights_5_1 = reader.get_tensor("vgg_16/conv5/conv5_1/weights")
        self._t_weights_5_1 = tf.Variable(initial_value=self._weights_5_1, name='weights', trainable=TRAINABLE)

        self._biases_5_1 = reader.get_tensor("vgg_16/conv5/conv5_1/biases")
        self._t_biases_5_1 = tf.Variable(initial_value=self._biases_5_1, name='biases', trainable=TRAINABLE)

        self._weights_5_2 = reader.get_tensor("vgg_16/conv5/conv5_2/weights")
        self._t_weights_5_2 = tf.Variable(initial_value=self._weights_5_2, name='weights', trainable=TRAINABLE)

        self._biases_5_2 = reader.get_tensor("vgg_16/conv5/conv5_2/biases")
        self._t_biases_5_2 = tf.Variable(initial_value=self._biases_5_2, name='biases', trainable=TRAINABLE)

        self._weights_5_3 = reader.get_tensor("vgg_16/conv5/conv5_3/weights")
        self._t_weights_5_3 = tf.Variable(initial_value=self._weights_5_3, name='weights', trainable=TRAINABLE)

        self._biases_5_3 = reader.get_tensor("vgg_16/conv5/conv5_3/biases")
        self._t_biases_5_3 = tf.Variable(initial_value=self._biases_5_3, name='biases', trainable=TRAINABLE)

        # Layer 6
        self._weights_6 = reader.get_tensor("vgg_16/fc6/weights")
        self._t_weights_6 = tf.Variable(initial_value=self._weights_6, name='weights', trainable=TRAINABLE)

        self._biases_6 = reader.get_tensor("vgg_16/fc6/biases")
        self._t_biases_6 = tf.Variable(initial_value=self._biases_6, name='biases', trainable=TRAINABLE)

        # Layer 7
        self._weights_7 = reader.get_tensor("vgg_16/fc7/weights")
        self._t_weights_7 = tf.Variable(initial_value=self._weights_7, name='weights', trainable=TRAINABLE)

        self._biases_7 = reader.get_tensor("vgg_16/fc7/biases")
        self._t_biases_7 = tf.Variable(initial_value=self._biases_7, name='biases', trainable=TRAINABLE)

        # Layer 8
        self._weights_8 = reader.get_tensor("vgg_16/fc8/weights")
        self._t_weights_8 = tf.Variable(initial_value=self._weights_8, name='weights', trainable=True)

        self._biases_8 = reader.get_tensor("vgg_16/fc8/biases")
        self._t_biases_8 = tf.Variable(initial_value=self._biases_8, name='biases', trainable=True)

    def build_Net(self, x_input):
        with tf.name_scope("vgg_16") as scope:
            with tf.name_scope("conv1") as scope:
                with tf.name_scope("conv1_1") as scope:
                    self._conv1_1 = tf.nn.conv2d(input=x_input,
                                                 filter=self._t_weights_1_1,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv1_1 = self._conv1_1 + self._t_bias_1_1
                    self._conv1_1 = tf.nn.relu(self._conv1_1, name='activation')

                with tf.name_scope("conv1_2") as scope:
                    self._conv1_2 = tf.nn.conv2d(input=self._conv1_1,
                                                 filter=self._t_weights_1_2,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv1_2 = self._conv1_2 + self._t_biases_1_2
                    self._conv1_2 = tf.nn.relu(self._conv1_2, name='activation')

            with tf.name_scope("pool1") as scope:
                self._pooling1 = tf.nn.max_pool(self._conv1_2,
                                                ksize=[1, 2, 2, 1],
                                                strides=[1, 2, 2, 1],
                                                padding='VALID',
                                                name='pooling1')

            with tf.name_scope("conv2") as scope:
                with tf.name_scope("conv2_1") as scope:
                    self._conv2_1 = tf.nn.conv2d(input=self._pooling1,
                                                 filter=self._t_weights_2_1,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv2_1 = self._conv2_1 + self._t_biases_2_1
                    self._conv2_1 = tf.nn.relu(self._conv2_1)

                with tf.name_scope("conv2_2") as scope:
                    self._conv2_2 = tf.nn.conv2d(input=self._conv2_1,
                                                 filter=self._t_weights_2_2,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv2_2 = self._conv2_2 + self._t_biases_2_2
                    self._conv2_2 = tf.nn.relu(self._conv2_2, name='conv2_2')

            with tf.name_scope("pool2") as scope:
                self._pooling2 = tf.nn.max_pool(self._conv2_2,
                                                ksize=[1, 2, 2, 1],
                                                strides=[1, 2, 2, 1],
                                                padding='VALID',
                                                name='pooling2')

            with tf.name_scope("conv3") as scope:
                with tf.name_scope("conv3_1") as scope:
                    self._conv3_1 = tf.nn.conv2d(input=self._pooling2,
                                                 filter=self._t_weights_3_1,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv3_1 = self._conv3_1 + self._t_biases_3_1
                    self._conv3_1 = tf.nn.relu(self._conv3_1, name='conv3_1')

                with tf.name_scope("conv3_2") as scope:
                    self._conv3_2 = tf.nn.conv2d(input=self._conv3_1,
                                                 filter=self._t_weights_3_2,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv3_2 = self._conv3_2 + self._t_biases_3_2
                    self._conv3_2 = tf.nn.relu(self._conv3_2, name='conv3_2')

                with tf.name_scope("conv3_3") as scope:
                    self._conv3_3 = tf.nn.conv2d(input=self._conv3_2,
                                                 filter=self._t_weights_3_3,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv3_3 = self._conv3_3 + self._t_biases_3_3
                    self._conv3_3 = tf.nn.relu(self._conv3_3, name='conv3_3')

            with tf.name_scope("pool3") as scope:
                self._pooling3 = tf.nn.max_pool(self._conv3_3,
                                                ksize=[1, 2, 2, 1],
                                                strides=[1, 2, 2, 1],
                                                padding='VALID',
                                                name='pooling3')

            with tf.name_scope("conv4") as scope:
                with tf.name_scope("conv4_1") as scope:
                    self._conv4_1 = tf.nn.conv2d(input=self._pooling3,
                                                 filter=self._t_weights_4_1,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv4_1 = self._conv4_1 + self._t_biases_4_1
                    self._conv4_1 = tf.nn.relu(self._conv4_1, name='conv4_1')

                with tf.name_scope("conv4_2") as scope:
                    self._conv4_2 = tf.nn.conv2d(input=self._conv4_1,
                                                 filter=self._t_weights_4_2,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv4_2 = self._conv4_2 + self._t_biases_4_2
                    self._conv4_2 = tf.nn.relu(self._conv4_2, name='conv4_2')

                with tf.name_scope("conv4_3") as scope:
                    self._conv4_3 = tf.nn.conv2d(input=self._conv4_2,
                                                 filter=self._t_weights_4_3,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv4_3 = self._conv4_3 + self._t_biases_4_3
                    self._conv4_3 = tf.nn.relu(self._conv4_3, name='conv4_3')

            with tf.name_scope("pool4") as scope:
                self._pooling4 = tf.nn.max_pool(self._conv4_3,
                                                ksize=[1, 2, 2, 1],
                                                strides=[1, 2, 2, 1],
                                                padding='VALID',
                                                name='pooling4')

            with tf.name_scope("conv5") as scope:
                with tf.name_scope("conv5_1") as scope:
                    self._conv5_1 = tf.nn.conv2d(input=self._pooling4,
                                                 filter=self._t_weights_5_1,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv5_1 = self._conv5_1 + self._t_biases_5_1
                    self._conv5_1 = tf.nn.relu(self._conv5_1, name='conv5_1')

                with tf.name_scope("conv5_2") as scope:
                    self._conv5_2 = tf.nn.conv2d(input=self._conv5_1,
                                                 filter=self._t_weights_5_2,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv5_2 = self._conv5_2 + self._t_biases_5_2
                    self._conv5_2 = tf.nn.relu(self._conv5_2, name='conv5_2')

                with tf.name_scope("conv5_3") as scope:
                    self._conv5_3 = tf.nn.conv2d(input=self._conv5_2,
                                                 filter=self._t_weights_5_3,
                                                 strides=[1, 1, 1, 1],
                                                 padding='SAME')
                    self._conv5_3 = self._conv5_3 + self._t_biases_5_3
                    self._conv5_3 = tf.nn.relu(self._conv5_3, name='conv5_3')

            with tf.name_scope("pool5") as scope:
                self._pooling5 = tf.nn.max_pool(self._conv5_3,
                                                ksize=[1, 2, 2, 1],
                                                strides=[1, 2, 2, 1],
                                                padding='VALID',
                                                name='pooling5')

            with tf.name_scope("dense6") as scope:
                self._fc6 = tf.nn.conv2d(input=self._pooling5,
                                         filter=self._t_weights_6,
                                         strides=[1, 1, 1, 1],
                                         padding='VALID')
                self._fc6 = self._fc6 + self._t_biases_6

                self._fc6 = tf.nn.relu(self._fc6, name='fc6')

            with tf.name_scope("dense7") as scope:
                self._fc7 = tf.nn.conv2d(input=self._fc6,
                                         filter=self._t_weights_7,
                                         strides=[1, 1, 1, 1],
                                         padding='VALID')

                self._fc7 = self._fc7 + self._t_biases_7
                self._fc7 = tf.nn.relu(self._fc7, name='fc7')

            with tf.name_scope("dense8") as scope:
                self._fc8 = tf.nn.conv2d(input=self._fc7,
                                         filter=self._t_weights_8,
                                         strides=[1, 1, 1, 1],
                                         padding='VALID')
                self._fc8 = self._fc8 + self._t_biases_8
                self._fc8 = tf.nn.softmax(self._fc8, name='fc8')

        return self._fc8

    def VGG_tranfer_learning(self, sess=None, images=None, labels=None, NEPOCH=1000, NBATCH=128):

        if sess == None or images == None or labels == None:
            return None
        else:
            n_classes = len(labels[0])

            vgg_without_top = self._fc7


VGG16_1000 = VGG16()
ouput = VGG16_1000.build_Net(x_input)

# test VGG16_1000
cifa = CifarDataManager()
test_img = cifa.test.images
test_labels = cifa.test.labels

merged_summary_operation = tf.summary.merge_all()
LOG_DIR = "./tmp"
file_name = os.path.basename(__file__)
file_name = file_name.split(".")[0]

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    test_summary = tf.summary.FileWriter(os.path.join(LOG_DIR, file_name) + "/test_vgg16_1000", session.graph)

    img = cv.imread("./vgg_16_2016_08_28/tiger.jpeg")
    # cv.imshow('image', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    img = cv.resize(img, (224, 224))
    img = img.reshape((1, 224, 224, 3))

    out = session.run([ouput], feed_dict={x_input: img})

    out = np.reshape(out, newshape=[-1])

    print(out[np.argmax(out)])
    print(np.argmax(out))

    SYNSET_DIR = './vgg_16_2016_08_28/synset.txt'
    synset = [l.strip() for l in open(SYNSET_DIR).readlines()]

    title0 = np.argsort(out)[::-1]
    top1_title0 = synset[title0[0]]

    print(top1_title0)
