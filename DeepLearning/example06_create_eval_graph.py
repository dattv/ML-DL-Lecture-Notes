import tensorflow as tf
import os

from DeepLearning.example06 import net, WideResNet

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


WIDTH, HEIGHT, CHANEL = train_images[0].shape
NUM_CLASSES = train_labels.shape[1]

BATCH_SIZE = 64
TOTAL_BATCH = train_labels.shape[0] // BATCH_SIZE

STDDEV_ = 0.1

x_input = tf.placeholder(tf.float32, shape=[None, WIDTH, HEIGHT, CHANEL], name='x_input')
y_input = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name='y_input')
# keep_prob = tf.placeholder(tf.float32, shape=[1], name="keep_prob")
is_training = False# tf.placeholder(tf.float32, name="is_training")

# logits = net(x_input, 1., is_training)
logits = WideResNet(x_input, is_training=is_training, keep_prob=1., depth=16, k=8)
graph = tf.get_default_graph()


# Call the eval rewrite which rewrites the graph in-place with
# FakeQuantization nodes and fold batchnorm for eval.
tf.contrib.quantize.create_eval_graph(input_graph=graph)

saver = tf.train.Saver()

with tf.Session(graph=graph) as session:
    checkpoint = tf.train.latest_checkpoint('./cifar_model/')
    saver.restore(session, checkpoint)
    # Save the checkpoint and eval graph proto to disk for freezing
    # and providing to TFLite.
    eval_graph_file = "./eval_graph_def.pb"
    checkpoint_name = "./checkpoint/checkpoint.ckpt"

    with open(eval_graph_file, "w") as f:
        f.write(str(graph.as_graph_def()))

    saver.save(session, checkpoint_name)

    builder = tf.saved_model.Builder('exports')

    signature_def = tf.saved_model.predict_signature_def(
        inputs={'x_input': x_input},
        outputs={"WideResNet/classifier_block/predict_gender/softmax": logits}
    )

    builder.add_meta_graph_and_variables(
        sess=session,
        tags=[
            tf.saved_model.tag_constants.SERVING
        ],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
        },
        saver=saver
    )

    builder.save()

    # We use a built-in TF helper to export variables to constants
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        session,  # The session is used to retrieve the weights
        tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
        ["WideResNet/classifier_block/predict_gender/softmax"]# The output node names are used to select the usefull nodes
    )

    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile("frozen_model.pb", "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))


