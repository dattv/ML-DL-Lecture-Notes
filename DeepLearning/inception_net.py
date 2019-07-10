import tensorflow as tf
from urllib.request import urlretrieve

from tqdm import tqdm
import tarfile

import os

inception_net_url = "http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz"


def my_hook(t):
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


root_path = os.path.dirname(os.path.dirname(__file__))

DeepLearning_path = os.path.join(root_path, "DeepLearning")

inception_file_name = inception_net_url.split("/")
inception_file_name = inception_file_name[len(inception_file_name) - 1]
name = inception_file_name.split(".")[0]

inception_folder = os.path.join(DeepLearning_path, name)
if os.path.isdir(inception_folder) == False:
    os.mkdir(inception_folder)

inception_full_file_path = os.path.join(inception_folder, inception_file_name)

if os.path.isfile(inception_full_file_path) == False:
    print("Downloading {}".format(inception_file_name))
    with tqdm(unit="B", unit_scale=True, leave=True, miniters=1, desc=inception_net_url.split("/")[-1]) as t:
        urlretrieve(inception_net_url, filename=inception_full_file_path, reporthook=my_hook(t), data=None)

tar = tarfile.open(inception_full_file_path)
tar.extractall(path=inception_folder)
tar.close()

from tensorflow.python import pywrap_tensorflow

reader = pywrap_tensorflow.NewCheckpointReader(inception_folder + "/inception_v1.ckpt")
var_to_shape_map = reader.get_variable_to_shape_map()

for name in var_to_shape_map.keys():
    print(name, var_to_shape_map[name])


class inception(object):
    def __init__(self):
        super(inception, self).__init__()

        print("inception")

        # initial value
        self.loss = None
        self.accuracy = None
        self.summary = []

    def get_summary(self):
        return self.summary

    def get_convolution(self, layer_name="conv", inputs=None, out_chanels=1, kernel_size=1, strides=1, padding="SAME"):
        in_chanels = inputs.shape[3]
        with tf.variable_scope(layer_name) as scope:
            self.scope[layer_name] = scope

            w = tf.get_variable("weights",
                                trainable=True,
                                shape=[kernel_size, kernel_size, in_chanels, out_chanels],
                                initializer=tf.contrib.layers.xavier_initializer())

            b = tf.get_variable("biases",
                                trainable=True,
                                shape=[out_chanels],
                                initializer=tf.constant_initializer(0.1))

            inputs = tf.nn.conv2d(input=inputs,
                                  filter=w,
                                  strides=[1, strides, strides, 1],
                                  padding=padding,
                                  name="conv")

            inputs = tf.nn.bias_add(inputs,
                                    b,
                                    name="biases_add")

            inputs = tf.nn.relu(inputs, name="relue")

        return inputs

    def get_max_pool(self, layer_name="pool", inputs=None, pool_size=1, strides=1, padding="SAME"):
        with tf.variable_scope(layer_name) as scope:
            self.scope[layer_name] = scope

            inputs = tf.nn.max_pool(inputs,
                                    ksize=[1, pool_size, pool_size, 1],
                                    strides=[1, strides, strides, 1],
                                    padding="SAME",
                                    name="layer_name")

        return inputs

    def get_avg_pool(self, layer_name="avg_pool", inputs=None, pool_size=2, strides=2, padding="SAME"):
        with tf.variable_scope(layer_name) as scope:
            return tf.nn.avg_pool(inputs,
                                  ksize=[1, pool_size, pool_size, 1],
                                  strides=[1, strides, strides, 1],
                                  padding=padding,
                                  name="layer_name")

    def concate(self, layer_name, inputs):
        with tf.name_scope(layer_name) as scope:
            one_by_one = inputs[0]
            three_by_three = inputs[1]
            five_by_five = inputs[2]
            pooling = inputs[3]
            return tf.concat([one_by_one, three_by_three, five_by_five, pooling], axis=3)

    def dropout(self, layer_name, inputs, keep_prop):
        # dropout_rate = 1 - keep_prop
        with tf.variable_scope(layer_name) as scope:
            return tf.nn.dropout(layer_name, inputs, keep_prob=keep_prop)



