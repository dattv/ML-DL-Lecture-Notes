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


class inception(object):
    def __init__(self, tensor_x, tensor_y):
        super(inception, self).__init__()

        print("inception")

        # initial value
        self.loss = None
        self.accuracy = None
        self.summary = []
        self.scope = {}

        self.net = self.inference(tensor_x)

    def summary(self):
        return self.summary

    def convolution(self, layer_name="conv", inputs=None, out_chanels=1, kernel_size=1, strides=1, padding="SAME"):
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

    def max_pool(self, layer_name="pool", inputs=None, pool_size=1, strides=1, padding="SAME"):
        with tf.variable_scope(layer_name) as scope:
            self.scope[layer_name] = scope

            inputs = tf.nn.max_pool(inputs,
                                    ksize=[1, pool_size, pool_size, 1],
                                    strides=[1, strides, strides, 1],
                                    padding="SAME",
                                    name="layer_name")

        return inputs

    def avg_pool(self, layer_name="avg_pool", inputs=None, pool_size=2, strides=2, padding="SAME"):
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

    def bn(self, layer_name, inputs, epsilon=1e-3):
        with tf.name_scope(layer_name):
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
            inputs = tf.nn.batch_normalization(inputs, mean=batch_mean, variance=batch_var, offset=None,
                                               scale=None, variance_epsilon=epsilon)
        return inputs

    def fc(self, layer_name, inputs, out_nodes):
        shape = inputs.get_shape()
        if len(shape) == 4:
            size = shape[1].value * shape[2].value * shape[3].value

        else:
            size = shape[1].value

        with tf.variable_scope(layer_name) as scope:
            self.scope[layer_name] = scope
            w = tf.get_variable("weights",
                                shape=[size, out_nodes],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)
            b = tf.get_variable("biases",
                                shape=[out_nodes],
                                initializer=tf.constant_initializer(0.),
                                dtype=tf.float32)

            flat_x = tf.reshape(inputs, [-1, size])
            inputs = tf.nn.bias_add(tf.matmul(flat_x, w), b)
            inputs = tf.nn.relu(inputs)
        return inputs

    def lrn(self, layer_name, inputs, depth_radius=5, alpha=0.0001, beta=0.75):
        with tf.name_scope(layer_name):
            return tf.nn.local_response_normalization(name='pool1_norm1', input=inputs, depth_radius=depth_radius,
                                                      alpha=alpha, beta=beta)

    def inference(self, tensor_x):
        # conv 7x7+2(s)
        with tf.variable_scope("InceptionV1") as scope:
            Conv2d_1a_7x7 = self.convolution(layer_name="Conv2d_1a_7x7",
                                             inputs=tensor_x,
                                             out_chanels=64,
                                             kernel_size=7,
                                             strides=2,
                                             padding="SAME")

            # MaxPool3x3+2(S)
            pool1_3x3_s2 = self.max_pool(layer_name="pool1_3x3_s2",
                                         inputs=Conv2d_1a_7x7,
                                         pool_size=3,
                                         strides=2,
                                         padding="SAME")

            # LocalRespNorm
            pool1_norm1 = self.lrn(layer_name="pool1_norm1",
                                   inputs=pool1_3x3_s2)

            # Conv1x1+1(V)
            conv2_3x3_reduce = self.convolution(layer_name="conv2_3x3_reduce",
                                                inputs=pool1_norm1,
                                                out_chanels=64,
                                                kernel_size=1,
                                                strides=1,
                                                padding="SAME")

            # Conv3x3+1(S)
            conv2_3x3 = self.convolution(layer_name="conv2_3x3",
                                         inputs=conv2_3x3_reduce,
                                         out_chanels=192,
                                         kernel_size=3,
                                         strides=1,
                                         padding="SAME")

            conv2_norm2 = self.lrn(layer_name="conv2_norm2",
                                   inputs=conv2_3x3)

            pool2_3x3_s2 = self.max_pool(layer_name="pool2_3x3_s2",
                                         inputs=conv2_norm2,
                                         pool_size=3,
                                         strides=2,
                                         padding="SAME")

            # =================================================
            # inception_3a_1x1 =
            # =================================================

        return None


def main():
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

    LOG_DIR = os.path.join(inception_folder, "log")

    from tensorflow.python import pywrap_tensorflow

    reader = pywrap_tensorflow.NewCheckpointReader(inception_folder + "/inception_v1.ckpt")
    var_to_shape_map = reader.get_variable_to_shape_map()

    for key in var_to_shape_map:
        print("tensor_name: ", key)
        if key == "InceptionV1/Conv2d_2b_1x1/weights":
            print(reader.get_tensor(key).shape)
    #     if key == "InceptionV1/Conv2d_1a_7x7/BatchNorm/moving_mean":
    #         print(reader.get_tensor(key).shape)
    #     if key == "InceptionV1/Conv2d_1a_7x7/BatchNorm/moving_variance":
    #         print(reader.get_tensor(key).shape)
    #     if key == "InceptionV1/Conv2d_1a_7x7/BatchNorm/beta":
    #         print(reader.get_tensor(key).shape)
    #
    #     if key == "InceptionV1/Mixed_3c/Branch_1/Conv2d_0a_1x1/BatchNorm/beta":
    #         print(reader.get_tensor(key))

    # ===================================================================================================================
    x_tensor = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="x")
    y_tensor = tf.placeholder(tf.float32, shape=[None, 1000], name="y")

    model = inception(x_tensor, y_tensor)

    merged_summary_operation = tf.summary.merge_all()

    # with tf.Session() as session:
    #     train_summary_writer = tf.summary.FileWriter(LOG_DIR + "/train", session.graph)
    #     test_summary_writer = tf.summary.FileWriter(LOG_DIR + "/test")
if __name__ == '__main__':
    main()
