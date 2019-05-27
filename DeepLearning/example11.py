import multiprocessing
import os
import random
import sys
import tarfile
import threading
import urllib.request
import zipfile
import cv2 as cv
from datetime import datetime

import numpy as np
import six
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tqdm import tqdm


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


class ImageCoder(object):
    """
    Helper class which provides Tensorflow image coding utilities
    """

    def __init__(self):
        #
        # Create a single session to run all image coding calls
        #
        self._sess = tf.Session()

        #
        # Initializes function that convert PNG to JPEG data
        #
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        #
        # Initializes function that convert CMYK JPEG data to RGB JPEG data
        #
        self._cmyk_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
        self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

        #
        # Initializes function that decodes RGB JPEG data
        #
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def cmyk_to_rgb(self, image_data):
        return self._sess.run(self._cmyk_to_rgb,
                              feed_dict={self._cmyk_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


class VGG:
    def __init__(self, VGG_URL="http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz",
                 ):
        self._vgg_url = VGG_URL
        self._vgg_file_name = self._vgg_url.split("/")[-1]
        temp_vgg_file_name = self._vgg_file_name
        temp_vgg_file_name = temp_vgg_file_name.split(".")[0]

        self._work_dir = os.getcwd()
        temp_vgg_file_name = os.path.join(self._work_dir, temp_vgg_file_name)
        if os.path.isdir(temp_vgg_file_name) == False:
            os.mkdir(temp_vgg_file_name)

        self._vgg_file_name = os.path.join(temp_vgg_file_name, self._vgg_file_name)

        # Download VGG model from the internet
        if not os.path.exists(self._vgg_file_name):
            with tqdm(unit="B", unit_scale=True, leave=True, miniters=1, desc=VGG_URL.split("/")[-1]) as t:
                self._vgg_file_name, _ = urllib.request.urlretrieve(self._vgg_url, filename=self._vgg_file_name,
                                                                    reporthook=my_hook(t), data=None)

        # Extract the the downloaded file
        if self._vgg_file_name.endswith("gz") == True:
            with tarfile.open(name=self._vgg_file_name) as tar:
                self._vgg_model_file_name = os.path.join(temp_vgg_file_name, tar.getnames()[0])

                for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
                    tar.extract(member=member, path=temp_vgg_file_name)

        # Load VGG parameters
        self.read_VGG_model(file_path=self._vgg_model_file_name)

    def read_VGG_model(self, file_path="", TRAINABLE=False):
        if file_path.endswith("ckpt"):

            reader = pywrap_tensorflow.NewCheckpointReader(file_path)
            var_to_shape_map = reader.get_variable_to_shape_map()

            for name in var_to_shape_map:
                print("tensor: {:30}, size: {}".format(name, var_to_shape_map[name]))

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

    def build_vgg(self, x_input, keep_prob=1.):
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

                self._fc6 = tf.nn.dropout(self._fc6, keep_prob=keep_prob)

            with tf.name_scope("dense7") as scope:
                self._fc7 = tf.nn.conv2d(input=self._fc6,
                                         filter=self._t_weights_7,
                                         strides=[1, 1, 1, 1],
                                         padding='VALID')

                self._fc7 = self._fc7 + self._t_biases_7
                self._fc7 = tf.nn.relu(self._fc7, name='fc7')

                self._fc7 = tf.nn.dropout(self._fc7, keep_prob=keep_prob)

            with tf.name_scope("dense8") as scope:
                self._fc8 = tf.nn.conv2d(input=self._fc7,
                                         filter=self._t_weights_8,
                                         strides=[1, 1, 1, 1],
                                         padding='VALID')
                self._fc8 = self._fc8 + self._t_biases_8
                self._fc8 = tf.nn.softmax(self._fc8, name='fc8')

    def build_VGG_classify(self, x_input, keep_prob=1, n_output=1000):
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

                self._fc6 = tf.nn.dropout(self._fc6, keep_prob=keep_prob)

            with tf.name_scope("dense7") as scope:
                self._fc7 = tf.nn.conv2d(input=self._fc6,
                                         filter=self._t_weights_7,
                                         strides=[1, 1, 1, 1],
                                         padding='VALID')

                self._fc7 = self._fc7 + self._t_biases_7
                self._fc7 = tf.nn.relu(self._fc7, name='fc7')

                self._fc7 = tf.nn.dropout(self._fc7, keep_prob=keep_prob)

            with tf.name_scope("vgg_16") as scope:
                with tf.name_scope("fc8") as scope:
                    with tf.name_scope("new_weights") as scope:
                        self._new_t_weights_8 = tf.Variable(tf.random_normal([1, 1, 4096, int(n_output)],
                                                                             stddev=0.1), name="new_t_weights_8",
                                                            trainable=True)
                    with tf.name_scope("new_biases") as scope:
                        self._new_t_biases_8 = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[n_output]),
                                                           name="new_t_biases_8",
                                                           trainable=True)

            with tf.name_scope("dense8") as scope:
                self._fc8 = tf.nn.conv2d(input=self._fc7,
                                         filter=self._new_t_weights_8,
                                         strides=[1, 1, 1, 1],
                                         padding='VALID')
                self._fc8 = self._fc8 + self._new_t_biases_8
                self._fc8 = tf.nn.softmax(self._fc8, name='fc8')

        return self._fc8


def _process_image_file_batch(coder, thread_index, ranges, name, directory, all_records, num_shards):
    """

    :param coder:
    :param thread_index:
    :param ranges:
    :param name:
    :param directory:
    :param all_records:
    :param num_shards:
    :return:
    """
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)

    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        shard = thread_index * num_shards_per_batch + s
        output_filename = "{}-{:05d}-of-{:05d}".format(name, shard, num_shards)
        output_file = os.path.join(directory, output_filename)

        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            cur_record = all_records[i]
            with tf.gfile.FastGFile(cur_record[0], 'rb') as f:
                cur_img = f.read()

            # cur_img = cv.imread(cur_record[0])
            # try:
            #     cur_img = cv.cvtColor(cur_img, cv.COLOR_BGR2RGB)
            #     cur_img = cur_img.astype(np.float32)
            # except:
            #     # print(cur_img.shape)
            #     print(cur_record)

            temp_cur_img = coder.decode_jpeg(cur_img)
            cur_label = cur_record[1:]
            w, h, c = temp_cur_img.shape

            if not isinstance(w, list):
                w = [w]
            if not isinstance(h, list):
                h = [h]
            if not isinstance(c, list):
                c = [c]
            if isinstance(cur_img, six.string_types):
                cur_img = six.binary_type(cur_img, encoding='utf-8')

            temp_file_name = cur_record[0].encode('utf8')
            if isinstance(temp_file_name, six.string_types):
                temp_file_name = six.binary_type(temp_file_name, encoding='utf-8')

            if not isinstance(cur_label, list):
                cur_label = [int(cur_label[0]), int(cur_label[1])]

            image_format = 'JPEG'
            if isinstance(image_format, six.string_types):
                image_format = six.binary_type(image_format, encoding='utf-8')
            # print("")
            example = tf.train.Example(features=tf.train.Features(
                feature={'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=h)),
                         'image/weight': tf.train.Feature(int64_list=tf.train.Int64List(value=w)),
                         'image/chanels': tf.train.Feature(int64_list=tf.train.Int64List(value=c)),
                         'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
                         'image/label': tf.train.Feature(int64_list=tf.train.Int64List(value=cur_label)),
                         'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[temp_file_name])),
                         'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[cur_img]))}))
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1
            if not counter % 1000:
                print("{} [thread {}]: Processed {} of {} images in thread batch.".format(datetime.now(), thread_index,
                                                                                          counter, num_files_in_thread))

                sys.stdout.flush()

    writer.close()
    print("{} [thread {}]: Wrote {} images to {}".format(datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
    print("{} [thread {}]: Wrote {} images to {} shared".format(datetime.now(), thread_index, counter,
                                                                num_files_in_thread))
    sys.stdout.flush()


VGG16 = VGG()
n_output = 2
x_input = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='x_input')
y_input = tf.placeholder(dtype=tf.float32, shape=[None, n_output], name='y_input')
logits = VGG16.build_VGG_classify(x_input, keep_prob=0.5, n_output=n_output)

# Download data cat and dog from microsoft

cat_dog_url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip"
temp_cat_dog_file = cat_dog_url.split("/")[-1]
temp_cat_dog_folder = temp_cat_dog_file.split(".")[0]
work_dir = os.getcwd()

cat_dog_folder = os.path.join(work_dir, temp_cat_dog_folder)
cat_dog_file = os.path.join(cat_dog_folder, temp_cat_dog_file)

if os.path.isdir(cat_dog_folder) == False:
    os.mkdir(cat_dog_folder)

if os.path.exists(cat_dog_file) == False:
    with tqdm(unit="B", unit_scale=True, leave=True, miniters=1, desc=cat_dog_url.split("/")[-1]) as t:
        file_path, _ = urllib.request.urlretrieve(cat_dog_url, filename=cat_dog_file, reporthook=my_hook(t),
                                                  data=None)

# Extract dog and cat dataset
# if cat_dog_file.endswith("zip"):
#     with zipfile.ZipFile(cat_dog_file) as zip:
#         for member in tqdm(iterable=zip.namelist(), total=len(zip.namelist())):
#             zip.extract(member=member, path=cat_dog_folder)
#             if member.endswith("jpg"):
#                 img = cv.imread(os.path.join(cat_dog_folder, member))
#                 try:
#                     img = cv.resize(img, (224, 224), interpolation=cv.INTER_CUBIC)
#                     cv.imwrite(os.path.join(cat_dog_folder, member), img)
#                 except:
#                     os.remove(os.path.join(cat_dog_folder, member))
#                     print("There are some error with file: {}, so we remove it".format(member))
#             else:
#                 if os.path.isfile(os.path.join(cat_dog_folder, member)):
#                     os.remove(os.path.join(cat_dog_folder, member))
#                     print("remove: ", os.path.join(cat_dog_folder, member))

# Prepare data
# split it into train and test dataset
cat_dog_train_dir = os.path.join(cat_dog_folder, "train")
cat_dog_test_dir = os.path.join(cat_dog_folder, "test")

temp_img_file = os.path.join(os.path.join(cat_dog_folder, "PetImages"), "Cat")
cat_img_names = [[os.path.join(temp_img_file, name), 1, 0]
                 for name in os.listdir(temp_img_file)
                 if os.path.isfile(os.path.join(temp_img_file, name))]

cat_img_names = np.asarray(cat_img_names)
len_cat_img = len(cat_img_names)
print("number of cat imgage is :{}".format(len_cat_img))

temp_img_file = os.path.join(os.path.join(cat_dog_folder, "PetImages"), "Dog")
dog_img_names = [[os.path.join(temp_img_file, name), 0, 1]
                 for name in os.listdir(temp_img_file)
                 if os.path.isfile(os.path.join(temp_img_file, name))]

dog_img_names = np.asarray(dog_img_names)
len_dog_img = len(dog_img_names)
print("number of dog image is:{}".format(len_dog_img))

img_files = np.concatenate((cat_img_names, dog_img_names))
num_nmg_files = len(img_files)

RANDOM_SEED = 180428
shuffled_index = list(range(len(img_files)))
random.seed(RANDOM_SEED)
random.shuffle(shuffled_index)
random.shuffle(shuffled_index)

img_files = [img_files[i][:] for i in shuffled_index]

# 80% for training, 20% for testing
pivot = int(len(img_files) * 0.8)
train_data = img_files[0:pivot][:]
test_data = img_files[pivot:][:]

NUM_THREADS = multiprocessing.cpu_count()

# process train_data
spacing = np.linspace(0, len(train_data), NUM_THREADS + 1).astype(np.int)

ranges = []
threads = []

for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])

print("Launching {} Threads for spacing {}".format(NUM_THREADS, ranges))

coord = tf.train.Coordinator()

coder = ImageCoder()
# ================================================================================================
name = "train"
num_shards = 16
for thread_index in range(len(ranges)):
    # _process_image_file_batch(coder, thread_index, ranges, name, cat_dog_folder, train_data, num_shards)
    args = (coder, thread_index, ranges, name, cat_dog_folder, train_data, num_shards)
    t = threading.Thread(target=_process_image_file_batch, args=args)
    t.start()
    threads.append(t)

coord.join(threads)
print("{} Finish writing all {} image to data set.".format(datetime.now(), len(train_data)))
sys.stdout.flush()
# =================================================================================================
spacing = np.linspace(0, len(test_data), NUM_THREADS + 1).astype(np.int)

ranges = []
threads = []

for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])

print("Launching {} Threads for spacing {}".format(NUM_THREADS, ranges))

coord = tf.train.Coordinator()

coder = ImageCoder()

name = "test"
num_shards = 16
for thread_index in range(len(ranges)):
    args = (coder, thread_index, ranges, name, cat_dog_folder, test_data, num_shards)
    t = threading.Thread(target=_process_image_file_batch, args=args)
    t.start()
    threads.append(t)

coord.join(threads)
print("{} Finish writing all {} image to data set.".format(datetime.now(), len(test_data)))
sys.stdout.flush()

