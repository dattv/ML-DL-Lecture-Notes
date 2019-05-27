from __future__ import print_function
import os
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist
import numpy as np

work_dir = os.getcwd()
mnist_dir = os.path.join(work_dir, "mnist")

if os.path.isdir(mnist_dir) == False:
    os.mkdir(mnist_dir)

save_dir = mnist_dir
datasets = mnist.read_data_sets(save_dir,
                               dtype=tf.uint8,
                               reshape=False,
                               validation_size=1000)

data_splits = ["train", "test", "validation"]

for d in range(len(data_splits)):
    print("saving: {}".format(data_splits[d]))
    data_set = datasets[d]

    file_name = os.path.join(save_dir, data_splits[d] + ".tfrecords")


