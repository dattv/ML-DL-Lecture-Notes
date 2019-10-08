import os

import tensorflow as tf
import numpy as np

data = np.array([10., 11., 12., 13., 14., 15.])


def npy_to_tfrecord(fname, data):
    write = tf.io.TFRecordWriter(fname)
    feature = {}
    feature["data"] = tf.train.Feature(float_list=tf.train.FloatList(value=data))
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    serialized = example.SerializeToString()
    write.write(serialized)
    write.close()


file_name = __file__
file_name = os.path.split(file_name)[1]
file_name = file_name.split(".")[0]
npy_to_tfrecord(file_name + ".tfrecord", data)

dataset = tf.data.TFRecordDataset(file_name + ".tfrecord")


def parse_function(example_proto):
    key_to_features = {'data': tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True)}
    parsed_features = tf.io.parse_single_example(serialized=example_proto, features=key_to_features)
    return parsed_features["data"]


dataset = dataset.map(parse_function)
iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
# array is retrieved as one item
item = iterator.get_next()
print(item)
print(item.numpy())
print(item[2].numpy())
