import os
import pickle

import tensorflow as tf
import numpy as np
import numpy.random as rng

from siamese_model import siamese

input_img1 = tf.placeholder(tf.float32, shape=[None, 50, 50, 3], name="input_img1")
input_img2 = tf.placeholder(tf.float32, shape=[None, 50, 50, 3], name="input_img2")

target = tf.placeholder(tf.float32, shape=[None, 1], name="target")

net = siamese()
net = net.make_model(input_img1, input_img2)

with tf.name_scope("loss") as scope:
    loss = tf.reduce_mean(net - target)

    tf.summary.scalar("loss", loss)

with tf.name_scope("optimiser") as scope:
    optimiser = tf.train.AdamOptimizer(1.e-4).minimize(loss)

with tf.name_scope("accuracy") as scope:
    correct = tf.equal(net, target)


merged_summation = tf.summary.merge_all()

root_path = os.path.dirname(os.path.dirname(__file__))
log_dir = os.path.join(root_path, "siamese_tf")
if os.path.isdir(log_dir) == False:
    os.mkdir(log_dir)
log_dir = os.path.join(log_dir, "log")

train_path = os.path.join(root_path, "siamese_tf")
train_path = os.path.join(train_path, "train.pickle")
with open(train_path, "rb") as f:
    (Xtrain, train_classes) = pickle.load(f)

print(list(train_classes.keys()))

test_path = os.path.join(root_path, "siamese_tf")
test_path = os.path.join(test_path, "val.pickle")
with open(test_path, "rb") as f:
    (Xtest, test_classes) = pickle.load(f)

print(list(test_classes.keys()))

def get_batch(batch_size, s="train"):
    """
    Create batch of n pairs, half same class, half different class

    :param batch_size:
    :param s:
    :return:
    """
    if s == "train":
        X = Xtrain
        categories = train_classes
    else:
        X = Xtest
        categories = test_classes

    n_classes, n_examples, w, h = X.shape

    # randomly sample several classes to use in the batch
    categories = rng.choice(n_classes, size=(batch_size), replace=False)


with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    train_summary_writer = tf.summary.FileWriter(log_dir + "/train", session.graph)
    test_summary_writer = tf.summary.FileWriter(log_dir + "/test")

