import os
import pickle

import tensorflow as tf
import numpy as np
import numpy.random as rng

from siamese_model import siamese

# I tempolary reduce the size of image from 105x105x3 to 50x50x3 because my GPU device doesn't have enough memory
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

    # Initial 2 empty arrays for the input image_batch
    pairs = [np.zeros((batch_size, h, w, l)) for l in range(2)]

    # initialize vector fo the targets
    targets = np.zeros((batch_size,))

    # make one half of it "1"s so 2nd half of batch has same class
    targets[batch_size // 2:] = 1

    for i in range(batch_size):
        category = categories[i]
        idx_1 = rng.randint(0, n_examples)
        pairs[0][i, :, :, :] = X[category, idx_1].reshape(w, h, 1)
        idx_2 = rng.randint(0, n_examples)

        # pick images of same class for 1st half, different for 2nd
        if i >= batch_size // 2:
            category_2 = category
        else:
            # add a random number to the category modulo n classes to ensure 2nd image has a different category
            category_2 = (category + rng.randint(1, n_classes)) % n_classes

        pairs[1][i, :, :, :] = X[category_2, idx_2].reshape(w, h, 1)

    return pairs, targets


def generate(bat_size, s="train"):
    """
    A generator for batches, so model.fit_generator can be used
    :param bat_size:
    :param s:
    :return:
    """
    while True:
        pairs, targets = get_batch(bat_size, s)
        yield (pairs, targets)


def make_oneshot_task(N, s="val", language=None):
    """
    Create pairs of test image, support set of testing N way one-shot learning
    :param N:
    :param s:
    :param language:
    :return:
    """
    if s == "train":
        X = Xtrain
        categories = train_classes
    else:
        X = Xtest
        categories = test_classes

    n_classes, n_examples, w, h = X.shape

    indices = rng.randint(0, n_examples, size=(N,))
    if language is not None:
        low, high = categories[language]
        if N > high - low:
            raise ValueError("This languages ({}) has less than {} letters".format(language, N))
        categories = rng.choice(range(low, high), size=(N, ), replace=False)

    else:
        categories = rng.choice(range(n_classes), size=(N,), replace=False)

    true_category = categories[0]
    ex1, ex2 = rng.choice(n_examples, replace=False, size=(2, ))
    test_image = np.asarray([X[true_category, ex1, :, :]]*N).reshape(N, w, h, l)




with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    train_summary_writer = tf.summary.FileWriter(log_dir + "/train", session.graph)
    test_summary_writer = tf.summary.FileWriter(log_dir + "/test")
