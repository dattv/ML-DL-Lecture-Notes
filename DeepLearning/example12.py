# save and load Deep learning model
import os

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = os.getcwd()
DATA_DIR = os.path.join(DATA_DIR, "MNIST_data")

NUM_STEPS = 1000
MINIBATCH_SIZE = 100

data = input_data.read_data_sets(DATA_DIR, one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))

y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred,
                                                                       labels=y_true))

gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(NUM_STEPS):
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
        sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})
    # Test
    ans = sess.run(accuracy, feed_dict={x: data.test.images,
                                        y_true: data.test.labels})
    print("Accuracy: {:.4}%".format(ans * 100))

    # save weights to numpy format
    weights = sess.run(W)

    weights_file_name = os.path.join(DATA_DIR, 'weight_storage')
    np.savez(weights_file_name, weights)

#load weights was saved in numpy format
loaded_w = np.load(weights_file_name + ".npz")
param_values = [loaded_w['arr_%d' %i] for i in range(len(loaded_w.files))]
param_values = np.asarray(param_values)
param_values = np.reshape(param_values, [784, 10])

x = tf.placeholder(tf.float32, [None, 784], name='x')
W = tf.Variable(tf.zeros([784, 10]))
y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred,
                                                                       labels=y_true))
gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_mask = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

with tf.Session() as session:
    session.run(W.assign(param_values))
    acc = session.run(accuracy, feed_dict={x: data.test.images,
                                           y_true: data.test.labels})

    print("Accuracy: {}".format(acc))
