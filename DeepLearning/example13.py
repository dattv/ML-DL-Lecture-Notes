# more complexity save and load weights in npz format
import tensorflow as tf
import numpy as np
import os

tf.test.is_gpu_available()


# Create a model
def weight_variables(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variables(shape, name=None):
    initial = tf.constant(0.1, shape=[shape])
    return tf.Variable(initial, name=name)


def conv2d(x, W, name=None):
    return tf.nn.conv2d(input=x,
                        filter=W,
                        strides=[1, 1, 1, 1],
                        padding='SAME')


def max_pool_2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')


def conv_layer(input, shape, name=None):
    W = weight_variables(shape, name=name + "/weights")
    b = bias_variables(shape[3], name=name + "/biases")
    return tf.nn.relu(conv2d(input, W) + b, name=name)


def full_layer(input, size, name=None):
    in_size = int(input.get_shape()[1])
    W = weight_variables([in_size, size], name=name + "/weights")
    b = bias_variables(size, name=name + '/biases')
    return tf.matmul(input, W) + b


x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])
conv1 = conv_layer(x_image, shape=[5, 5, 1, 32], name="conv1")
conv1_pool = max_pool_2d(conv1)

conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64], name="conv2")
conv2_pool = max_pool_2d(conv2)

conv2_flat = tf.reshape(conv2_pool, shape=[-1, 7 * 7 * 64])
full_1 = tf.nn.relu(full_layer(conv2_flat, 1024, name="full_1"))

keep_prop = tf.placeholder(tf.float32)
full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prop)

y_conv = full_layer(full1_drop, 10, name='fully_2')

from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = "./MNIST_data"
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(1.e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for i in range(1000):

        batch = mnist.train.next_batch(124)
        if i % 100 == 0:
            train_accuracy = session.run(accuracy, feed_dict={x: batch[0],
                                                              y_: batch[1],
                                                              keep_prop: 1.})
            print("acc: {}".format(train_accuracy))
        session.run(train_step, feed_dict={x: batch[0],
                                           y_: batch[1],
                                           keep_prop: 0.5})

    data_dict = {}
    idx = 0
    for tensor in tf.get_default_graph().get_collection(tf.GraphKeys.VARIABLES):
        temp_value = session.run(tf.get_default_graph().get_tensor_by_name(tensor.name))
        temp_shape = tensor.shape
        temp_value = np.reshape(temp_value, temp_shape)

        temp_name = tensor.name
        temp_name = temp_name.split("/")
        if temp_name[len(temp_name) - 1].split(":")[0] == "weights":
            data_dict[tensor.name] = temp_value
        if temp_name[len(temp_name) - 1].split(":")[0] == "biases":
            data_dict[tensor.name] = temp_value

# save to cross-platform format
model_file_name = os.path.join(DATA_DIR, "model01")
np.savez(model_file_name, data_dict)

load_model = np.load(model_file_name + ".npz")["arr_0"]

print(load_model[:])