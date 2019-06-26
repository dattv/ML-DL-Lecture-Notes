import os

import numpy
from numpy.random import rand
import tensorflow as tf
from matplotlib import pyplot as plt


# Definition of simple function
def calculate(x):
    return x * x

def generate_output(a, b, n):
    # define
    inputs = numpy.linspace(a, b, n)
    outputs = [calculate(i) for i in inputs]

    return inputs, outputs


def generate_sample(a, b, n):
    inputs = a + numpy.random.rand(n) * (b - a)
    outputs = [calculate(x) for x in inputs]
    return inputs, outputs

# define the standalone discriminator model
def define_discriminator(input_tensor):
    n_in = input_tensor.shape[1]

    with tf.name_scope("hidden_layer") as scope:
        with tf.name_scope("weights") as scope:
            w_h = tf.Variable(tf.truncated_normal([int(n_in), 25], stddev=0.1), name="weights_h")

        with tf.name_scope("biases") as scope:
            bias_h = tf.Variable(tf.constant(0.1, shape=[25]), name="biases_h")

        fully_h = tf.matmul(input_tensor, w_h) + bias_h
        fully_h = tf.nn.relu(fully_h, name="fully_h")

    with tf.name_scope("output_layer") as scope:
        with tf.name_scope("weights") as scope:
            w_out = tf.Variable(tf.truncated_normal([25, 1], stddev=0.1), name="weights_out")

        with tf.name_scope("biases") as scope:
            bias_out = tf.Variable(tf.constant(0.1, shape=[1]), name="biases_out")

        logit = tf.matmul(fully_h, w_out) + bias_out
        logit = tf.nn.sigmoid(logit, name="output")

    return logit

root_path = os.path.dirname(os.path.dirname(__file__))
deep_learning_path = os.path.join(root_path, "DeepLearning")
log_dir = os.path.join(deep_learning_path, "tmp")

temp_file = __file__.split("/")
temp_file = temp_file[len(temp_file) - 1]
temp_file = temp_file.split(".")[0]

log_dir = os.path.join(log_dir, temp_file)
if os.path.isdir(log_dir) == False:
    os.mkdir(log_dir)

if __name__ == '__main__':
    inputs, outputs = generate_sample(-0.5, 0.5, 100)

    input_tensor = tf.placeholder(tf.float32, shape=[None, 1], name="input_tensor")
    target_tensor = tf.placeholder(tf.float32, shape=[None, 1], name="target_tensor")

    logits = define_discriminator(input_tensor)

    with tf.name_scope("loss") as scope:
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=target_tensor,
                                                       logits=logits)
        tf.summary.scalar("loss", loss)

    with tf.name_scope("optimization") as scope:
        optimiser = tf.train.AdamOptimizer(0.01).minimize(loss)

    merged_summary_operation = tf.summary.merge_all()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())





