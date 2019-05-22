import tensorflow as tf
import numpy as np
import os

from tensorflow.examples.tutorials.mnist import input_data

input_size = 28*28
x_input = tf.placeholder(tf.float32, shape=[None, input_size])
y_input = tf.placeholder(tf.float32, shape=[None, input_size])


def dense_layer(input_layer, units, activation=tf.nn.tanh):
    layer = tf.layers.dense(
        inputs=input_layer,
        units=units,
        activation=activation
    )
    # tf.add_variable_summary(layer, 'dense')
    return layer

layer_1 = dense_layer(x_input, 500)
layer_2 = dense_layer(layer_1, 250)
layer_3 = dense_layer(layer_2, 50)
layer_4 = dense_layer(layer_3, 250)
layer_5 = dense_layer(layer_4, 500)
layer_6 = dense_layer(layer_5, 784)

with tf.name_scope("loss"):
    soft_max_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_input,
                                                                     logits=layer_6)
    loss_operation = tf.reduce_mean(soft_max_cross_entropy, name='loss')
    tf.summary.scalar('loss', loss_operation)

with tf.name_scope('optimiser'):
    optimiser = tf.train.AdamOptimizer().minimize(loss_operation)


x_input_reshaped = tf.reshape(x_input, [-1, 28, 28, 1])
tf.summary.image("noisy_images", x_input_reshaped)

y_input_reshaped = tf.reshape(y_input, [-1, 28, 28, 1])
tf.summary.image("original_images", y_input_reshaped)

layer_6_reshaped = tf.reshape(layer_6, [-1, 28, 28, 1])
tf.summary.image("reconstructed_images", layer_6_reshaped)

merged_summary_operation = tf.summary.merge_all()

session =tf.Session()
session.run(tf.global_variables_initializer())

train_summary_writer = tf.summary.FileWriter('./tmp/train', session.graph)
total_batch = 1000
mnist_data = input_data.read_data_sets("MNIST_data", one_hot=True)
batch_size = 128
for batch_no in range(total_batch):
    mnist_batch = mnist_data.train.next_batch(batch_size)
    train_images, _ = mnist_batch[0], mnist_batch[1]
    train_images_noise = train_images + 0.2 * np.random.normal(size=train_images.shape)
    train_images_noise = np.clip(train_images_noise, 0., 1.)
    _, merged_summary = session.run([optimiser, merged_summary_operation],
                                    feed_dict={
                                        x_input: train_images_noise,
                                        y_input: train_images,
                                    })
    train_summary_writer.add_summary(merged_summary, batch_no)
    print("epoch: {}".format(batch_no))


