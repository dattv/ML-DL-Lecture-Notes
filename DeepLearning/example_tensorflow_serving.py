import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

work_dir = '/tmp'
model_version = 9
training_iteration = 1000
input_size = 784
no_classes = 10
batch_size = 100
total_batches = 200

tf_example = tf.parse_example(tf.placeholder(tf.string, name="tf_example"),
                              {"x": tf.FixedLenFeature(shape=[input_size], dtype=tf.float32)})

x_input = tf.identity(tf_example["x"], name="x")
# x_input = tf.placeholder(tf.float32, shape=[None, input_size])

y_input = tf.placeholder(tf.float32, shape=[None, no_classes])
weights = tf.Variable(tf.random_normal([input_size, no_classes]))
bias = tf.Variable(tf.random_normal([no_classes]))

logits = tf.matmul(x_input, weights) + bias

softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_input,
                                                                logits=logits)

loss_operation = tf.reduce_mean(softmax_cross_entropy)
optimiser = tf.train.GradientDescentOptimizer(0.01).minimize(loss_operation)

# train
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for batch_no in range(total_batches):
        mnist_batch = mnist.train.next_batch(batch_size)

        _, loss_value = session.run([optimiser, loss_operation], feed_dict={x_input: mnist_batch[0],
                                                                            y_input: mnist_batch[1]})

        if batch_no % 10 == 0:
            print(loss_value)

    signature_def = (tf.saved_model.signature_def_utils.build_signature_def(inputs={"x": tf.saved_model.utils.build_tensor_info(x_input)},
                                                                            outputs={"y": tf.saved_model.utils.build_tensor_info(y_input)},
                                                                            method_name="tensorflow/serving/predict"))

    model_path = os.path.join(work_dir, str(model_version))

