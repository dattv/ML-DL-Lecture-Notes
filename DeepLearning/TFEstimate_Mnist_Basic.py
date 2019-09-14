from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
import numpy as np
from pathlib import Path
import logging
import sys
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import device_lib
import tensorflow_estimator as TFE

print(device_lib.list_local_devices())

file_name = __file__
file_name = os.path.split(file_name)[1]
file_name = file_name.split(".")[0]
model_file = "./" + file_name + "_model"
Path(model_file).mkdir(exist_ok=True)
tf.logging.set_verbosity(logging.INFO)
handlers = [logging.FileHandler(model_file + '/train.log'),
            logging.StreamHandler(sys.stdout)
            ]
logging.getLogger('tensorflow').handlers = handlers


def cnn_model_fn(features, labels, mode):
    with tf.variable_scope("cnn_model_fn") as scope:
        input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

        conv1 = tf.layers.conv2d(inputs=input_layer, filters=64, kernel_size=[3, 3], padding="same",
                                 activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

        pool2_flat = tf.reshape(conv3, [-1, 7 * 7 * 128])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

        dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == TFE.estimator.ModeKeys.TRAIN)

        logits = tf.layers.dense(inputs=dropout, units=10)

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels,
                                        predictions=predictions["classes"])
    }

    if mode == TFE.estimator.ModeKeys.PREDICT:
        return TFE.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == TFE.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())

        return TFE.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    return TFE.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)


# Load training and eval data
((train_data, train_labels), (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data("./MNIST_data")

train_data = train_data / np.float32(255)
train_labels = train_labels.astype(np.int32)  # not required

eval_data = eval_data / np.float32(255)
eval_labels = eval_labels.astype(np.int32)  # not required

# Create the Estimator
mnist_classifier = TFE.estimator.Estimator(model_fn=cnn_model_fn, model_dir=model_file + "/mnist_convnet_model")

# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}

logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

train_input_fn = TFE.estimator.inputs.numpy_input_fn(x={"x": train_data},
                                                     y=train_labels,
                                                     batch_size=350,
                                                     num_epochs=1,
                                                     shuffle=True,
                                                     num_threads=8)

eval_input_fn = TFE.estimator.inputs.numpy_input_fn(x={"x": eval_data},
                                                    y=eval_labels,
                                                    num_epochs=1,
                                                    shuffle=False,
                                                    num_threads=8)


for _ in range(20):
    mnist_classifier.train(input_fn=train_input_fn)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)

    print(eval_results)
