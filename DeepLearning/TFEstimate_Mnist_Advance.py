from __future__ import print_function, division, absolute_import, unicode_literals

import logging
import os
import shutil
from pathlib import Path
import numpy as np
import sys
import tensorflow_estimator as TFE
import tensorflow as tf

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

IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
IMAGE_DEPTH = 1
NUM_CLASSES = 10


def inference(images, mode):
    input_layer = tf.reshape(images, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES])

    conv1 = tf.layers.conv2d(inputs=input_layer, filters=64, kernel_size=[3, 3], padding="same",
                             activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

    pool2_flat = tf.reshape(conv3, [-1, 7 * 7 * 128])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == TFE.estimator.ModeKeys.TRAIN)

    output = tf.layers.dense(inputs=dropout, units=10)

    return output


def cnn_model_fn(features, labels, mode):
    images = features["x"]
    with tf.variable_scope("model") as scope:
        logits = inference(images, mode)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }


    if mode in (TFE.estimator.ModeKeys.TRAIN, TFE.estimator.ModeKeys.EVAL):

        global_step = tf.train.get_global_step()
        labels_indices = tf.argmax(labels, axis=1)
        eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels_indices,
                                                           predictions=predictions["classes"])}

        loss = tf.losses.softmax_cross_entropy(labels, logits)
        tf.summary.scalar('accuracy', eval_metric_ops["accuracy"][1])

    if mode == TFE.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=1.e-3).minimize(loss, global_step=global_step)

        return TFE.estimator.EstimatorSpec(mode, loss=loss, train_op=optimizer, eval_metric_ops=eval_metric_ops)

    if mode == TFE.estimator.ModeKeys.EVAL:
        return TFE.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

    if mode == TFE.estimator.ModeKeys.PREDICT:

        export_outputs = {'predictions': TFE.estimator.export.PredictOutput(predictions)}
        return TFE.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)


# Load training and eval data
((train_data, train_labels), (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data("./MNIST_data")

train_data = train_data / np.float32(255)
train_labels = train_labels.astype(np.int32)  # not required
train_labels_one_hot = np.zeros((len(train_labels), 10))
train_labels_one_hot[np.arange(len(train_labels)), train_labels] = 1
train_labels = train_labels_one_hot

eval_data = eval_data / np.float32(255)
eval_labels = eval_labels.astype(np.int32)  # not required
eval_labels_one_hot = np.zeros((len(eval_labels), 10))
eval_labels_one_hot[np.arange(len(eval_labels)), eval_labels] = 1
eval_labels = eval_labels_one_hot

class FLAGS():
    pass


FLAGS.batch_size = 200
FLAGS.max_steps = None
FLAGS.nb_epoch = 10
FLAGS.eval_steps = int(len(eval_data) // FLAGS.batch_size)
FLAGS.save_checkpoints_steps = int(len(train_data) // FLAGS.batch_size)
FLAGS.tf_random_seed = 19851211
FLAGS.model_name = model_file
FLAGS.use_checkpoint = False

run_config = TFE.estimator.RunConfig(save_checkpoints_steps=FLAGS.save_checkpoints_steps,
                                     tf_random_seed=FLAGS.tf_random_seed,
                                     model_dir=FLAGS.model_name
                                     )

estimator = TFE.estimator.Estimator(model_fn=cnn_model_fn, config=run_config)
# Set up logging for predictions
tensors_to_log = {"accuracy": "accuracy"}

logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)


def preprocess_image(image, is_training=False):
    """Preprocess a single image of layout [height, width, depth]."""
    if is_training:
        # Resize the image to add four extra pixels on each side.
        image = tf.image.resize_image_with_crop_or_pad(
            image, IMAGE_HEIGHT + 8, IMAGE_WIDTH + 8)

        # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
        image = tf.random_crop(image, [IMAGE_HEIGHT, IMAGE_WIDTH, 1])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

    # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_standardization(image)
    return image


def serving_input_fn():
    receiver_tensor = {'x': tf.placeholder(
        shape=[None, 28, 28, 1], dtype=tf.float32)}
    features = {'x': tf.map_fn(preprocess_image, receiver_tensor['x'])}

    return TFE.estimator.export.ServingInputReceiver(features, receiver_tensor)


# There is another Exporter named FinalExporter
exporter = TFE.estimator.LatestExporter(name='Servo',
                                        serving_input_receiver_fn=serving_input_fn,
                                        assets_extra=None,
                                        as_text=False,
                                        exports_to_keep=5)

train_input_fn = TFE.estimator.inputs.numpy_input_fn(x={"x": train_data},
                                                     y=train_labels,
                                                     batch_size=FLAGS.batch_size,
                                                     num_epochs=None,
                                                     shuffle=True,
                                                     num_threads=8)
train_spec = TFE.estimator.TrainSpec(input_fn=train_input_fn,
                                     max_steps=int(FLAGS.nb_epoch * len(train_data) // FLAGS.batch_size))

eval_input_fn = TFE.estimator.inputs.numpy_input_fn(x={"x": eval_data},
                                                    y=eval_labels,
                                                    num_epochs=1,
                                                    shuffle=False,
                                                    num_threads=8)
eval_spec = TFE.estimator.EvalSpec(input_fn=eval_input_fn,
                                   steps=FLAGS.eval_steps,
                                   exporters=exporter,
                                   throttle_secs=1)

if not FLAGS.use_checkpoint:
    print("Removing model dir")
    shutil.rmtree(model_file, ignore_errors=True)

TFE.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

