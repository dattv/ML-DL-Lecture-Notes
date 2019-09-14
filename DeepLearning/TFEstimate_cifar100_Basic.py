import logging
import os
import shutil
from pathlib import Path
import cv2
import numpy as np
import sys
import tensorflow_estimator as TFE
import tensorflow as tf
from DeepLearning.wide_resnet import WideResNet
from DeepLearning.Dataset.cifar100 import cifar100, cifar10
from DeepLearning.cifar10_model import cifar10_inference
from DeepLearning.mnist_model import mnist_inference
from DeepLearning.vgg16_model import VGG16_inference

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

ROOT_PATH = os.getcwd()
print(ROOT_PATH)

# Load training and eval data
dataset = cifar100()
(train_data, train_labels), (eval_data, eval_labels) = dataset.get_data()
IMAGE_HEIGHT = dataset._img_size
IMAGE_WIDTH = dataset._img_size
IMAGE_DEPTH = dataset._img_depth
NUM_CLASSES = dataset._nb_class

def preprocess_image(image, is_training=False):
    """Preprocess a single image of layout [height, width, depth]."""
    if is_training:
        # Resize the image to add four extra pixels on each side.
        image = tf.image.resize_image_with_crop_or_pad(
            image, IMAGE_HEIGHT + 8, IMAGE_WIDTH + 8)

        # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
        image = tf.random_crop(image, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

    # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_standardization(image)
    return image

img = preprocess_image(train_data[0], is_training=True)



def cnn_model_fn(features, labels, mode):
    images = features["x"]
    with tf.variable_scope("model") as scope:
        input_tensor = tf.reshape(images, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH], name="input")
        # logits = cifar10_inference(input_tensor, mode, NUM_CLASSES)
        logits = VGG16_inference(input_tensor, mode, NUM_CLASSES)

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

    if mode == TFE.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=1.e-3).minimize(loss, global_step=global_step)

        return TFE.estimator.EstimatorSpec(mode, loss=loss, train_op=optimizer)

    if mode == TFE.estimator.ModeKeys.EVAL:
        return TFE.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

    if mode == TFE.estimator.ModeKeys.PREDICT:
        export_outputs = {'predictions': TFE.estimator.export.PredictOutput(predictions)}
        return TFE.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)


class FLAGS():
    pass


FLAGS.batch_size = 128
FLAGS.max_steps = None
FLAGS.nb_epoch = 50
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
        shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH], dtype=tf.float32)}
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
print("MAX_STEP: {}".format(int(FLAGS.nb_epoch * len(train_data) // FLAGS.batch_size)))
